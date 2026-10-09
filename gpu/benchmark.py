import argparse
import csv
import json
import os
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from myconv import ConvModel

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def wait_output(y):
    if isinstance(y, torch.Tensor):
        torch.cuda.synchronize()
    else:
        y.block_until_ready()
    return y


def trace_times(path):
    events = json.loads(path.read_text())["traceEvents"]

    gpu = [
        e for e in events
        if e.get("ph") == "X"
        and e.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}
    ]

    spans = []
    sums = []

    for i in range(5):
        marker = next(
            e for e in events
            if e.get("name") == f"iteration_{i}"
        )

        start = marker["ts"]
        end = start + marker["dur"]

        selected = [
            e for e in gpu
            if e["ts"] >= start and e["ts"] + e["dur"] <= end
        ]

        kernels = [e for e in selected if e["cat"] == "kernel"]

        if not kernels:
            print(f"No GPU kernels captured: {path.name}")
            return None, None

        spans.append(
            (
                max(e["ts"] + e["dur"] for e in selected)
                - min(e["ts"] for e in selected)
            ) / 1000
        )

        sums.append(
            sum(e["dur"] for e in kernels) / 1000
        )

    return statistics.median(spans), statistics.median(sums)


def profile_run(run, path):
    for _ in range(3):
        wait_output(run())

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
    ) as prof:
        for i in range(5):
            with record_function(f"iteration_{i}"):
                wait_output(run())

    prof.export_chrome_trace(str(path))

    return trace_times(path)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "backend",
        choices=["eager", "inductor", "jax", "cuda"],
    )

    backend = parser.parse_args().backend
    RESULTS.mkdir(exist_ok=True)

    if backend == "inductor":
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = tempfile.mkdtemp(
            prefix="mp2_inductor_"
        )
        os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(
            prefix="mp2_triton_"
        )

    torch.backends.cuda.matmul.allow_tf32 = False
    build_load_s = None

    if backend == "cuda":
        from torch.utils.cpp_extension import load

        build_dir = tempfile.mkdtemp(prefix="mp2_cuda_")

        start = time.perf_counter()

        extension = load(
            name="mp2_benchmark_cuda",
            sources=[str(ROOT / "myconv_kernel.cu")],
            build_directory=build_dir,
            verbose=True,
        )

        build_load_s = time.perf_counter() - start

    if backend == "jax":
        import jax
        import jax.numpy as jnp
        from myconv_jax import conv2d_manual_jax

        assert jax.default_backend() == "gpu"

    rows = []

    for H in (16, 32, 48):
        for K in (3, 5):
            torch.manual_seed(0)

            N, C, C_out = 2, 3, 8
            stride, padding = 1, K // 2

            model = ConvModel(
                H, H, C, C_out, K, stride, padding
            )

            model.eval().requires_grad_(False)

            x_cpu = torch.randn(N, C, H, H)

            reference = F.conv2d(
                x_cpu,
                model.weight,
                model.bias,
                stride=stride,
                padding=padding,
            ).numpy()

            tag = f"{backend}_H{H}_K{K}"
            lower_s = compile_s = first_call_s = None

            if backend == "jax":
                x = jnp.array(x_cpu.numpy())
                weight = jnp.array(model.weight.numpy())
                bias = jnp.array(model.bias.numpy())

                for a in (x, weight, bias):
                    a.block_until_ready()

                start = time.perf_counter()

                lowered = jax.jit(
                    conv2d_manual_jax,
                    static_argnames=("stride", "padding"),
                ).lower(
                    x,
                    weight,
                    bias,
                    stride=stride,
                    padding=padding,
                )

                lower_s = time.perf_counter() - start

                start = time.perf_counter()
                compiled = lowered.compile()
                compile_s = time.perf_counter() - start

                run = lambda: compiled(x, weight, bias)

            else:
                model = model.cuda()
                x = x_cpu.cuda()
                torch.cuda.synchronize()

                if backend == "eager":
                    run = lambda: model(x)

                elif backend == "inductor":
                    torch._dynamo.reset()

                    compiled = torch.compile(
                        model, backend="inductor"
                    )

                    run = lambda: compiled(x)

                    start = time.perf_counter()
                    wait_output(run())
                    first_call_s = time.perf_counter() - start

                    from torch._dynamo.utils import compile_times

                    names, values = compile_times(
                        repr="csv", aggregate=True
                    )

                    timings = dict(zip(names, values))

                    if "_compile.compile_inner" in timings:
                        compile_s = float(
                            timings["_compile.compile_inner"]
                        )

                    (RESULTS / f"{tag}_compile.txt").write_text(
                        compile_times()
                    )

                else:
                    run = lambda: (
                        extension.conv_cuda(
                            x, model.weight, stride, padding
                        )
                        + model.bias.reshape(1, -1, 1, 1)
                    )

            y = wait_output(run())

            actual = (
                y.cpu().numpy()
                if isinstance(y, torch.Tensor)
                else np.asarray(y)
            )

            np.testing.assert_allclose(
                actual, reference, rtol=1e-4, atol=1e-4
            )

            trace_path = RESULTS / f"{tag}.json"

            gpu_span_ms, kernel_sum_ms = profile_run(
                run, trace_path
            )

            row = {
                "backend": backend,
                "H": H,
                "K": K,
                "lower_s": lower_s,
                "compile_s": compile_s,
                "build_load_s": build_load_s,
                "first_call_s": first_call_s,
                "gpu_span_ms": gpu_span_ms,
                "kernel_sum_ms": kernel_sum_ms,
            }

            rows.append(row)
            print(row)

    with (RESULTS / f"{backend}.csv").open(
        "w", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()