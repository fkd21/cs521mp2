#include <iostream>
#include <cstdlib>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// example
#define TILE_H 8
#define TILE_W 8
#define TILE_C 8

// Kernel declaration
__global__ void gemm_gpu_o4_kernel(
    const float* __restrict__ x,       // input: N x C x H x W
    const float* __restrict__ w,       // weights: C_out x C_in x KH x KW
    float* __restrict__ out,           // output: N x C x H x W
    int N, int C_in, int H, int W,
    int C_out, int KH, int KW,
    int stride, int pad,
    int out_h, int out_w
) {
    extern __shared__ float shmem[];  // shared memory for partial sums

    // TO DO : Tiled matrix multiplication by using shmem
    float* sh_w = shmem;
    float* sh_x = shmem + TILE_C;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    int thread_count = blockDim.x * blockDim.y;

    int batch = blockIdx.z;
    int oh = blockIdx.y * blockDim.y + ty;
    int ow = blockIdx.x * blockDim.x + tx;

    int K = C_in * KH * KW;

    for (int co = 0; co < C_out; ++co) {
        float acc = 0.0f;
        for (int start = 0; start < K; start += TILE_C) {
            for (int k = tid; k < TILE_C; k += thread_count) {
                int weight_k = start + k;
                if (weight_k < K) {
                    sh_w[k] = w[co * K + weight_k];
                } else {
                    sh_w[k] = 0.0f;
                }
            }

            for (int k = 0; k < TILE_C; ++k) {
                int input_k = start + k;
                float value = 0.0f;
                if (oh < out_h && ow < out_w && input_k < K) {
                    int ci = input_k / (KH * KW);
                    int kh = (input_k / KW) % KH;
                    int kw = input_k % KW;
                    int ih = oh * stride + kh - pad;
                    int iw = ow * stride + kw - pad;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                        value = x[((batch * C_in + ci) * H + ih) * W + iw];
                    }
                }
                sh_x[k * thread_count + tid] = value;
            }

            __syncthreads();

            for (int k = 0; k < TILE_C; ++k) {
                acc += sh_w[k] * sh_x[k * thread_count + tid];
            }

            __syncthreads();
        }

        if (oh < out_h && ow < out_w) {
            out[((batch * C_out + co) * out_h + oh) * out_w + ow] = acc;
        }
    }
}

// Function for Python binding
torch::Tensor conv_cuda(torch::Tensor x, torch::Tensor w,
                          int stride, int pad) {
    int N = x.size(0);
    int C_in = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    int C_out = w.size(0);
    int KH = w.size(2);
    int KW = w.size(3);

    // int out_h = ...
    int out_h = (H + 2 * pad - KH) / stride + 1;
    // int out_w = ...
    int out_w = (W + 2 * pad - KW) / stride + 1;

    auto out = torch::zeros({N, C_out, out_h, out_w}, x.options());

    dim3 block(TILE_W, TILE_H);
    dim3 grid(
        (out_w + block.x - 1) / block.x,
        (out_h + block.y - 1) / block.y,
        N
    );

    size_t shared_bytes =
        (TILE_C + TILE_C * TILE_H * TILE_W) * sizeof(float);

    gemm_gpu_o4_kernel<<<grid, block, shared_bytes>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C_in, H, W,
        C_out, KH, KW,
        stride, pad,
        out_h, out_w);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_cuda", &conv_cuda, "Custom Conv2D (CUDA)");
}