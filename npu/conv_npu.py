import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


"""
A convolution kernel that you need to implement.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height
out_pool_width = out_width

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""

@nki.jit
def conv2d(X, W, bias):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height
    out_pool_width = out_width
    
    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Various tiling dimensions (You may want to define more of them)
    c_in_pmax = nl.tile_size.pmax
    n_tiles_c_in = in_channels // c_in_pmax

    c_out_fmax = nl.tile_size.gemm_stationary_fmax
    n_tiles_c_out = out_channels // c_out_fmax

    
    rows_per_tile = min(
        out_height, nl.tile_size.gemm_moving_fmax // out_width
    )
    tiles_per_stripe = min(
        4, (out_height + rows_per_tile - 1) // rows_per_tile
    )
    rows_per_stripe = tiles_per_stripe * rows_per_tile
    n_stripes = (out_height + rows_per_stripe - 1) // rows_per_stripe

    input_rows = rows_per_stripe + filter_height - 1
    pixels_per_tile = rows_per_tile * out_width
    
    weights = nl.ndarray(
        (c_in_pmax, n_tiles_c_out, n_tiles_c_in,
         filter_height, filter_width, c_out_fmax),
        dtype=W.dtype, buffer=nl.sbuf,
    )
    biases = nl.ndarray(
        (c_out_fmax, n_tiles_c_out, 1),
        dtype=nl.float32, buffer=nl.sbuf,
    )
    for co in nl.affine_range(n_tiles_c_out):
        biases[:, co, :] = nl.load(
            bias[co*c_out_fmax:(co+1)*c_out_fmax],
            dtype=nl.float32,
        ).reshape((c_out_fmax, 1))

        for ci in nl.affine_range(n_tiles_c_in):
            w = nl.load(
                W[co*c_out_fmax:(co+1)*c_out_fmax,
                  ci*c_in_pmax:(ci+1)*c_in_pmax, :, :]
            )
            for kh in nl.affine_range(filter_height):
                for kw in nl.affine_range(filter_width):
                    weights[:, co, ci, kh, kw, :] = nl.copy(
                        nisa.nc_transpose(w[:, :, kh, kw]),
                        dtype=W.dtype,
                    )
    #mask   
    i_c = nl.arange(c_in_pmax)[:, None, None]
    i_h = nl.arange(input_rows)[None, :, None]
    i_w = nl.arange(input_width)[None, None, :]

    o_c = nl.arange(c_out_fmax)[:, None, None]
    o_h = nl.arange(rows_per_tile)[None, :, None]
    o_w = nl.arange(out_width)[None, None, :]
    
    for b in nl.affine_range(batch_size):
        for stripe in nl.affine_range(n_stripes):
            row_start = stripe * rows_per_stripe

            image = nl.zeros(
                (c_in_pmax, n_tiles_c_in, input_rows, input_width),
                dtype=X.dtype, buffer=nl.sbuf,
            )
            for ci in nl.affine_range(n_tiles_c_in):
                nisa.dma_copy(
                    dst=image[:, ci, :, :],
                    src=X[b, ci*c_in_pmax + i_c,
                          row_start + i_h, i_w],
                    mask=row_start + i_h < input_height,
                )

            for co in nl.affine_range(n_tiles_c_out):
                for tile in nl.affine_range(tiles_per_stripe):
                    local_row = tile * rows_per_tile
                    acc = nl.zeros(
                        (c_out_fmax, pixels_per_tile),
                        dtype=nl.float32, buffer=nl.psum,
                    )

                    for ci in nl.affine_range(n_tiles_c_in):
                        for kh in nl.affine_range(filter_height):
                            for kw in nl.affine_range(filter_width):
                                patch = nl.copy(
                                    image[:, ci,
                                          local_row+kh:
                                          local_row+kh+rows_per_tile,
                                          kw:kw+out_width]
                                ).reshape((c_in_pmax, pixels_per_tile))

                                acc += nisa.nc_matmul(
                                    weights[:, co, ci, kh, kw, :],
                                    patch,
                                )

                    result = nl.add(nl.copy(acc), biases[:, co, :])
                    result = nl.copy(result, dtype=X.dtype).reshape(
                        (c_out_fmax, rows_per_tile, out_width)
                    )

                    output_row = row_start + local_row
                    nl.store(
                        X_out[b, co*c_out_fmax + o_c,
                              output_row + o_h, o_w],
                        value=result,
                        mask=output_row + o_h < out_height,
                    )

    return X_out

