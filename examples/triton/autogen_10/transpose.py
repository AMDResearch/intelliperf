#!/usr/bin/env python

import triton
import triton.language as tl


@triton.jit
def transpose_kernel(
    input_ptr, output_ptr,
    M, N,
    stride_in_m, stride_in_n,
    stride_out_m, stride_out_n,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    input_ptrs = input_ptr + (offsets_m[:, None] * stride_in_m + offsets_n[None, :] * stride_in_n)
    
    mask = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)
    
    tile = tl.load(input_ptrs, mask=mask, other=0.0)
    
    # Transpose inside the block
    transposed_tile = tl.trans(tile)
    
    # Adjust output pointers for transposed storage
    output_ptrs_transposed = output_ptr + (offsets_n[:, None] * stride_out_m + offsets_m[None, :] * stride_out_n)
    
    mask_transposed = (offsets_n[:, None] < N) & (offsets_m[None, :] < M)
    
    tl.store(output_ptrs_transposed, transposed_tile, mask=mask_transposed) 