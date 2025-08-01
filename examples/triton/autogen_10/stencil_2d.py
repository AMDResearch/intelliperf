#!/usr/bin/env python

import triton
import triton.language as tl


@triton.jit
def stencil_2d_kernel(
    input_ptr, output_ptr,
    M, N,
    stride_m, stride_n,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    input_ptrs = input_ptr + offsets_m[:, None] * stride_m + offsets_n[None, :] * stride_n
    
    mask = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)
    
    center = tl.load(input_ptrs, mask=mask, other=0.0)
    
    # Boundary checks for neighbors
    up_mask = ((offsets_m - 1)[:, None] >= 0) & mask
    down_mask = ((offsets_m + 1)[:, None] < M) & mask
    left_mask = ((offsets_n - 1)[None, :] >= 0) & mask
    right_mask = ((offsets_n + 1)[None, :] < N) & mask
    
    up = tl.load(input_ptrs - stride_m, mask=up_mask, other=0.0)
    down = tl.load(input_ptrs + stride_m, mask=down_mask, other=0.0)
    left = tl.load(input_ptrs - stride_n, mask=left_mask, other=0.0)
    right = tl.load(input_ptrs + stride_n, mask=right_mask, other=0.0)

    # Simple 5-point stencil operation
    output = 0.5 * center + 0.125 * (up + down + left + right)
    
    output_ptrs = output_ptr + offsets_m[:, None] * stride_m + offsets_n[None, :] * stride_n
    tl.store(output_ptrs, output, mask=mask) 