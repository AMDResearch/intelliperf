#!/usr/bin/env python

import triton
import triton.language as tl


@triton.jit
def layer_norm_kernel(
    x_ptr, y_ptr,
    w_ptr, b_ptr,
    M, N,
    stride_x_m, stride_x_n,
    stride_y_m, stride_y_n,
    eps,
    BLOCK_SIZE_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    
    offset_m = pid_m * stride_x_m
    
    x_row_ptr = x_ptr + offset_m
    y_row_ptr = y_ptr + offset_m
    
    mean = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    var = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    
    for off in range(0, N, BLOCK_SIZE_N):
        cols = off + tl.arange(0, BLOCK_SIZE_N)
        mask = cols < N
        
        x = tl.load(x_row_ptr + cols * stride_x_n, mask=mask, other=0.0).to(tl.float32)
        
        mean += tl.sum(x, axis=0)
        var += tl.sum(x * x, axis=0)

    mean = mean / N
    var = var / N - mean * mean
    
    rstd = 1 / tl.sqrt(var + eps)
    
    for off in range(0, N, BLOCK_SIZE_N):
        cols = off + tl.arange(0, BLOCK_SIZE_N)
        mask = cols < N
        
        x = tl.load(x_row_ptr + cols * stride_x_n, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask)
        b = tl.load(b_ptr + cols, mask=mask)
        
        y = (x - mean) * rstd * w + b
        
        tl.store(y_row_ptr + cols * stride_y_n, y.to(y_ptr.dtype.element_ty), mask=mask) 