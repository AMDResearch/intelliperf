#!/usr/bin/env python

import triton
import triton.language as tl


@triton.jit
def spmv_kernel(
    y_ptr, x_ptr,
    data_ptr, indices_ptr, indptr_ptr,
    M,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    row_start = tl.load(indptr_ptr + pid)
    row_end = tl.load(indptr_ptr + pid + 1)
    
    acc = 0.0
    for i in range(row_start, row_end, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < row_end
        
        col_indices = tl.load(indices_ptr + offsets, mask=mask)
        data = tl.load(data_ptr + offsets, mask=mask)
        x_vals = tl.load(x_ptr + col_indices, mask=mask)
        
        acc += tl.sum(data * x_vals)
        
    tl.store(y_ptr + pid, acc) 