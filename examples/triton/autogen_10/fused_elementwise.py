#!/usr/bin/env python

import triton
import triton.language as tl


@triton.jit
def fused_elementwise_kernel(
    x_ptr, y_ptr, z_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    z = tl.load(z_ptr + offsets, mask=mask)

    # Fused operations: (x * y) + z -> ReLU
    result = x * y + z
    output = tl.where(result > 0, result, 0) # ReLU activation

    tl.store(output_ptr + offsets, output, mask=mask) 