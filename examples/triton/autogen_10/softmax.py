#!/usr/bin/env python

import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr, input_ptr,
    input_row_stride, output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    
    row_start_ptr = input_ptr + row_idx * input_row_stride
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    
    row_minus_max = row - tl.max(row, axis=0)
    
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)

def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
        
    y = torch.empty_like(x)
    
    softmax_kernel[(n_rows,)](
        y, x,
        x.stride(0), y.stride(0),
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return y

def main():
    n_rows, n_cols = 8192, 8192
    x = torch.randn((n_rows, n_cols), device='cuda', dtype=torch.float16)
    
    rep = 100
    
    for _ in range(10):
        y_triton = softmax(x)

    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    for _ in range(rep):
        y_triton = softmax(x)
    end_time.record()
    torch.cuda.synchronize()

    triton_time = start_time.elapsed_time(end_time) / rep
    print(f"Triton softmax time: {triton_time:.4f} ms")

    # y_torch = torch.nn.functional.softmax(x, dim=1)
    # print(f"Triton output: {y_triton}")
    # print(f"Torch output: {y_torch}")
    # assert torch.allclose(y_triton, y_torch, atol=1e-2, rtol=0), "Triton and PyTorch results differ"


if __name__ == "__main__":
    main() 