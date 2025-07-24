#!/usr/bin/env python

import torch
import triton
import triton.language as tl
import argparse

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
    output_ptrs = output_ptr + (offsets_m[:, None] * stride_out_m + offsets_n[None, :] * stride_out_n)

    mask = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)
    
    tile = tl.load(input_ptrs, mask=mask)
    
    # Transpose inside the block
    transposed_tile = tl.trans(tile)
    
    # Adjust output pointers for transposed storage
    output_ptrs_transposed = output_ptr + (offsets_n[:, None] * stride_out_n + offsets_m[None, :] * stride_out_m)
    
    mask_transposed = (offsets_n[:, None] < N) & (offsets_m[None, :] < M)
    
    tl.store(output_ptrs_transposed, transposed_tile, mask=mask_transposed)


def transpose(x):
    M, N = x.shape
    y = torch.empty((N, M), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']),
        triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    transpose_kernel[grid](
        x, y,
        M, N,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_SIZE_M=32, BLOCK_SIZE_N=32
    )
    return y

def main(M=8192, N=8192):
    x = torch.randn((M, N), device='cuda', dtype=torch.float16)

    rep = 100
    
    for _ in range(10):
        y_triton = transpose(x)

    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    for _ in range(rep):
        y_triton = transpose(x)
    end_time.record()
    torch.cuda.synchronize()

    triton_time = start_time.elapsed_time(end_time) / rep
    print(f"Triton transpose time: {triton_time:.4f} ms")

    # y_torch = x.T
    # print(f"Triton output: {y_triton}")
    # print(f"Torch output: {y_torch}")
    # assert torch.allclose(y_triton, y_torch), "Triton and PyTorch results differ"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton Transpose Benchmark")
    parser.add_argument("--M", type=int, default=8192, help="Number of rows")
    parser.add_argument("--N", type=int, default=8192, help="Number of columns")
    args = parser.parse_args()
    
    main(args.M, args.N) 