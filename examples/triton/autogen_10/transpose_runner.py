#!/usr/bin/env python3

import torch
import triton
import argparse

from transpose import transpose_kernel


def transpose(x, validate=False):
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

    if validate:
        y_torch = x.T
        if torch.allclose(y, y_torch):
            print("Validation Successful!")
        else:
            print("Validation Failed!")
            print(f"max diff: {(y - y_torch).abs().max().item()}")

    return y


def main(M=10000, N=10000, validate=False):
    x = torch.randn((M, N), device='cuda', dtype=torch.float16)

    rep = 10

    for _ in range(10):
        transpose(x)

    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    for _ in range(rep):
        transpose(x)
    end_time.record()
    torch.cuda.synchronize()

    triton_time = start_time.elapsed_time(end_time) / rep
    print(f"Triton transpose time: {triton_time:.4f} ms")

    if validate:
        transpose(x, validate=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton Transpose Benchmark")
    parser.add_argument("--M", type=int, default=4096*8, help="Number of rows")
    parser.add_argument("--N", type=int, default=4096*8, help="Number of columns")
    parser.add_argument("--validate", action="store_true", help="Validate the Triton implementation against PyTorch.")
    args = parser.parse_args()

    main(args.M, args.N, validate=args.validate) 