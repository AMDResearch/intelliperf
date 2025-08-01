#!/usr/bin/env python

import torch
import triton
import argparse

from stencil_2d import stencil_2d_kernel


def stencil_2d(x, validate=False):
    M, N = x.shape
    y = torch.empty_like(x)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']),
        triton.cdiv(N, META['BLOCK_SIZE_N'])
    )

    stencil_2d_kernel[grid](
        x, y,
        M, N,
        x.stride(0), x.stride(1),
        BLOCK_SIZE_M=32, BLOCK_SIZE_N=32
    )

    if validate:
        def torch_stencil(x):
            weight = torch.tensor([[0, 0.125, 0],
                                   [0.125, 0.5, 0.125],
                                   [0, 0.125, 0]], device=x.device, dtype=x.dtype).reshape(1, 1, 3, 3)
            return torch.nn.functional.conv2d(x.reshape(1, 1, M, N), weight, padding='same').reshape(M, N)

        y_torch = torch_stencil(x)
        if torch.allclose(y, y_torch, atol=1e-2, rtol=0):
            print("Validation Successful!")
        else:
            print("Validation Failed!")
            print(f"max diff: {(y - y_torch).abs().max().item()}")

    return y


def main(M=8192, N=8192, validate=False):
    x = torch.randn((M, N), device='cuda', dtype=torch.float16)

    rep = 10

    for _ in range(10):
        stencil_2d(x)

    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    for _ in range(rep):
        stencil_2d(x)
    end_time.record()
    torch.cuda.synchronize()

    triton_time = start_time.elapsed_time(end_time) / rep
    print(f"Triton 2D stencil time: {triton_time:.4f} ms")

    if validate:
        stencil_2d(x, validate=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton 2D Stencil Benchmark")
    parser.add_argument("--M", type=int, default=8192, help="Number of rows")
    parser.add_argument("--N", type=int, default=8192, help="Number of columns")
    parser.add_argument("--validate", action="store_true", help="Validate the Triton implementation against PyTorch.")
    args = parser.parse_args()

    main(args.M, args.N, validate=args.validate) 