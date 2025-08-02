#!/usr/bin/env python

import torch
import triton
import argparse

from layer_norm import layer_norm_kernel


def layer_norm(x, w, b, eps, validate=False):
    M, N = x.shape
    y = torch.empty_like(x)

    grid = (M,)

    layer_norm_kernel[grid](
        x, y, w, b,
        M, N,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        eps,
        BLOCK_SIZE_N=triton.next_power_of_2(N)
    )

    if validate:
        y_torch = torch.nn.functional.layer_norm(x, (N,), w, b, eps)
        if torch.allclose(y, y_torch, atol=1e-1, rtol=0):
            print("Validation Successful!")
        else:
            print("Validation Failed!")
            print(f"max diff: {(y - y_torch).abs().max().item()}")

    return y


def main(M=20480, N=20480, validate=False):
    x = torch.randn((M, N), device='cuda', dtype=torch.float16)
    w = torch.randn((N,), device='cuda', dtype=torch.float16)
    b = torch.randn((N,), device='cuda', dtype=torch.float16)
    eps = 1e-5

    rep = 10

    for _ in range(10):
        layer_norm(x, w, b, eps)

    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    for _ in range(rep):
        layer_norm(x, w, b, eps)
    end_time.record()
    torch.cuda.synchronize()

    triton_time = start_time.elapsed_time(end_time) / rep
    print(f"Triton layer norm time: {triton_time:.4f} ms")

    if validate:
        layer_norm(x, w, b, eps, validate=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton LayerNorm Benchmark")
    parser.add_argument("--M", type=int, default=20480, help="Number of rows")
    parser.add_argument("--N", type=int, default=20480, help="Number of columns")
    parser.add_argument("--validate", action="store_true", help="Validate the Triton implementation against PyTorch.")
    args = parser.parse_args()

    main(args.M, args.N, validate=args.validate) 