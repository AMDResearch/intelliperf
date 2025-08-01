#!/usr/bin/env python

import torch
import triton
import argparse

from softmax import softmax_kernel


def softmax(x, validate=False):
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

    if validate:
        y_torch = torch.nn.functional.softmax(x, dim=1)
        if torch.allclose(y, y_torch, atol=1e-2, rtol=0):
            print("Validation Successful!")
        else:
            print("Validation Failed!")
            print(f"max diff: {(y - y_torch).abs().max().item()}")

    return y


def main(n_rows=32768, n_cols=32768, validate=False):
    x = torch.randn((n_rows, n_cols), device='cuda', dtype=torch.float16)

    rep = 10

    for _ in range(10):
        softmax(x)

    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    for _ in range(rep):
        softmax(x)
    end_time.record()
    torch.cuda.synchronize()

    triton_time = start_time.elapsed_time(end_time) / rep
    print(f"Triton softmax time: {triton_time:.4f} ms")

    if validate:
        softmax(x, validate=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton Softmax Benchmark")
    parser.add_argument("--n_rows", type=int, default=32768, help="Number of rows")
    parser.add_argument("--n_cols", type=int, default=32768, help="Number of columns")
    parser.add_argument("--validate", action="store_true", help="Validate the Triton implementation against PyTorch.")
    args = parser.parse_args()

    main(args.n_rows, args.n_cols, validate=args.validate) 