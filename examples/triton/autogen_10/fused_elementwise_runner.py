#!/usr/bin/env python

import torch
import triton
import argparse

from fused_elementwise import fused_elementwise_kernel


def fused_elementwise(x, y, z, validate=False):
    n_elements = x.numel()
    output = torch.empty_like(x)

    grid = lambda META: (triton.cdiv(n_elements, META['BLOCK_SIZE']),)

    fused_elementwise_kernel[grid](
        x, y, z, output,
        n_elements,
        BLOCK_SIZE=1024
    )

    if validate:
        output_torch = torch.nn.functional.relu(x * y + z)
        if torch.allclose(output, output_torch, atol=1e-2, rtol=0):
            print("Validation Successful!")
        else:
            print("Validation Failed!")
            print(f"max diff: {(output - output_torch).abs().max().item()}")

    return output


def main(size=2**24, validate=False):
    x = torch.randn(size, device='cuda', dtype=torch.float16)
    y = torch.randn(size, device='cuda', dtype=torch.float16)
    z = torch.randn(size, device='cuda', dtype=torch.float16)

    rep = 10

    for _ in range(10):
        fused_elementwise(x, y, z)

    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    for _ in range(rep):
        fused_elementwise(x, y, z)
    end_time.record()
    torch.cuda.synchronize()

    triton_time = start_time.elapsed_time(end_time) / rep
    print(f"Triton fused element-wise time: {triton_time:.4f} ms")

    if validate:
        fused_elementwise(x, y, z, validate=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton Fused Element-wise Benchmark")
    parser.add_argument("--size", type=int, default=2**24, help="Size of the input tensors")
    parser.add_argument("--validate", action="store_true", help="Validate the Triton implementation against PyTorch.")
    args = parser.parse_args()

    main(args.size, validate=args.validate) 