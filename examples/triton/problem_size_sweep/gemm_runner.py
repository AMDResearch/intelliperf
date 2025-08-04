#!/usr/bin/env python

import torch
import triton
import argparse

# Define the swizzled and unswizzled kernels
from gemm_swizzle import streamk_gemm as gemm_swizzle
from gemm_unswizzle import streamk_gemm as gemm_unswizzle


def run_gemm(m, n, k, version):
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    bias = torch.zeros((m,), device="cuda", dtype=a.dtype)

    if version == "swizzle":
        kernel_to_run = gemm_swizzle
    elif version == "unswizzle":
        kernel_to_run = gemm_unswizzle
    else:
        raise ValueError("Invalid version specified. Choose 'swizzle' or 'unswizzle'.")

    # Kernel meta-parameters
    BLOCK_SIZE_M = 256
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 64
    GROUP_SIZE_M = 8
    total_sm = 304
    locks = torch.zeros((total_sm,), device="cuda", dtype=torch.int32)
    P = torch.zeros((total_sm, BLOCK_SIZE_M * BLOCK_SIZE_N), device="cuda", dtype=torch.float32)
    
    NUM_SMS = total_sm
    STREAMK_TILES = 0
    NUM_XCDS = 8
    BIAS = False
    EVEN_K = (k % BLOCK_SIZE_K) == 0

    grid = (NUM_SMS,)

    kernel_to_run[grid](
        a,
        b,
        c,
        bias,
        P,
        locks,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        0,  # stride_bias, since BIAS is False
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        NUM_SMS=NUM_SMS,
        STREAMK_TILES=STREAMK_TILES,
        NUM_XCDS=NUM_XCDS,
        BIAS=BIAS,
        EVEN_K=EVEN_K,
    )
    return c


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GEMM kernel with specified dimensions.")
    parser.add_argument("--m", type=int, required=True, help="Dimension M")
    parser.add_argument("--n", type=int, required=True, help="Dimension N")
    parser.add_argument("--k", type=int, required=True, help="Dimension K")
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        choices=["swizzle", "unswizzle"],
        help="Kernel version to run",
    )

    args = parser.parse_args()

    run_gemm(args.m, args.n, args.k, args.version) 