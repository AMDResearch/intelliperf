#!/usr/bin/env python

import torch
import triton
import triton.language as tl
import argparse

# Define the swizzled and unswizzled kernels
from layer_norm_swizzle import layer_norm_kernel as layer_norm_swizzle
from layer_norm_unswizzle import layer_norm_kernel as layer_norm_unswizzle

def run_layer_norm(batch_size, seq_len, hidden_size, version):
    x = torch.randn((batch_size * seq_len, hidden_size), device='cuda', dtype=torch.float16)
    y = torch.empty_like(x)
    weight = torch.randn((hidden_size,), device='cuda', dtype=torch.float16)
    bias = torch.randn((hidden_size,), device='cuda', dtype=torch.float16)
    eps = 1e-5
    
    if version == "swizzle":
        kernel_to_run = layer_norm_swizzle
    elif version == "unswizzle":
        kernel_to_run = layer_norm_unswizzle
    else:
        raise ValueError("Invalid version specified. Choose 'swizzle' or 'unswizzle'.")

    M, N = x.shape
    grid = (M,)
    
    kernel_to_run[grid](
        x, y, weight, bias,
        M, N,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        eps,
        BLOCK_SIZE_N=triton.next_power_of_2(N)
    )
    return y

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LayerNorm kernel with specified dimensions.")
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size')
    parser.add_argument('--seq_len', type=int, required=True, help='Sequence length')
    parser.add_argument('--hidden_size', type=int, required=True, help='Hidden size')
    parser.add_argument('--version', type=str, required=True, choices=['swizzle', 'unswizzle'], help='Kernel version to run')
    
    args = parser.parse_args()
    
    run_layer_norm(args.batch_size, args.seq_len, args.hidden_size, args.version) 