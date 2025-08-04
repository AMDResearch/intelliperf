#!/usr/bin/env python

import torch
import triton
import triton.language as tl
import argparse

# Define the swizzled and unswizzled kernels
from softmax_swizzle import softmax_kernel as softmax_swizzle
from softmax_unswizzle import softmax_kernel as softmax_unswizzle

def run_softmax(batch_size, seq_len, vocab_size, version):
    x = torch.randn((batch_size * seq_len, vocab_size), device='cuda', dtype=torch.float16)
    y = torch.empty_like(x)

    if version == "swizzle":
        kernel_to_run = softmax_swizzle
    elif version == "unswizzle":
        kernel_to_run = softmax_unswizzle
    else:
        raise ValueError("Invalid version specified. Choose 'swizzle' or 'unswizzle'.")

    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
        
    grid = (n_rows,)

    kernel_to_run[grid](
        y, x,
        x.stride(0), y.stride(0),
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return y

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Softmax kernel with specified dimensions.")
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size')
    parser.add_argument('--seq_len', type=int, required=True, help='Sequence length')
    parser.add_argument('--vocab_size', type=int, required=True, help='Vocabulary size')
    parser.add_argument('--version', type=str, required=True, choices=['swizzle', 'unswizzle'], help='Kernel version to run')
    
    args = parser.parse_args()
    
    run_softmax(args.batch_size, args.seq_len, args.vocab_size, args.version) 