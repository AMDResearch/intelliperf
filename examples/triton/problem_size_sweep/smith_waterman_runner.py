#!/usr/bin/env python

import torch
import triton
import triton.language as tl
import argparse

# Define the swizzled and unswizzled kernels
from smith_waterman_swizzle import smith_waterman_kernel as smith_waterman_swizzle
from smith_waterman_unswizzle import smith_waterman_kernel as smith_waterman_unswizzle

def run_smith_waterman(seq1_len, seq2_len, version):
    seq1 = torch.randint(0, 4, (seq1_len,), device='cuda', dtype=torch.int32).to(torch.int8)
    seq2 = torch.randint(0, 4, (seq2_len,), device='cuda', dtype=torch.int32).to(torch.int8)
    scores = torch.zeros((seq1_len, seq2_len), device='cuda', dtype=torch.int32)
    gap_penalty = -1
    match_score = 1
    mismatch_penalty = -1
    
    if version == "swizzle":
        kernel_to_run = smith_waterman_swizzle
    elif version == "unswizzle":
        kernel_to_run = smith_waterman_unswizzle
    else:
        raise ValueError("Invalid version specified. Choose 'swizzle' or 'unswizzle'.")

    grid = (triton.cdiv(seq1_len, 32), triton.cdiv(seq2_len, 32))

    kernel_to_run[grid](
        seq1, seq2, scores,
        seq1_len, seq2_len,
        scores.stride(0), scores.stride(1),
        gap_penalty, match_score, mismatch_penalty,
        BLOCK_SIZE_M=32, BLOCK_SIZE_N=32
    )
    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Smith-Waterman kernel with specified sequence lengths.")
    parser.add_argument('--seq1_len', type=int, required=True, help='Length of sequence 1')
    parser.add_argument('--seq2_len', type=int, required=True, help='Length of sequence 2')
    parser.add_argument('--version', type=str, required=True, choices=['swizzle', 'unswizzle'], help='Kernel version to run')
    
    args = parser.parse_args()
    
    run_smith_waterman(args.seq1_len, args.seq2_len, args.version) 