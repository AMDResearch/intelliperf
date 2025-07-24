#!/usr/bin/env python

import torch
import triton
import triton.language as tl
import argparse

@triton.jit
def smith_waterman_kernel(
    seq1_ptr, seq2_ptr,
    score_ptr,
    M, N,
    stride_m, stride_n,
    gap_penalty, match_score, mismatch_penalty,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Load sequence fragments
    mask_m = offsets_m[:, None] < M
    mask_n = offsets_n[None, :] < N
    seq1_chars = tl.load(seq1_ptr + offsets_m[:, None], mask=mask_m, other=0)
    seq2_chars = tl.load(seq2_ptr + offsets_n[None, :], mask=mask_n, other=0)
    
    # Perform element-wise comparison
    match = tl.where(seq1_chars == seq2_chars, match_score, mismatch_penalty)
    
    # A simplified scoring model (no dependencies)
    score = match
    
    # Store the scores
    score_ptrs = score_ptr + (offsets_m[:, None] * stride_m + offsets_n[None, :] * stride_n)
    tl.store(score_ptrs, score, mask=mask_m & mask_n)


def smith_waterman(seq1, seq2, gap_penalty, match_score, mismatch_penalty):
    M, N = len(seq1), len(seq2)
    scores = torch.zeros(M, N, device='cuda', dtype=torch.int32)
    
    seq1_tensor = torch.tensor(list(seq1.encode('ascii')), device='cuda', dtype=torch.int8)
    seq2_tensor = torch.tensor(list(seq2.encode('ascii')), device='cuda', dtype=torch.int8)

    grid = (triton.cdiv(M, 32), triton.cdiv(N, 32))
    
    smith_waterman_kernel[grid](
        seq1_tensor, seq2_tensor,
        scores,
        M, N,
        scores.stride(0), scores.stride(1),
        gap_penalty, match_score, mismatch_penalty,
        BLOCK_SIZE_M=32, BLOCK_SIZE_N=32
    )
    return scores

def main(M=4096, N=4096):
    seq1 = "A" * M # Simplified sequences for demonstration
    seq2 = "C" * N
    
    gap_penalty = -2
    match_score = 3
    mismatch_penalty = -3

    rep = 100
    
    # Warm-up
    for _ in range(10):
        scores = smith_waterman(seq1, seq2, gap_penalty, match_score, mismatch_penalty)

    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    for _ in range(rep):
        scores = smith_waterman(seq1, seq2, gap_penalty, match_score, mismatch_penalty)
    end_time.record()
    torch.cuda.synchronize()

    triton_time = start_time.elapsed_time(end_time) / rep
    print(f"Triton Smith-Waterman (simplified) time: {triton_time:.4f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton Smith-Waterman Benchmark")
    parser.add_argument("--M", type=int, default=4096, help="Length of sequence 1")
    parser.add_argument("--N", type=int, default=4096, help="Length of sequence 2")
    args = parser.parse_args()
    
    main(args.M, args.N) 