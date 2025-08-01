import torch
import triton
import argparse

from smith_waterman import smith_waterman_kernel


def smith_waterman(seq1, seq2, gap_penalty, match_score, mismatch_penalty, validate=False):
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

    if validate:
        s1 = seq1_tensor.view(M, 1)
        s2 = seq2_tensor.view(1, N)
        match_matrix = (s1 == s2)
        scores_torch = torch.where(match_matrix, match_score, mismatch_penalty).to(torch.int32)
        if torch.allclose(scores, scores_torch):
            print("Validation Successful!")
        else:
            print("Validation Failed!")
            print(f"max diff: {(scores - scores_torch).abs().max().item()}")

    return scores


def main(M=4096, N=4096, validate=False):
    seq1 = "A" * M
    seq2 = "C" * N

    gap_penalty = -2
    match_score = 3
    mismatch_penalty = -3

    rep = 100

    for _ in range(10):
        smith_waterman(seq1, seq2, gap_penalty, match_score, mismatch_penalty)

    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    for _ in range(rep):
        smith_waterman(seq1, seq2, gap_penalty, match_score, mismatch_penalty)
    end_time.record()
    torch.cuda.synchronize()

    triton_time = start_time.elapsed_time(end_time) / rep
    print(f"Triton Smith-Waterman (simplified) time: {triton_time:.4f} ms")

    if validate:
        smith_waterman(seq1, seq2, gap_penalty, match_score, mismatch_penalty, validate=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton Smith-Waterman Benchmark")
    parser.add_argument("--M", type=int, default=4096, help="Length of sequence 1")
    parser.add_argument("--N", type=int, default=4096, help="Length of sequence 2")
    parser.add_argument("--validate", action="store_true", help="Validate the Triton implementation against PyTorch.")
    args = parser.parse_args()

    main(args.M, args.N, validate=args.validate) 