#!/usr/bin/env python

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import sys
import torch
import argparse

from lean_attn import persistent_lean_attention


def main(args):
    """
    Main function to run the Lean Attention benchmark and validation.
    """
    torch.manual_seed(20)

    # Unpack arguments
    causal = args.causal
    batch = args.batch
    h = args.h
    n_ctx_q = args.n_ctx_q
    n_ctx = args.n_ctx
    d = args.d
    total_programs = args.total_programs
    init_dtype = torch.float16 if args.fp16 else torch.float32
    BLOCK_M = args.BLOCK_M
    BLOCK_N = args.BLOCK_N
    waves_per_eu = args.waves_per_eu
    num_warps = args.num_warps

    print("--- Lean Attention Benchmark Configuration ---")
    print(f"  Causal: {causal}")
    print(f"  Batch Size: {batch}")
    print(f"  Heads: {h}")
    print(f"  Query SeqLen: {n_ctx_q}")
    print(f"  Key/Value SeqLens: {n_ctx}")
    print(f"  Head Dimension: {d}")
    print(f"  Total Programs: {total_programs}")
    print(f"  Data Type: {'float16' if args.fp16 else 'float32'}")
    print(f"  BLOCK_M: {BLOCK_M}, BLOCK_N: {BLOCK_N}")
    print(f"  Waves per EU: {waves_per_eu}, Num Warps: {num_warps}")
    print("---------------------------------------------")

    if any(item > 524288 for item in n_ctx):
        BLOCK_N = 256
        d = 16
        print("\nAdjusting BLOCK_N to 256 and d to 16 for long sequences.")

    assert batch == len(
        n_ctx
    ), f"Batch size ({batch}) must match the number of context lengths provided ({len(n_ctx)})."

    try:
        sum_n_ctx = sum(int(n) for n in n_ctx)
    except ValueError:
        print(f"N_CTX contains non-numeric values: {n_ctx}")
        sys.exit(1)

    list_num_block_n = [
        (int(str(s).strip()) + BLOCK_N - 1) // BLOCK_N for s in n_ctx
    ]
    len_sum = 0
    list_sum_block_n = []
    for i in range(batch):
        len_sum += list_num_block_n[i]
        list_sum_block_n.append(len_sum)
    batch_num_block_n = torch.tensor(
        list_sum_block_n, device="cuda", dtype=torch.int32
    )

    sm_scale = 0.5

    q = torch.empty((n_ctx_q * batch, h, d), dtype=init_dtype, device="cuda").normal_(
        mean=0.0, std=0.5
    )
    k = torch.empty((sum_n_ctx, h, d), dtype=init_dtype, device="cuda").normal_(
        mean=0.0, std=0.5
    )
    v = torch.empty((sum_n_ctx, h, d), dtype=init_dtype, device="cuda").normal_(
        mean=0.0, std=0.5
    )

    Mp = torch.empty((total_programs, n_ctx_q), device=q.device, dtype=torch.float32)
    Lp = torch.empty((total_programs, n_ctx_q), device=q.device, dtype=torch.float32)
    Op = torch.empty(
        (total_programs, n_ctx_q, d), device=q.device, dtype=torch.float32
    )
    locks = torch.zeros((total_programs,), device=q.device, dtype=torch.int32)

    for _ in range(5):
        _ = persistent_lean_attention(
            q,
            k,
            v,
            Mp,
            Lp,
            Op,
            locks,
            batch_num_block_n,
            total_programs,
            BLOCK_M,
            BLOCK_N,
            causal,
            batch,
            sm_scale,
            num_warps,
            waves_per_eu,
        )
    torch.cuda.synchronize()

    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    for _ in range(args.repeat):
        la_out = persistent_lean_attention(
            q,
            k,
            v,
            Mp,
            Lp,
            Op,
            locks,
            batch_num_block_n,
            total_programs,
            BLOCK_M,
            BLOCK_N,
            causal,
            batch,
            sm_scale,
            num_warps,
            waves_per_eu,
        )
    end_time.record()

    torch.cuda.synchronize()
    triton_time = start_time.elapsed_time(end_time) / args.repeat
    print(f"\nLean Attention Implementation Time: {triton_time:.4f} ms")

    if args.validate:
        print("\nRunning validation...")
        ref_out = torch.empty_like(q, dtype=v.dtype)
        start = 0
        start_q = 0

        for b in n_ctx:
            qb = q[start_q : (start_q + int(n_ctx_q)), :, :]
            qb_reshaped = qb.transpose(0, 1)
            kb = k[start : (start + int(b)), :, :]
            kb_reshaped = kb.transpose(0, 1)
            vb = v[start : (start + int(b)), :, :]
            vb_reshaped = vb.transpose(0, 1)
            p = torch.matmul(qb_reshaped, kb_reshaped.transpose(-2, -1)) * sm_scale
            if causal:
                M = torch.tril(torch.ones((n_ctx_q, b), device="cuda"))
                mask = M == 0
                p[:, mask] = float("-inf")
            p = torch.softmax(p.float(), dim=-1).to(q.dtype)
            refb = torch.matmul(p, vb_reshaped)
            ref_out[start_q : (start_q + int(n_ctx_q)), :, :] = refb.transpose(0, 1)
            start += b
            start_q += n_ctx_q

        atol = 1e-2
        rtol = 3e-3
        try:
            torch.testing.assert_close(ref_out, la_out, atol=atol, rtol=rtol)
            print("✅ Validation Successful!")
        except AssertionError as e:
            print("❌ Validation Failed: Triton output does not match reference output.")
            print(e)
            sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lean Attention Runner and Benchmark")
    parser.add_argument(
        "--causal", action="store_true", default=False, help="Enable causal masking"
    )
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--h", type=int, default=64, help="Number of heads")
    parser.add_argument(
        "--n_ctx_q", type=int, default=16, help="Sequence length for Query"
    )
    parser.add_argument(
        "--n_ctx",
        type=int,
        nargs="+",
        default=[65536],
        help="List of context lengths for Key/Value",
    )
    parser.add_argument("--d", type=int, default=64, help="Head dimension")
    parser.add_argument(
        "--total_programs",
        type=int,
        default=912,
        help="Total programs for Lean Attention",
    )
    parser.add_argument(
        "--fp16", action="store_true", default=True, help="Use float16 data type"
    )
    parser.add_argument("--BLOCK_M", type=int, default=16)
    parser.add_argument("--BLOCK_N", type=int, default=64)
    parser.add_argument("--waves_per_eu", type=int, default=2)
    parser.add_argument("--num_warps", type=int, default=4)
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the Triton implementation against a reference.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=10,
        help="Number of repetitions for performance measurement",
    )

    args = parser.parse_args()
    main(args)