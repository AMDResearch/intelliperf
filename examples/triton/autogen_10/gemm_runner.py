#!/usr/bin/env python

import torch
import triton
import triton.language as tl
import argparse
import random
import sys
from gemm import streamk_gemm

class matmul(torch.autograd.Function):

    _debug = True

    @staticmethod
    def set_debug(debug: bool):
        matmul._debug = debug

    @staticmethod
    def _call(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, bias: torch.Tensor, P: torch.Tensor,
            locks: torch.Tensor, total_programs_streamk: int, BLK_M: int, BLK_N: int, BLK_K: int, gsize_m: int,
            two_tiles: bool, num_stages: int, num_warps: int, waves_per_eu: int, mfmaInstrSize: int, kpack: int):

        assert a.shape[1] == b.shape[0], "incompatible dimensions"
        M, K = a.shape
        _, N = b.shape

        total_blocks_M = triton.cdiv(M, BLK_M)
        total_blocks_N = triton.cdiv(N, BLK_N)
        iters_per_tile = triton.cdiv(K, BLK_K)
        total_tiles = total_blocks_M * total_blocks_N
        even_k = K % BLK_K == 0

        if total_programs_streamk > 0:
            total_tiles_streamk = total_tiles % total_programs_streamk
            total_blocking_tiles = total_tiles - total_tiles_streamk
            total_iters_streamk = total_tiles_streamk * iters_per_tile
            total_full_tiles_streamk = total_iters_streamk // total_programs_streamk
            total_partial_tiles_streamk = total_iters_streamk % total_programs_streamk
        else:
            total_blocking_tiles = total_tiles
            total_tiles_streamk = 0
            total_full_tiles_streamk = 0
            total_partial_tiles_streamk = 0
            total_iters_streamk = 0
        
        use_bias = False
        grids = total_programs_streamk
        stride_bias = bias.stride(0) if use_bias else 0
        num_xcds = 8
        
        kk = streamk_gemm[(grids, )](
            a,
            b,
            c,
            bias,
            P,
            locks,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            stride_bias,
            BLOCK_SIZE_M=BLK_M,
            BLOCK_SIZE_N=BLK_N,
            BLOCK_SIZE_K=BLK_K,
            GROUP_SIZE_M=gsize_m,
            NUM_SMS=total_programs_streamk,
            STREAMK_TILES=total_tiles_streamk,
            NUM_XCDS=num_xcds,
            BIAS=use_bias,
            EVEN_K=even_k,
        )

        return c

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, bias: torch.Tensor, P: torch.Tensor,
                locks: torch.Tensor, grid: int, BLK_M=128, BLK_N=128, BLK_K=32, gsize_m=1, two_tiles=True, num_stages=3,
                num_warps=4, waves_per_eu=2, mfmaInstrSize=16, kpack=1):
        matmul._call(a=a, b=b, c=c, bias=bias, P=P, locks=locks, total_programs_streamk=grid, BLK_M=BLK_M, BLK_N=BLK_N,
                    BLK_K=BLK_K, gsize_m=gsize_m, two_tiles=two_tiles, num_warps=num_warps, num_stages=num_stages,
                    waves_per_eu=waves_per_eu, mfmaInstrSize=mfmaInstrSize, kpack=kpack)
        return c

def main(M=8192, N=8192, K=8192, validate=False):
    torch.manual_seed(123)
    random.seed(123)

    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    c = torch.zeros((M, N), device="cuda", dtype=a.dtype)
    bias = torch.zeros((M, ), device="cuda", dtype=a.dtype)
    
    total_sm = 304
    
    BLK_M = 256
    BLK_N = 256
    BLK_K = 64
    
    gsize_m = 8
    two_tiles = True
    num_stages = 2
    num_warps = 8
    waves_per_eu = 0
    mfmaInstrSize = 16
    kpack = 2

    locks = torch.zeros((total_sm, ), device="cuda", dtype=torch.int32)
    P = torch.zeros((total_sm, BLK_M * BLK_N), device="cuda", dtype=torch.float32)
    
    # repetitions for performance measurement
    rep = 1
    
    # Warm-up
    for _ in range(1):
        c_triton = matmul.apply(a, b, c, bias, P, locks, total_sm, BLK_M, BLK_N, BLK_K, gsize_m, two_tiles, num_stages, num_warps, waves_per_eu, mfmaInstrSize, kpack)

    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    for _ in range(rep):
        c_triton = matmul.apply(a, b, c, bias, P, locks, total_sm, BLK_M, BLK_N, BLK_K, gsize_m, two_tiles, num_stages, num_warps, waves_per_eu, mfmaInstrSize, kpack)
    end_time.record()
    torch.cuda.synchronize()

    triton_time = start_time.elapsed_time(end_time) / rep

    # torch_output = torch.matmul(a,b)
    # print(f"Triton output: {c_triton}")
    # print(f"Torch output: {torch_output}")
    print(f"Triton matmul time: {triton_time:.4f} ms")

    if validate:
        expected = torch.matmul(a, b)
        if torch.allclose(c_triton, expected, atol=1, rtol=0):
            print("Validation Successful!")
        else:
            print("Validation Failed: Triton output does not match PyTorch output.")
            print(f"max diff: {(c_triton - expected).abs().max().item()}")
            sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton GEMM Benchmark")
    parser.add_argument("--M", type=int, default=8192, help="Number of rows in A and C")
    parser.add_argument("--N", type=int, default=8192, help="Number of columns in B and C")
    parser.add_argument("--K", type=int, default=8192, help="Number of columns in A and rows in B")
    parser.add_argument("--validate", action="store_true", help="Validate the Triton implementation against PyTorch.")
    args = parser.parse_args()
    
    main(args.M, args.N, args.K, validate=args.validate) 