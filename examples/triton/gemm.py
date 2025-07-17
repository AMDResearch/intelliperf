#!/usr/bin/env python

import triton
import triton.language as tl

@triton.jit()
def streamk_gemm(
    A,
    B,
    C,
    bias_ptr,
    P,
    locks,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bias,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    STREAMK_TILES: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    # original program‐ID
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M) 
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    iters_per_tile = tl.cdiv(K, BLOCK_SIZE_K)
    total_tiles = num_pid_m * num_pid_n
    total_full_tiles = total_tiles - STREAMK_TILES

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32

    for tile_id in range(pid, total_full_tiles, NUM_SMS):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
        tl.assume(pid_m > 0)
        tl.assume(pid_n > 0)

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rk = tl.arange(0, BLOCK_SIZE_K)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, loop_k):
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            A_BASE = tl.multiple_of(A_BASE, (1, 16))
            B_BASE = tl.multiple_of(B_BASE, (16, 1))
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)
            acc += tl.dot(a, b)

        c = acc.to(C.type.element_ty)
        if BIAS:
            c += bias[:, None]

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        tl.store(C_, c, mask=mask)

    tl.assume(pid >= 0)
    total_streamk_iters = STREAMK_TILES * iters_per_tile
    streamk_iters_pcu = total_streamk_iters // NUM_SMS
    streamk_remainder_iters = total_streamk_iters % NUM_SMS
    start_iter = total_full_tiles * iters_per_tile + pid * streamk_iters_pcu + tl.minimum(pid, streamk_remainder_iters)
    last_iter = total_full_tiles * iters_per_tile + (pid + 1) * streamk_iters_pcu + tl.minimum(
        pid + 1, streamk_remainder_iters)
    while start_iter < last_iter:
        remainder = start_iter % iters_per_tile
        end_iter = tl.minimum(start_iter + (iters_per_tile - remainder), last_iter)
        tile_id = start_iter // iters_per_tile
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
        tl.assume(pid_m > 0)
        tl.assume(pid_n > 0)

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rk = tl.arange(0, BLOCK_SIZE_K)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak + BLOCK_SIZE_K * stride_ak * remainder
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn + BLOCK_SIZE_K * stride_bk * remainder
        A_BASE = tl.multiple_of(A_BASE, (1, 16))
        B_BASE = tl.multiple_of(B_BASE, (16, 1))

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for current_iter in range(start_iter, end_iter):
            if EVEN_K:
                a = tl.load(A_BASE)
                b = tl.load(B_BASE)
            else:
                global_k_offset = (current_iter % iters_per_tile) * BLOCK_SIZE_K
                k_mask = global_k_offset + rk < K
                a = tl.load(A_BASE, mask=k_mask[None, :], other=0.0)
                b = tl.load(B_BASE, mask=k_mask[:, None], other=0.0)
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        tile_iter = tile_id * iters_per_tile

        if start_iter != tile_iter:
            rm1 = tl.arange(0, BLOCK_SIZE_M)
            rn1 = tl.arange(0, BLOCK_SIZE_N)
            rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
            P_ = P + pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
            tl.store(P_, acc, cache_modifier=".wt")
            tl.debug_barrier()
            tl.store(locks + pid, 1, cache_modifier=".wt")
        else:
            next_pid = pid + 1
            tile_iter_end = tile_iter + iters_per_tile
            end = end_iter
            while (end < tile_iter_end and next_pid < NUM_SMS):
                while tl.load(locks + next_pid, cache_modifier=".cv", volatile=True) != 1:
                    pass
                rm1 = tl.arange(0, BLOCK_SIZE_M)
                rn1 = tl.arange(0, BLOCK_SIZE_N)
                rm1 = tl.max_contiguous(tl.multiple_of(rm1, BLOCK_SIZE_M), BLOCK_SIZE_M)
                rn1 = tl.max_contiguous(tl.multiple_of(rn1, BLOCK_SIZE_N), BLOCK_SIZE_N)
                P_ = P + next_pid * BLOCK_SIZE_M * BLOCK_SIZE_N + rm1[:, None] * BLOCK_SIZE_N + rn1[None, :]
                acc += tl.load(tl.multiple_of(P_, (1, 16)), cache_modifier=".cv")
                end += streamk_iters_pcu + (next_pid < streamk_remainder_iters)
                next_pid += 1

            c = acc.to(C.type.element_ty)
            if BIAS:
                c += bias[:, None]

            rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
            rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
            rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
            rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
            C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
            mask = (rm < M)[:, None] & (rn < N)[None, :]
            tl.store(C_, c, mask=mask)

        start_iter = end_iter

if __name__ == "__main__":
    import torch
    import triton
    import random

    torch.manual_seed(123)
    random.seed(123)

    total_sm = 304
    print(f"total SMs: {total_sm}")


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

            if matmul._debug:
                print(f"M,N,K={M},{N},{K} ; BLK_M,N,K={BLK_M},{BLK_N},{BLK_K}")
                print(f"{total_blocks_M=} x {total_blocks_N=} = {total_tiles=}")
                print(f"{total_tiles_streamk=} + {total_blocking_tiles=} = {total_tiles=}")
                print(f"{total_programs_streamk=}")
                print(f"{total_blocking_tiles=}")
                print(f"{total_full_tiles_streamk=}")
                print(f"{iters_per_tile=}")
                print(f"{total_iters_streamk=}")
                print("total_remainder_iters_streamk=", total_partial_tiles_streamk)
            
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
            if matmul._debug:
                print(f"{kk.n_regs} registers used, {kk.n_spills} spills")

            return c

        @staticmethod
        def forward(ctx, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, bias: torch.Tensor, P: torch.Tensor,
                    locks: torch.Tensor, grid: int, BLK_M=128, BLK_N=128, BLK_K=32, gsize_m=1, two_tiles=True, num_stages=3,
                    num_warps=4, waves_per_eu=2, mfmaInstrSize=16, kpack=1):
            matmul._call(a=a, b=b, c=c, bias=bias, P=P, locks=locks, total_programs_streamk=grid, BLK_M=BLK_M, BLK_N=BLK_N,
                        BLK_K=BLK_K, gsize_m=gsize_m, two_tiles=two_tiles, num_warps=num_warps, num_stages=num_stages,
                        waves_per_eu=waves_per_eu, mfmaInstrSize=mfmaInstrSize, kpack=kpack)
            return c

    m, n, k = 8192, 8192, 8192
    A = torch.randn(m, k, device="cuda", dtype=torch.float16)
    B = torch.randn(n, k, device="cuda", dtype=torch.float16).T
    C = torch.zeros((m, n), device="cuda", dtype=A.dtype)
    bias = torch.zeros((m, ), device="cuda", dtype=A.dtype)
    
    BLK_M = 256
    BLK_N = 256
    BLK_K = 64
    
    total_blocks_M = triton.cdiv(m, BLK_M)
    total_blocks_N = triton.cdiv(n, BLK_N)
    total_tiles = total_blocks_M * total_blocks_N
    
    gsize_m = 8
    two_tiles = 'True'
    num_stages = 2
    num_warps = 8
    waves_per_eu = 0
    mfmaInstrSize = 16
    kpack = 2
    
    print(f"{total_sm=}")
    matmul.set_debug(True)
    locks = torch.zeros((total_sm, ), device="cuda", dtype=torch.int32)
    P = torch.zeros((total_sm, BLK_M * BLK_N), device="cuda", dtype=torch.float32)
    
    C = matmul.apply(A, B, C, bias, P, locks, total_sm, BLK_M, BLK_N, BLK_K, gsize_m, two_tiles, num_stages, num_warps,
                 waves_per_eu, mfmaInstrSize, kpack)
    
    expected = A @ B
    
    assert torch.allclose(C, expected, atol=1), f"max: {(C - expected).abs().max().item()}\n{C}\n{expected}"
    print("pass validation test")