# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple

# Note: The backward pass functions are kept for autograd but are assumed to exist.
# You would need to provide these files or replace them with your backward implementation.
# from aiter.ops.triton.mha_onekernel_bwd import flash_attn_onekernel_backward
# from aiter.ops.triton.mha_fused_bwd import flash_attn_fused_backward

# --- Globals for Kernel Behavior ---

_USE_FUSED_BWD_KERNEL = False
_USE_INT64_STRIDES = True

def mha_set_use_fused_bwd_kernel(value: bool):
    global _USE_FUSED_BWD_KERNEL
    _USE_FUSED_BWD_KERNEL = value

def mha_set_use_int64_strides(value: bool):
    global _USE_INT64_STRIDES
    _USE_INT64_STRIDES = value


# --- Triton JIT Helper Functions ---

@triton.jit
def _cdiv_fn(x, y):
    """Ceiling division utility function."""
    return (x + y - 1) // y

@triton.jit
def _load_fn(ptrs, offset_first, offset_second, boundary_first, boundary_second):
    """Generic masked load function."""
    if offset_first is not None and offset_second is not None:
        mask = (offset_first[:, None] < boundary_first) & (
            offset_second[None, :] < boundary_second
        )
        return tl.load(ptrs, mask=mask, other=0.0)
    elif offset_first is not None:
        mask = offset_first[:, None] < boundary_first
        return tl.load(ptrs, mask=mask, other=0.0)
    elif offset_second is not None:
        mask = offset_second[None, :] < boundary_second
        return tl.load(ptrs, mask=mask, other=0.0)
    else:
        return tl.load(ptrs)

@triton.jit
def swizzle_mha_wid_balanced(
    wid: tl.int32,
    NUM_Q_HEADS: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    NUM_XCD: tl.constexpr,
) -> tl.int32:
    """
    Encapsulates the balanced head-first mapping logic into a single swizzle function.
    Takes an original wid and returns a new wid that can be decomposed with simple math.
    """
    wids_per_batch = NUM_Q_HEADS * NUM_BLOCKS
    off_z = wid // wids_per_batch
    local_wid = wid % wids_per_batch

    heads_per_xcd_short = NUM_Q_HEADS // NUM_XCD
    heads_per_xcd_tall = heads_per_xcd_short + 1
    num_tall_xcds = NUM_Q_HEADS % NUM_XCD
    
    wids_per_tall_xcd = heads_per_xcd_tall * NUM_BLOCKS
    wids_per_short_xcd = heads_per_xcd_short * NUM_BLOCKS
    wids_in_tall_xcds_total = num_tall_xcds * wids_per_tall_xcd

    is_in_tall_group = local_wid < wids_in_tall_xcds_total
    
    # Tall Path
    xcd_idx_tall = local_wid // wids_per_tall_xcd
    local_wid_in_xcd_tall = local_wid % wids_per_tall_xcd
    target_q_head_tall = xcd_idx_tall * heads_per_xcd_tall + (local_wid_in_xcd_tall // NUM_BLOCKS)
    
    # Short Path
    wid_after_tall = local_wid - wids_in_tall_xcds_total
    xcd_local_idx_short = wid_after_tall // wids_per_short_xcd
    local_wid_in_xcd_short = wid_after_tall % wids_per_short_xcd
    target_q_head_short = (
        (num_tall_xcds * heads_per_xcd_tall) 
        + xcd_local_idx_short * heads_per_xcd_short 
        + (local_wid_in_xcd_short // NUM_BLOCKS)
    )

    local_wid_in_xcd = tl.where(is_in_tall_group, local_wid_in_xcd_tall, local_wid_in_xcd_short)
    target_start_m = local_wid_in_xcd % NUM_BLOCKS
    target_q_head = tl.where(is_in_tall_group, target_q_head_tall, target_q_head_short)
    
    swizzled_local_wid = target_q_head * NUM_BLOCKS + target_start_m
    
    return off_z * wids_per_batch + swizzled_local_wid

# --- Triton Kernels ---

@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i, q,
    k_ptrs, v_ptrs,
    stride_kn, stride_vk, stride_sn,
    start_m, seqlen_k, seqlen_q,
    dropout_p, sd_mask_ptrs, dropout_mask_ptrs,
    philox_seed, philox_ptrs,
    block_min, block_max, offs_n_causal,
    masked_blocks, n_extra_tokens,
    OFFS_M: tl.constexpr, OFFS_N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr, BLOCK_DMODEL_POW2: tl.constexpr,
    SM_SCALE: tl.constexpr, IS_CAUSAL: tl.constexpr,
    MASK_STEPS: tl.constexpr, ENABLE_DROPOUT: tl.constexpr,
    RETURN_SCORES: tl.constexpr, PADDED_HEAD: tl.constexpr
):
    """Inner loop for forward attention computation."""
    RCP_LN2: tl.constexpr = 1.4426950408889634

    for start_n in range(block_min, block_max, BLOCK_N):
        k_offs_n = start_n + tl.arange(0, BLOCK_N) if MASK_STEPS else None
        k_offs_k = tl.arange(0, BLOCK_DMODEL_POW2) if PADDED_HEAD else None
        k = _load_fn(k_ptrs, k_offs_k, k_offs_n, BLOCK_DMODEL, seqlen_k)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)

        mask = tl.full([BLOCK_M, BLOCK_N], 1, dtype=tl.int1)
        if MASK_STEPS:
            bound_cond = (start_n + BLOCK_N == block_max) and (n_extra_tokens != 0)
            boundary_m = tl.full([BLOCK_M], seqlen_k, dtype=tl.int32)
            size_n = start_n + OFFS_N[None, :]
            mask_partial = size_n < boundary_m[:, None]
            mask = tl.where(bound_cond, mask_partial, mask)

        if IS_CAUSAL:
            causal_boundary = start_n + offs_n_causal
            causal_mask = OFFS_M[:, None] >= causal_boundary[None, :]
            mask = mask & causal_mask

        qk = tl.where(mask, qk, float("-inf"))
        
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        m_ij_scaled = m_ij * SM_SCALE * RCP_LN2
        q_shifted = qk * SM_SCALE * RCP_LN2 - m_ij_scaled[:, None]
        p = tl.math.exp2(q_shifted)
        
        l_ij = tl.sum(p, 1)

        # Dropout logic
        if ENABLE_DROPOUT:
            q_mask = OFFS_M[:, None] < seqlen_q
            k_mask = (start_n + tl.arange(0, BLOCK_N))[None, :] < seqlen_k
            p_mask = q_mask & k_mask
            rng_output = tl.rand(philox_seed, philox_ptrs)
            dropout_mask = rng_output > dropout_p
            tl.store(dropout_mask_ptrs, dropout_mask, mask=p_mask)
            p = tl.where(dropout_mask, p, 0.0)
        elif RETURN_SCORES:
            tl.store(sd_mask_ptrs, p, mask=p_mask)

        m_diff_scaled = m_i * SM_SCALE * RCP_LN2 - m_ij_scaled
        alpha = tl.math.exp2(m_diff_scaled)
        acc = acc * alpha[:, None]
        
        v = _load_fn(v_ptrs, k_offs_n, k_offs_k, seqlen_k, BLOCK_DMODEL)
        acc += tl.dot(p.to(v.dtype), v)
        
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk
        if RETURN_SCORES:
            sd_mask_ptrs += BLOCK_N * stride_sn
        if ENABLE_DROPOUT:
            dropout_mask_ptrs += BLOCK_N * stride_sn
            philox_ptrs += BLOCK_N * stride_sn

    return acc, l_i, m_i

@triton.jit
def _attn_fwd(
    q_ptr, k_ptr, v_ptr, out_ptr,
    s_dmask_ptr, dropout_mask_ptr, softmax_lse_ptr,
    stride_qz_in, stride_qh_in, stride_qm_in, stride_qk_in,
    stride_kz_in, stride_kh_in, stride_kn_in, stride_kk_in,
    stride_vz_in, stride_vh_in, stride_vn_in, stride_vk_in,
    stride_oz_in, stride_oh_in, stride_om_in, stride_on_in,
    stride_sd_z_in, stride_sd_h_in, stride_sd_m_in, stride_sd_n_in,
    stride_lse_z_in, stride_lse_h_in, stride_lse_m_in,
    sm_scale, dropout_p, philox_seed, philox_offset_base_in,
    SEQLEN_Q: tl.constexpr, SEQLEN_K: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr, NUM_K_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr, BLOCK_DMODEL_POW2: tl.constexpr,
    RETURN_SCORES: tl.constexpr, ENABLE_DROPOUT: tl.constexpr,
    BATCH: tl.constexpr, NUM_XCD: tl.constexpr,
    USE_INT64_STRIDES: tl.constexpr  # <-- ADDED ARGUMENT
):
    """Main Triton kernel for forward attention."""
    NUM_BLOCKS = _cdiv_fn(SEQLEN_Q, BLOCK_M)
    wid = tl.program_id(0)

    swizzled_wid = swizzle_mha_wid_balanced(wid, NUM_Q_HEADS, NUM_BLOCKS, NUM_XCD)
    start_m = swizzled_wid % NUM_BLOCKS
    off_q_head = (swizzled_wid // NUM_BLOCKS) % NUM_Q_HEADS
    off_z = swizzled_wid // (NUM_Q_HEADS * NUM_BLOCKS)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL_POW2)

    # Stride casting for large tensors
    if USE_INT64_STRIDES: # <-- USE KERNEL ARGUMENT
        stride_qz, stride_qh, stride_qm, stride_qk = tl.cast(stride_qz_in, tl.int64), tl.cast(stride_qh_in, tl.int64), tl.cast(stride_qm_in, tl.int64), tl.cast(stride_qk_in, tl.int64)
        stride_kz, stride_kh, stride_kn, stride_kk = tl.cast(stride_kz_in, tl.int64), tl.cast(stride_kh_in, tl.int64), tl.cast(stride_kn_in, tl.int64), tl.cast(stride_kk_in, tl.int64)
        stride_vz, stride_vh, stride_vn, stride_vk = tl.cast(stride_vz_in, tl.int64), tl.cast(stride_vh_in, tl.int64), tl.cast(stride_vn_in, tl.int64), tl.cast(stride_vk_in, tl.int64)
        stride_oz, stride_oh, stride_om, stride_on = tl.cast(stride_oz_in, tl.int64), tl.cast(stride_oh_in, tl.int64), tl.cast(stride_om_in, tl.int64), tl.cast(stride_on_in, tl.int64)
        philox_offset_base = tl.cast(philox_offset_base_in, tl.int64)
        stride_sd_z, stride_sd_h, stride_sd_m, stride_sd_n = tl.cast(stride_sd_z_in, tl.int64), tl.cast(stride_sd_h_in, tl.int64), tl.cast(stride_sd_m_in, tl.int64), tl.cast(stride_sd_n_in, tl.int64)
        stride_lse_z, stride_lse_h, stride_lse_m = tl.cast(stride_lse_z_in, tl.int64), tl.cast(stride_lse_h_in, tl.int64), tl.cast(stride_lse_m_in, tl.int64)
    else:
        stride_qz, stride_qh, stride_qm, stride_qk = stride_qz_in, stride_qh_in, stride_qm_in, stride_qk_in
        stride_kz, stride_kh, stride_kn, stride_kk = stride_kz_in, stride_kh_in, stride_kn_in, stride_kk_in
        stride_vz, stride_vh, stride_vn, stride_vk = stride_vz_in, stride_vh_in, stride_vn_in, stride_vk_in
        stride_oz, stride_oh, stride_om, stride_on = stride_oz_in, stride_oh_in, stride_om_in, stride_on_in
        philox_offset_base = philox_offset_base_in
        stride_sd_z, stride_sd_h, stride_sd_m, stride_sd_n = stride_sd_z_in, stride_sd_h_in, stride_sd_m_in, stride_sd_n_in
        stride_lse_z, stride_lse_h, stride_lse_m = stride_lse_z_in, stride_lse_h_in, stride_lse_m_in

    seqlen_q, seqlen_k = SEQLEN_Q, SEQLEN_K
    n_blocks = _cdiv_fn(seqlen_k, BLOCK_N)

    # Early exit for causal masking
    if IS_CAUSAL:
        n_blocks_seqlen = _cdiv_fn((start_m + 1) * BLOCK_M + seqlen_k - seqlen_q, BLOCK_N)
        n_blocks = min(n_blocks, n_blocks_seqlen)
        if n_blocks <= 0:
            # Zero out the output and write LSE if this block is fully masked
            offs_out = off_z * stride_oz + off_q_head * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
            acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_POW2], dtype=out_ptr.type.element_ty)
            out_mask = (offs_m[:, None] < seqlen_q) & (offs_d[None, :] < BLOCK_DMODEL)
            tl.store(out_ptr + offs_out, acc, mask=out_mask)
            if softmax_lse_ptr is not None:
                offs_lse = off_z * stride_lse_z + off_q_head * stride_lse_h + offs_m * stride_lse_m
                lse_mask = offs_m < SEQLEN_Q
                lse = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)
                tl.store(softmax_lse_ptr + offs_lse, lse, mask=lse_mask)
            return

    # Grouped Query Attention logic
    grp_sz: tl.constexpr = NUM_Q_HEADS // NUM_K_HEADS
    off_k_head = off_q_head // grp_sz if grp_sz != 1 else off_q_head

    # Pointer setup
    q_offs = off_z * stride_qz + off_q_head * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_offs = off_z * stride_kz + off_k_head * stride_kh + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
    v_offs = off_z * stride_vz + off_k_head * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
    q_ptrs, k_ptrs, v_ptrs = q_ptr + q_offs, k_ptr + k_offs, v_ptr + v_offs

    s_dmask_ptrs, dropout_mask_ptrs, philox_ptrs = None, None, None
    if s_dmask_ptr is not None:
        s_dmask_offs = off_z * stride_sd_z + off_q_head * stride_sd_h + offs_m[:, None] * stride_sd_m + offs_n[None, :] * stride_sd_n
        s_dmask_ptrs = s_dmask_ptr + s_dmask_offs
    if dropout_mask_ptr is not None:
        dropout_mask_offs = off_z * stride_sd_z + off_q_head * stride_sd_h + offs_m[:, None] * stride_sd_m + offs_n[None, :] * stride_sd_n
        dropout_mask_ptrs = dropout_mask_ptr + dropout_mask_offs
        philox_ptrs = philox_offset_base + dropout_mask_offs

    # Initialize accumulators
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_POW2], dtype=tl.float32)

    q_mask = (offs_m[:, None] < seqlen_q) & (offs_d[None, :] < BLOCK_DMODEL)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Determine block partitioning for masking
    n_extra_tokens = seqlen_k % BLOCK_N or 0
    padded_block_k = n_extra_tokens != 0
    is_modulo_mn = not padded_block_k and (seqlen_q % BLOCK_M == 0)
    masked_blocks = BLOCK_M // BLOCK_N + (not is_modulo_mn) if IS_CAUSAL else padded_block_k
    masked_blocks = min(masked_blocks, n_blocks)
    n_full_blocks = n_blocks - masked_blocks

    # Main computation loop
    if n_full_blocks > 0:
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q, k_ptrs, v_ptrs,
            stride_kn, stride_vk, stride_sd_n,
            start_m, seqlen_k, seqlen_q,
            dropout_p, s_dmask_ptrs, dropout_mask_ptrs,
            philox_seed, philox_ptrs,
            0, n_full_blocks * BLOCK_N, 0, 0, 0,
            offs_m, offs_n, BLOCK_M, BLOCK_N, BLOCK_DMODEL, BLOCK_DMODEL_POW2,
            sm_scale, False, False, ENABLE_DROPOUT, RETURN_SCORES, BLOCK_DMODEL != BLOCK_DMODEL_POW2
        )
    
    if masked_blocks > 0:
        offs_n_causal = offs_n + (seqlen_q - seqlen_k) if IS_CAUSAL else 0
        k_ptrs += n_full_blocks * BLOCK_N * stride_kn
        v_ptrs += n_full_blocks * BLOCK_N * stride_vk
        if RETURN_SCORES: s_dmask_ptrs += n_full_blocks * BLOCK_N * stride_sd_n
        if ENABLE_DROPOUT: dropout_mask_ptrs += n_full_blocks * BLOCK_N * stride_sd_n
        
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q, k_ptrs, v_ptrs,
            stride_kn, stride_vk, stride_sd_n,
            start_m, seqlen_k, seqlen_q,
            dropout_p, s_dmask_ptrs, dropout_mask_ptrs,
            philox_seed, philox_ptrs,
            n_full_blocks * BLOCK_N, n_blocks * BLOCK_N, offs_n_causal,
            masked_blocks, n_extra_tokens,
            offs_m, offs_n, BLOCK_M, BLOCK_N, BLOCK_DMODEL, BLOCK_DMODEL_POW2,
            sm_scale, IS_CAUSAL, True, ENABLE_DROPOUT, RETURN_SCORES, BLOCK_DMODEL != BLOCK_DMODEL_POW2
        )

    # Epilogue
    l_recip = 1.0 / l_i[:, None]
    acc = acc * l_recip
    if ENABLE_DROPOUT:
        acc = acc * (1.0 / (1.0 - dropout_p))

    # Write LSE
    if softmax_lse_ptr is not None:
        LN2: tl.constexpr = 0.6931471824645996
        softmax_lse = m_i + tl.log(l_i)
        offs_lse = off_z * stride_lse_z + off_q_head * stride_lse_h + offs_m * stride_lse_m
        lse_mask = offs_m < seqlen_q
        tl.store(softmax_lse_ptr + offs_lse, softmax_lse, mask=lse_mask)

    # Write output
    offs_out = off_z * stride_oz + off_q_head * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    out_mask = (offs_m[:, None] < seqlen_q) & (offs_d[None, :] < BLOCK_DMODEL)
    tl.store(out_ptr + offs_out, acc.to(out_ptr.dtype.element_ty), mask=out_mask)


# --- Python Dispatcher and Autograd Function ---

def _flash_attn_forward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    dropout_p: float, softmax_scale: float, causal: bool,
    return_lse: bool, return_softmax: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Dispatcher for the forward pass."""
    o = torch.zeros_like(q)
    batch, seqlen_q, num_q_heads, head_sz = q.shape
    _, seqlen_k, num_k_heads, _ = k.shape

    q_strides = (q.stride(0), q.stride(2), q.stride(1), q.stride(3))
    k_strides = (k.stride(0), k.stride(2), k.stride(1), k.stride(3))
    v_strides = (v.stride(0), v.stride(2), v.stride(1), v.stride(3))
    o_strides = (o.stride(0), o.stride(2), o.stride(1), o.stride(3))

    BLOCK_DMODEL_POW2 = triton.next_power_of_2(head_sz)
    BLOCK_DMODEL_POW2 = max(BLOCK_DMODEL_POW2, 16)

    softmax_lse = torch.zeros((batch, num_q_heads, seqlen_q), device=q.device, dtype=torch.float32)
    stride_lse_z, stride_lse_h, stride_lse_m = softmax_lse.stride()

    enable_dropout = dropout_p > 0.0
    philox_seed, philox_offset = 0, 0
    if enable_dropout:
        philox_seed = torch.randint(0, 0xFFFFFF, (1,)).item()
        philox_offset = torch.randint(0, 0xFFFFFF, (1,)).item()

    s_dmask, dropout_mask = None, None
    if return_softmax or enable_dropout:
        s_dmask = torch.zeros((batch, num_q_heads, seqlen_q, seqlen_k), device=q.device, dtype=torch.float32)
        dropout_mask = torch.zeros((batch, num_q_heads, seqlen_q, seqlen_k), device=q.device, dtype=torch.bool)

    # Default kernel configuration
    config = {"BLOCK_M": 128, "BLOCK_N": 64, "waves_per_eu": 2, "num_warps": 4, "num_stages": 1}
    if enable_dropout or q.dtype == torch.float32:
        config = {"BLOCK_M": 32, "BLOCK_N": 32, "waves_per_eu": 1, "num_warps": 2, "num_stages": 1}

    grid = lambda META: (batch * num_q_heads * triton.cdiv(seqlen_q, META["BLOCK_M"]),)

    _attn_fwd[grid](
        q, k, v, o, s_dmask, dropout_mask, softmax_lse,
        *q_strides, *k_strides, *v_strides, *o_strides,
        s_dmask.stride(0) if s_dmask is not None else 0,
        s_dmask.stride(1) if s_dmask is not None else 0,
        s_dmask.stride(2) if s_dmask is not None else 0,
        s_dmask.stride(3) if s_dmask is not None else 0,
        stride_lse_z, stride_lse_h, stride_lse_m,
        softmax_scale, dropout_p, philox_seed, philox_offset,
        SEQLEN_Q=seqlen_q, SEQLEN_K=seqlen_k, IS_CAUSAL=causal,
        NUM_Q_HEADS=num_q_heads, NUM_K_HEADS=num_k_heads,
        BLOCK_DMODEL=head_sz, BLOCK_DMODEL_POW2=BLOCK_DMODEL_POW2,
        RETURN_SCORES=return_softmax, ENABLE_DROPOUT=enable_dropout,
        BATCH=batch, NUM_XCD=8, 
        USE_INT64_STRIDES=_USE_INT64_STRIDES, # <-- PASS ARGUMENT TO KERNEL
        **config
    )
    return o, softmax_lse, s_dmask, philox_seed, philox_offset

class _FlashAttnFunc(torch.autograd.Function):
    """PyTorch autograd function for standard attention."""
    @staticmethod
    def forward(ctx, q, k, v, dropout_p, softmax_scale, causal, return_lse, return_softmax):
        is_grad_enabled = torch.is_grad_enabled()
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5
        
        head_size_og = q.size(3)
        if head_size_og % 16 != 0: # Pad to 16-byte alignment
            pad_size = 16 - (head_size_og % 16)
            q = torch.nn.functional.pad(q, (0, pad_size))
            k = torch.nn.functional.pad(k, (0, pad_size))
            v = torch.nn.functional.pad(v, (0, pad_size))

        out_padded, softmax_lse, s_dmask, philox_seed, philox_offset = _flash_attn_forward(
            q, k, v, dropout_p, softmax_scale, causal, return_lse, return_softmax
        )

        if is_grad_enabled:
            ctx.save_for_backward(q, k, v, out_padded, softmax_lse)
            ctx.philox_seed = philox_seed
            ctx.philox_offset = philox_offset
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal

        out = out_padded[..., :head_size_og]
        result = [out]
        if return_lse: result.append(softmax_lse)
        if return_softmax: result.append(s_dmask)
        return result[0] if len(result) == 1 else tuple(result)

    @staticmethod
    def backward(ctx, do, *args):
        # Placeholder for backward pass.
        # You would integrate your backward kernel call here.
        q, k, v, out, softmax_lse = ctx.saved_tensors
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        
        head_size_og = do.size(3)
        if head_size_og % 16 != 0:
             pad_size = 16 - (head_size_og % 16)
             do = torch.nn.functional.pad(do, (0, pad_size))

        # Example call to a backward function (needs to be provided)
        # flash_attn_onekernel_backward(
        #     do, q, k, v, out, softmax_lse, dq, dk, dv,
        #     ctx.softmax_scale, None, ctx.causal, None, None,
        #     q.shape[1], k.shape[1], ctx.dropout_p,
        #     ctx.philox_seed, ctx.philox_offset, _USE_INT64_STRIDES
        # )
        
        dq = dq[..., :q.shape[-1]]
        dk = dk[..., :k.shape[-1]]
        dv = dv[..., :v.shape[-1]]
        
        return dq, dk, dv, None, None, None, None, None

def flash_attn_func(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    dropout_p: float = 0.0, softmax_scale: Optional[float] = None,
    causal: bool = False, return_lse: bool = False,
    return_attn_probs: bool = False
):
    """
    Public-facing function for flash attention.
    
    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T. Default to 1/sqrt(headdim).
        causal: bool. Whether to apply causal attention mask.
        return_lse: bool. Whether to return the log-sum-exp of the attention scores.
        return_attn_probs: bool. Whether to return the attention probabilities (for testing).
    
    Returns:
        A tensor of shape (batch_size, seqlen, nheads, headdim).
    """
    return _FlashAttnFunc.apply(
        q, k, v, dropout_p, softmax_scale, causal,
        return_lse, return_attn_probs
    )
