#!/usr/bin/env python

import torch
import triton
import triton.language as tl
import math

@triton.jit
def fused_attention_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    D_HEAD: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    q_offset = off_hz * stride_qh
    k_offset = off_hz * stride_kh
    v_offset = off_hz * stride_vh
    o_offset = off_hz * stride_oh

    Q_block_ptr = tl.make_block_ptr(
        base=q_ptr + q_offset,
        shape=(N_CTX, D_HEAD),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, D_HEAD),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=k_ptr + k_offset,
        shape=(D_HEAD, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(D_HEAD, BLOCK_SIZE_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=v_ptr + v_offset,
        shape=(N_CTX, D_HEAD),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_N, D_HEAD),
        order=(1, 0)
    )
    O_block_ptr = tl.make_block_ptr(
        base=o_ptr + o_offset,
        shape=(N_CTX, D_HEAD),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, D_HEAD),
        order=(1, 0)
    )

    q = tl.load(Q_block_ptr, boundary_check=(0, 1))
    
    l_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32) - float('inf')
    m_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32) - float('inf')
    acc = tl.zeros((BLOCK_SIZE_M, D_HEAD), dtype=tl.float32)

    for start_n in range(0, (start_m + 1) * BLOCK_SIZE_M, BLOCK_SIZE_N):
        k = tl.load(K_block_ptr, boundary_check=(0, 1))
        
        s_ij = tl.dot(q, k, out_dtype=tl.float32) / math.sqrt(D_HEAD)

        m_ij = tl.max(s_ij, 1)
        p_ij = tl.exp(s_ij - m_ij[:, None])
        l_ij = tl.sum(p_ij, 1)

        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_ij - m_new)
        
        l_new = alpha * l_i + beta * l_ij
        
        p_safe_ij = p_ij / l_ij[:, None]
        
        v = tl.load(V_block_ptr, boundary_check=(0, 1))
        
        acc_scale = l_i / l_new
        acc_scale = tl.where(l_i == 0, 0, acc_scale) # handle first iteration
        acc = acc * acc_scale[:, None]
        
        acc += tl.dot(p_safe_ij.to(v.dtype), v)

        l_i = l_new
        m_i = m_new

        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_N, 0))

    tl.store(O_block_ptr, acc.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))

def fused_attention(q, k, v):
    Z, H, N_CTX, D_HEAD = q.shape
    o = torch.empty_like(q)

    grid = (triton.cdiv(N_CTX, 128), Z * H)
    
    fused_attention_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        Z, H, N_CTX,
        D_HEAD=D_HEAD,
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=64
    )
    return o

def main():
    Z, H, N_CTX, D_HEAD = 32, 48, 4096, 64
    q = torch.randn((Z, H, N_CTX, D_HEAD), device='cuda', dtype=torch.float16)
    k = torch.randn((Z, H, N_CTX, D_HEAD), device='cuda', dtype=torch.float16)
    v = torch.randn((Z, H, N_CTX, D_HEAD), device='cuda', dtype=torch.float16)

    rep = 100
    
    for _ in range(10):
        o_triton = fused_attention(q, k, v)

    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    for _ in range(rep):
        o_triton = fused_attention(q, k, v)
    end_time.record()
    torch.cuda.synchronize()

    triton_time = start_time.elapsed_time(end_time) / rep
    print(f"Triton fused attention time: {triton_time:.4f} ms")

    # For verification (a simplified version)
    # def torch_attention(q, k, v):
    #     scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D_HEAD)
    #     attn = torch.nn.functional.softmax(scores, dim=-1)
    #     output = torch.matmul(attn, v)
    #     return output
    #
    # o_torch = torch_attention(q, k, v)
    # assert torch.allclose(o_triton, o_torch, atol=1e-1, rtol=0), "Triton and PyTorch results differ"

if __name__ == "__main__":
    main() 