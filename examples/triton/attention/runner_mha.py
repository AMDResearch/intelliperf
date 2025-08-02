#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import argparse
import sys
from mha_unswizzle import flash_attn_func # Assuming your MHA code is in 'mha.py'

class MHALayer(torch.nn.Module):
    """
    A wrapper layer for the custom MHA implementation.
    """
    def __init__(self, causal=False, softmax_scale=None):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale

    def forward(self, q, k, v):
        """
        Calls the custom flash_attn_func.
        
        Args:
            q (torch.Tensor): Query tensor of shape (batch, seqlen_q, heads_q, dim)
            k (torch.Tensor): Key tensor of shape (batch, seqlen_k, heads_k, dim)
            v (torch.Tensor): Value tensor of shape (batch, seqlen_k, heads_k, dim)
        
        Returns:
            torch.Tensor: The output of the attention operation.
        """
        # If softmax_scale is not provided, calculate it from head dimension
        softmax_scale = self.softmax_scale or (1.0 / (q.size(-1)**0.5))

        # The custom function is not a class method, so we call it directly.
        return flash_attn_func(
            q,
            k,
            v,
            causal=self.causal,
            softmax_scale=softmax_scale,
        )

def main(args):
    """
    Main function to run the MHA benchmark and validation.
    """
    torch.manual_seed(123)

    # Test case configuration from your request
    # -b 1 -hq 16 -hk 16 -sq 65536 -sk 65536 -d 128 -layout bshd
    # fused-attention-fwd-D_HEAD-128-layout-bshd-fp8-False-causal-False
    
    # Unpack arguments
    batch_size = args.b
    num_q_heads = args.hq
    num_k_heads = args.hk
    seq_len_q = args.sq
    seq_len_k = args.sk
    head_dim = args.d
    causal = args.causal
    dtype = torch.float16 if args.fp16 else torch.float32

    print("--- MHA Benchmark Configuration ---")
    print(f"  Batch Size: {batch_size}")
    print(f"  Q Heads: {num_q_heads}, K Heads: {num_k_heads}")
    print(f"  Q SeqLen: {seq_len_q}, K SeqLen: {seq_len_k}")
    print(f"  Head Dimension: {head_dim}")
    print(f"  Data Type: {'float16' if args.fp16 else 'float32'}")
    print(f"  Causal: {causal}")
    print("------------------------------------")

    # Instantiate the MHA layer
    mha_layer = MHALayer(causal=causal).to('cuda')

    # Create random input tensors on the GPU
    # Layout is 'bshd' -> (batch, seqlen, heads, dim)
    q = torch.randn((batch_size, seq_len_q, num_q_heads, head_dim), device='cuda', dtype=dtype)
    k = torch.randn((batch_size, seq_len_k, num_k_heads, head_dim), device='cuda', dtype=dtype)
    v = torch.randn((batch_size, seq_len_k, num_k_heads, head_dim), device='cuda', dtype=dtype)
    
    # Warm-up run
    for _ in range(5):
        _ = mha_layer(q, k, v)
    
    torch.cuda.synchronize()

    # --- Benchmarking ---
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    for _ in range(args.repeat):
        output_triton = mha_layer(q, k, v)
    end_time.record()
    
    torch.cuda.synchronize()
    triton_time = start_time.elapsed_time(end_time) / args.repeat
    print(f"\nCustom MHA Implementation Time: {triton_time:.4f} ms")

    # --- Validation ---
    if args.validate:
        print("\nRunning validation against torch.nn.functional.scaled_dot_product_attention...")
        try:
            # Prepare tensors for PyTorch's reference implementation
            # It expects (batch, heads, seqlen, dim)
            q_ref = q.permute(0, 2, 1, 3)
            k_ref = k.permute(0, 2, 1, 3)
            v_ref = v.permute(0, 2, 1, 3)

            output_torch = torch.nn.functional.scaled_dot_product_attention(
                q_ref, k_ref, v_ref, is_causal=causal
            ).permute(0, 2, 1, 3) # Permute back to bshd

            # Set tolerance based on data type
            atol = 1e-1 if args.fp16 else 1e-4
            rtol = 1e-2 if args.fp16 else 1e-5

            if torch.allclose(output_triton, output_torch, atol=atol, rtol=rtol):
                print("✅ Validation Successful!")
            else:
                print("❌ Validation Failed: Triton output does not match PyTorch output.")
                max_diff = (output_triton - output_torch).abs().max().item()
                print(f"   Max absolute difference: {max_diff}")
                sys.exit(1)
        except torch.cuda.OutOfMemoryError:
            print("\n⚠️ Validation Failed: PyTorch reference implementation ran out of memory.")
            print("   This is expected for large sequence lengths and highlights the benefit of your custom kernel.")
        except Exception as e:
            print(f"\n❌ An error occurred during validation: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton MHA Runner and Benchmark")
    # MHA dimensions
    parser.add_argument("--b", type=int, default=1, help="Batch size")
    parser.add_argument("--hq", type=int, default=16, help="Number of heads for Query")
    parser.add_argument("--hk", type=int, default=16, help="Number of heads for Key/Value")
    parser.add_argument("--sq", type=int, default=65536, help="Sequence length for Query")
    parser.add_argument("--sk", type=int, default=65536, help="Sequence length for Key/Value")
    parser.add_argument("--d", type=int, default=128, help="Head dimension")
    
    # MHA options
    parser.add_argument("--causal", action="store_true", help="Enable causal masking")
    parser.add_argument("--fp16", action="store_true", help="Use float16 data type")
    
    # Runner options
    parser.add_argument("--validate", action="store_true", help="Validate the Triton implementation against PyTorch.")
    parser.add_argument("--repeat", type=int, default=10, help="Number of repetitions for performance measurement")
    
    args = parser.parse_args()
    main(args)
