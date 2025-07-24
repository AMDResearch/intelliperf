#!/usr/bin/env python

import torch
import triton
import triton.language as tl
import argparse

@triton.jit
def black_scholes_kernel(
    s_ptr, v_ptr,
    new_v_ptr,
    n_assets, n_timesteps,
    r, sigma, dt,
    stride_s, stride_t,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_assets

    s = tl.load(s_ptr + offsets, mask=mask)
    
    # Finite difference coefficients
    a = 0.5 * dt * (sigma*sigma*s*s - r*s)
    b = 1 - dt * (sigma*sigma*s*s + r)
    c = 0.5 * dt * (sigma*sigma*s*s + r*s)

    # Simplified step: in a real solver, this would be part of a loop
    # over timesteps, often working backwards from maturity.
    v_prev = tl.load(v_ptr + offsets - stride_s, mask=(offsets > 0) & mask)
    v_curr = tl.load(v_ptr + offsets, mask=mask)
    v_next = tl.load(v_ptr + offsets + stride_s, mask=(offsets < n_assets - 1) & mask)

    new_v = a * v_prev + b * v_curr + c * v_next
    
    tl.store(new_v_ptr + offsets, new_v, mask=mask)


def black_scholes_step(s, v, r, sigma, dt):
    n_assets = s.shape[0]
    n_timesteps = 1 # Not used in this simplified kernel
    new_v = torch.empty_like(v)
    
    grid = (triton.cdiv(n_assets, 1024),)
    
    black_scholes_kernel[grid](
        s, v, new_v,
        n_assets, n_timesteps,
        r, sigma, dt,
        s.stride(0), v.stride(0), # Simplified strides
        BLOCK_SIZE=1024
    )
    return new_v

def main(n_assets=2**18, n_timesteps=1000, r=0.05, sigma=0.2):
    T = 1.0       # Time to maturity
    dt = T / n_timesteps

    s = torch.linspace(1, 200, n_assets, device='cuda', dtype=torch.float32)
    # Initial condition: value at maturity (e.g., for a call option)
    K = 100.0 # Strike price
    v = torch.clamp(s - K, min=0)

    rep = 100
    
    # Warm-up
    for _ in range(5):
        current_v = v.clone()
        # A full solver would loop n_timesteps times
        current_v = black_scholes_step(s, current_v, r, sigma, dt)

    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    for _ in range(rep):
        current_v = v.clone()
        # A full solver would loop n_timesteps times
        current_v = black_scholes_step(s, current_v, r, sigma, dt)
    end_time.record()
    torch.cuda.synchronize()

    triton_time = start_time.elapsed_time(end_time) / rep
    print(f"Triton Black-Scholes (1 step) time: {triton_time:.4f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton Black-Scholes Benchmark")
    parser.add_argument("--n_assets", type=int, default=2**18, help="Number of assets")
    parser.add_argument("--n_timesteps", type=int, default=1000, help="Number of timesteps")
    parser.add_argument("--r", type=float, default=0.05, help="Risk-free interest rate")
    parser.add_argument("--sigma", type=float, default=0.2, help="Volatility")
    args = parser.parse_args()
    
    main(args.n_assets, args.n_timesteps, args.r, args.sigma) 