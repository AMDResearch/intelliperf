#!/usr/bin/env python

import torch
import triton
import argparse

from black_scholes import black_scholes_kernel


def black_scholes_step(s, v, r, sigma, dt, validate=False):
    n_assets = s.shape[0]
    n_timesteps = 1
    new_v = torch.empty_like(v)

    grid = (triton.cdiv(n_assets, 1024),)

    black_scholes_kernel[grid](
        s, v, new_v,
        n_assets, n_timesteps,
        r, sigma, dt,
        s.stride(0), v.stride(0),
        BLOCK_SIZE=1024
    )

    if validate:
        a = 0.5 * dt * (sigma**2 * s**2 - r * s)
        b = 1 - dt * (sigma**2 * s**2 + r)
        c = 0.5 * dt * (sigma**2 * s**2 + r * s)

        v_prev = torch.roll(v, 1)
        v_curr = v
        v_next = torch.roll(v, -1)
        v_prev[0] = 0
        v_next[-1] = 0

        v_torch = a * v_prev + b * v_curr + c * v_next
        if torch.allclose(new_v, v_torch, atol=1e-2, rtol=0):
            print("Validation Successful!")
        else:
            print("Validation Failed!")
            print(f"max diff: {(new_v - v_torch).abs().max().item()}")

    return new_v


def main(n_assets=2**28, n_timesteps=10000000, r=0.2, sigma=0.6, validate=False):
    T = 1.0
    dt = T / n_timesteps

    s = torch.linspace(1, 200, n_assets, device='cuda', dtype=torch.float32)
    K = 100.0
    v = torch.clamp(s - K, min=0)

    rep = 10

    for _ in range(5):
        black_scholes_step(s, v.clone(), r, sigma, dt)

    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    for _ in range(rep):
        black_scholes_step(s, v.clone(), r, sigma, dt)
    end_time.record()
    torch.cuda.synchronize()

    triton_time = start_time.elapsed_time(end_time) / rep
    print(f"Triton Black-Scholes (1 step) time: {triton_time:.4f} ms")

    if validate:
        black_scholes_step(s, v.clone(), r, sigma, dt, validate=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton Black-Scholes Benchmark")
    parser.add_argument("--n_assets", type=int, default=2**28, help="Number of assets")
    parser.add_argument("--n_timesteps", type=int, default=10000000, help="Number of timesteps")
    parser.add_argument("--r", type=float, default=0.2, help="Risk-free interest rate")
    parser.add_argument("--sigma", type=float, default=0.6, help="Volatility")
    parser.add_argument("--validate", action="store_true", help="Validate the Triton implementation against PyTorch.")
    args = parser.parse_args()

    main(args.n_assets, args.n_timesteps, args.r, args.sigma, validate=args.validate) 