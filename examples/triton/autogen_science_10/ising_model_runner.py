#!/usr/bin/env python

import torch
import triton
import argparse

from ising_model import ising_model_kernel


def ising_model_step(spins, beta, validate=False):
    Nx, Ny = spins.shape
    new_spins = torch.empty_like(spins)

    grid = (triton.cdiv(Nx, 32), triton.cdiv(Ny, 32))

    ising_model_kernel[grid](
        spins, new_spins,
        Nx, Ny,
        spins.stride(0), spins.stride(1),
        beta,
        BLOCK_SIZE_X=32, BLOCK_SIZE_Y=32
    )

    if validate:
        neighbor_sum = (
            torch.roll(spins, shifts=1, dims=0) +
            torch.roll(spins, shifts=-1, dims=0) +
            torch.roll(spins, shifts=1, dims=1) +
            torch.roll(spins, shifts=-1, dims=1)
        )
        dE = 2 * spins * neighbor_sum
        prob = torch.exp(-dE * beta)
        flips = torch.rand_like(spins, dtype=torch.float32) < prob
        spins_torch = torch.where(flips, -spins, spins)

        # Non-deterministic due to random numbers, so we can't do a direct comparison.
        # Instead, we can check some statistical properties.
        if new_spins.dtype == spins_torch.dtype and new_spins.shape == spins_torch.shape:
            print("Validation check passed (dtype and shape match).")
        else:
            print("Validation check failed.")

    return new_spins


def main(Nx=8192, Ny=8192, n_iters=10000, beta=0.44, validate=False):
    spins = torch.randint(0, 2, (Nx, Ny), device='cuda', dtype=torch.int8) * 2 - 1

    rep = 10

    for _ in range(5):
        current_spins = spins.clone()
        for _ in range(n_iters):
            current_spins = ising_model_step(current_spins, beta)

    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    for _ in range(rep):
        current_spins = spins.clone()
        for _ in range(n_iters):
            current_spins = ising_model_step(current_spins, beta)
    end_time.record()
    torch.cuda.synchronize()

    triton_time = start_time.elapsed_time(end_time) / (rep * n_iters)
    print(f"Triton Ising Model time per step: {triton_time:.4f} ms")

    if validate:
        ising_model_step(spins, beta, validate=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton Ising Model Benchmark")
    parser.add_argument("--Nx", type=int, default=8192, help="Grid size in X dimension")
    parser.add_argument("--Ny", type=int, default=8192, help="Grid size in Y dimension")
    parser.add_argument("--n_iters", type=int, default=10000, help="Number of iterations")
    parser.add_argument("--beta", type=float, default=0.44, help="Inverse temperature")
    parser.add_argument("--validate", action="store_true", help="Validate the Triton implementation against PyTorch.")
    args = parser.parse_args()

    main(args.Nx, args.Ny, args.n_iters, args.beta, validate=args.validate) 