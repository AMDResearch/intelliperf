#!/usr/bin/env python

import torch
import triton
import triton.language as tl
import argparse

@triton.jit
def ising_model_kernel(
    spins_ptr, new_spins_ptr,
    Nx, Ny,
    stride_x, stride_y,
    beta,
    BLOCK_SIZE_X: tl.constexpr, BLOCK_SIZE_Y: tl.constexpr
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)

    spins_ptrs = spins_ptr + (offsets_x[:, None] * stride_x + offsets_y[None, :] * stride_y)
    mask = (offsets_x[:, None] < Nx) & (offsets_y[None, :] < Ny)
    
    current_spin = tl.load(spins_ptrs, mask=mask)

    # Load neighbors with periodic boundary conditions
    up = tl.load(spins_ptr + (offsets_x[:, None] * stride_x + ((offsets_y[None, :] - 1 + Ny) % Ny) * stride_y), mask=mask)
    down = tl.load(spins_ptr + (offsets_x[:, None] * stride_x + ((offsets_y[None, :] + 1) % Ny) * stride_y), mask=mask)
    left = tl.load(spins_ptr + (((offsets_x[:, None] - 1 + Nx) % Nx) * stride_x + offsets_y[None, :] * stride_y), mask=mask)
    right = tl.load(spins_ptr + (((offsets_x[:, None] + 1) % Nx) * stride_x + offsets_y[None, :] * stride_y), mask=mask)
    
    # Calculate energy change if spin is flipped
    neighbor_sum = up + down + left + right
    dE = 2 * current_spin * neighbor_sum
    
    # Metropolis-Hastings update rule
    prob = tl.exp(-dE * beta)
    
    # Generate random numbers
    seed = pid_x * BLOCK_SIZE_X + pid_y * BLOCK_SIZE_Y
    rand_offsets = tl.arange(0, BLOCK_SIZE_X * BLOCK_SIZE_Y)
    rand = tl.rand(seed, rand_offsets)
    rand = tl.reshape(rand, (BLOCK_SIZE_X, BLOCK_SIZE_Y))
    
    new_spin = tl.where(rand < prob, -current_spin, current_spin)
    
    new_spins_ptrs = new_spins_ptr + (offsets_x[:, None] * stride_x + offsets_y[None, :] * stride_y)
    tl.store(new_spins_ptrs, new_spin, mask=mask)


def ising_model_step(spins, beta):
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
    return new_spins

def main(Nx=4096, Ny=4096, n_iters=100, beta=0.44):
    spins = torch.randint(0, 2, (Nx, Ny), device='cuda', dtype=torch.int8) * 2 - 1
    
    rep = 10
    
    # Warm-up
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton Ising Model Benchmark")
    parser.add_argument("--Nx", type=int, default=4096, help="Grid size in X dimension")
    parser.add_argument("--Ny", type=int, default=4096, help="Grid size in Y dimension")
    parser.add_argument("--n_iters", type=int, default=100, help="Number of iterations")
    parser.add_argument("--beta", type=float, default=0.44, help="Inverse temperature")
    args = parser.parse_args()
    
    main(args.Nx, args.Ny, args.n_iters, args.beta) 