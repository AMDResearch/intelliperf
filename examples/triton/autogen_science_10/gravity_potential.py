#!/usr/bin/env python

import torch
import triton
import triton.language as tl
import argparse

@triton.jit
def gravity_potential_kernel(
    grid_ptr, masses_ptr, pos_ptr,
    Nx, Ny, n_masses,
    stride_x, stride_y,
    BLOCK_SIZE_X: tl.constexpr, BLOCK_SIZE_Y: tl.constexpr
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)

    # Grid point coordinates
    grid_x = offsets_x[:, None]
    grid_y = offsets_y[None, :]

    potential = tl.zeros((BLOCK_SIZE_X, BLOCK_SIZE_Y), dtype=tl.float32)

    for i in range(0, n_masses):
        mass = tl.load(masses_ptr + i)
        mass_x = tl.load(pos_ptr + i)
        mass_y = tl.load(pos_ptr + n_masses + i)
        
        dx = grid_x - mass_x
        dy = grid_y - mass_y
        
        dist_sq = dx*dx + dy*dy
        dist = tl.sqrt(dist_sq + 1e-6) # Add epsilon to avoid division by zero
        
        potential -= mass / dist

    grid_ptrs = grid_ptr + (offsets_x[:, None] * stride_x + offsets_y[None, :] * stride_y)
    mask = (offsets_x[:, None] < Nx) & (offsets_y[None, :] < Ny)
    tl.store(grid_ptrs, potential, mask=mask)


def gravity_potential(masses, pos, Nx, Ny):
    grid = torch.zeros(Nx, Ny, device='cuda', dtype=torch.float32)
    n_masses = masses.shape[0]
    
    grid_launcher = (triton.cdiv(Nx, 32), triton.cdiv(Ny, 32))
    
    gravity_potential_kernel[grid_launcher](
        grid, masses, pos,
        Nx, Ny, n_masses,
        grid.stride(0), grid.stride(1),
        BLOCK_SIZE_X=32, BLOCK_SIZE_Y=32
    )
    return grid

def main(Nx=4096, Ny=4096, n_masses=16384):
    masses = torch.rand(n_masses, device='cuda', dtype=torch.float32) * 100
    pos = torch.rand(2, n_masses, device='cuda', dtype=torch.float32)
    pos[0, :] *= Nx
    pos[1, :] *= Ny

    rep = 100
    
    # Warm-up
    for _ in range(10):
        potential_grid = gravity_potential(masses, pos, Nx, Ny)

    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    for _ in range(rep):
        potential_grid = gravity_potential(masses, pos, Nx, Ny)
    end_time.record()
    torch.cuda.synchronize()

    triton_time = start_time.elapsed_time(end_time) / rep
    print(f"Triton Gravitational Potential time: {triton_time:.4f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton Gravitational Potential Benchmark")
    parser.add_argument("--Nx", type=int, default=4096, help="Grid size in X dimension")
    parser.add_argument("--Ny", type=int, default=4096, help="Grid size in Y dimension")
    parser.add_argument("--n_masses", type=int, default=16384, help="Number of masses")
    args = parser.parse_args()
    
    main(args.Nx, args.Ny, args.n_masses) 