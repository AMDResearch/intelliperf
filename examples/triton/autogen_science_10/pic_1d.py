#!/usr/bin/env python

import torch
import triton
import triton.language as tl
import argparse

@triton.jit
def pic_1d_kernel(
    pos_ptr, vel_ptr, E_ptr,
    new_pos_ptr, new_vel_ptr,
    n_particles, n_grid,
    dx, dt,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_particles

    # Load particle positions and velocities
    pos = tl.load(pos_ptr + offsets, mask=mask)
    vel = tl.load(vel_ptr + offsets, mask=mask)

    # Gather: Find grid index and interpolation factor
    grid_idx = (pos / dx).to(tl.int32)
    w = (pos / dx) - grid_idx.to(tl.float32)

    # Gather electric field from the grid
    E0 = tl.load(E_ptr + grid_idx, mask=(grid_idx >= 0) & (grid_idx < n_grid))
    E1 = tl.load(E_ptr + grid_idx + 1, mask=(grid_idx + 1 >= 0) & (grid_idx + 1 < n_grid))
    E_particle = (1 - w) * E0 + w * E1

    # Push particles
    new_vel = vel + E_particle * dt
    new_pos = pos + new_vel * dt

    # Enforce periodic boundary conditions
    new_pos = new_pos % (n_grid * dx)

    # Store new positions and velocities
    tl.store(new_pos_ptr + offsets, new_pos, mask=mask)
    tl.store(new_vel_ptr + offsets, new_vel, mask=mask)

def pic_1d_step(pos, vel, E, dx, dt):
    n_particles = pos.shape[0]
    n_grid = E.shape[0]
    new_pos = torch.empty_like(pos)
    new_vel = torch.empty_like(vel)

    grid = (triton.cdiv(n_particles, 1024),)
    
    pic_1d_kernel[grid](
        pos, vel, E,
        new_pos, new_vel,
        n_particles, n_grid,
        dx, dt,
        BLOCK_SIZE=1024
    )
    return new_pos, new_vel

def main(n_particles=2**20, n_grid=2**14, n_iters=100, dt=0.1):
    dx = 1.0

    pos = torch.rand(n_particles, device='cuda', dtype=torch.float32) * n_grid * dx
    vel = torch.randn(n_particles, device='cuda', dtype=torch.float32)
    E = torch.randn(n_grid, device='cuda', dtype=torch.float32)

    rep = 10
    
    # Warm-up
    for _ in range(5):
        current_pos, current_vel = pos.clone(), vel.clone()
        for _ in range(n_iters):
            current_pos, current_vel = pic_1d_step(current_pos, current_vel, E, dx, dt)
    
    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    for _ in range(rep):
        current_pos, current_vel = pos.clone(), vel.clone()
        for _ in range(n_iters):
            current_pos, current_vel = pic_1d_step(current_pos, current_vel, E, dx, dt)
    end_time.record()
    torch.cuda.synchronize()

    triton_time = start_time.elapsed_time(end_time) / (rep * n_iters)
    print(f"Triton 1D PIC time per step: {triton_time:.4f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton 1D PIC Benchmark")
    parser.add_argument("--n_particles", type=int, default=2**20, help="Number of particles")
    parser.add_argument("--n_grid", type=int, default=2**14, help="Number of grid points")
    parser.add_argument("--n_iters", type=int, default=100, help="Number of iterations")
    parser.add_argument("--dt", type=float, default=0.1, help="Time step")
    args = parser.parse_args()
    
    main(args.n_particles, args.n_grid, args.n_iters, args.dt) 