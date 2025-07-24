#!/usr/bin/env python

import torch
import triton
import triton.language as tl
import argparse

@triton.jit
def molecular_dynamics_kernel(
    pos_ptr, force_ptr,
    n_particles,
    sigma, epsilon,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_particles

    px = tl.load(pos_ptr + offsets, mask=mask)
    py = tl.load(pos_ptr + n_particles + offsets, mask=mask)
    pz = tl.load(pos_ptr + 2 * n_particles + offsets, mask=mask)

    fx = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    fy = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    fz = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for i in range(0, n_particles):
        other_px = tl.load(pos_ptr + i)
        other_py = tl.load(pos_ptr + n_particles + i)
        other_pz = tl.load(pos_ptr + 2 * n_particles + i)

        dx = other_px - px
        dy = other_py - py
        dz = other_pz - pz

        dist_sq = dx*dx + dy*dy + dz*dz
        dist_sq = tl.where(dist_sq == 0, 1e-12, dist_sq) # Avoid division by zero

        # Lennard-Jones potential calculation
        sr2 = (sigma * sigma) / dist_sq
        sr6 = sr2 * sr2 * sr2
        sr12 = sr6 * sr6
        
        force_scalar = 24 * epsilon * (2 * sr12 - sr6) / dist_sq
        
        fx += force_scalar * dx
        fy += force_scalar * dy
        fz += force_scalar * dz

    tl.store(force_ptr + offsets, fx, mask=mask)
    tl.store(force_ptr + n_particles + offsets, fy, mask=mask)
    tl.store(force_ptr + 2 * n_particles + offsets, fz, mask=mask)


def molecular_dynamics(pos, sigma, epsilon):
    n_particles = pos.shape[1]
    forces = torch.empty_like(pos)
    
    grid = lambda META: (triton.cdiv(n_particles, META['BLOCK_SIZE']),)
    
    molecular_dynamics_kernel[grid](
        pos, forces,
        n_particles,
        sigma, epsilon,
        BLOCK_SIZE=1024
    )
    return forces

def main(n_particles=32768, sigma=1.0, epsilon=1.0):
    pos = torch.randn(3, n_particles, device='cuda', dtype=torch.float32)

    rep = 100
    
    # Warm-up
    for _ in range(10):
        forces = molecular_dynamics(pos, sigma, epsilon)

    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    for _ in range(rep):
        forces = molecular_dynamics(pos, sigma, epsilon)
    end_time.record()
    torch.cuda.synchronize()

    triton_time = start_time.elapsed_time(end_time) / rep
    print(f"Triton Molecular Dynamics time: {triton_time:.4f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton Molecular Dynamics Benchmark")
    parser.add_argument("--n_particles", type=int, default=32768, help="Number of particles")
    parser.add_argument("--sigma", type=float, default=1.0, help="LJ sigma parameter")
    parser.add_argument("--epsilon", type=float, default=1.0, help="LJ epsilon parameter")
    args = parser.parse_args()
    
    main(args.n_particles, args.sigma, args.epsilon) 