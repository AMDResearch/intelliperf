#!/usr/bin/env python

import torch
import triton
import triton.language as tl
import argparse

@triton.jit
def n_body_kernel(
    pos_ptr, vel_ptr,
    new_pos_ptr, new_vel_ptr,
    n_particles,
    dt, eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_particles

    # Load particle positions and velocities
    px = tl.load(pos_ptr + offsets, mask=mask)
    py = tl.load(pos_ptr + n_particles + offsets, mask=mask)
    pz = tl.load(pos_ptr + 2 * n_particles + offsets, mask=mask)
    
    vx = tl.load(vel_ptr + offsets, mask=mask)
    vy = tl.load(vel_ptr + n_particles + offsets, mask=mask)
    vz = tl.load(vel_ptr + 2 * n_particles + offsets, mask=mask)

    # Accumulators for force
    fx = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    fy = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    fz = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Compute forces
    for i in range(0, n_particles):
        other_px = tl.load(pos_ptr + i)
        other_py = tl.load(pos_ptr + n_particles + i)
        other_pz = tl.load(pos_ptr + 2 * n_particles + i)

        dx = other_px - px
        dy = other_py - py
        dz = other_pz - pz

        dist_sq = dx*dx + dy*dy + dz*dz + eps
        inv_dist = 1.0 / tl.sqrt(dist_sq)
        inv_dist3 = inv_dist * inv_dist * inv_dist

        # Accumulate forces
        fx += dx * inv_dist3
        fy += dy * inv_dist3
        fz += dz * inv_dist3

    # Update velocity and position
    new_vx = vx + dt * fx
    new_vy = vy + dt * fy
    new_vz = vz + dt * fz
    
    new_px = px + dt * new_vx
    new_py = py + dt * new_vy
    new_pz = pz + dt * new_vz

    # Store new positions and velocities
    tl.store(new_pos_ptr + offsets, new_px, mask=mask)
    tl.store(new_pos_ptr + n_particles + offsets, new_py, mask=mask)
    tl.store(new_pos_ptr + 2 * n_particles + offsets, new_pz, mask=mask)

    tl.store(new_vel_ptr + offsets, new_vx, mask=mask)
    tl.store(new_vel_ptr + n_particles + offsets, new_vy, mask=mask)
    tl.store(new_vel_ptr + 2 * n_particles + offsets, new_vz, mask=mask)


def n_body_simulation(pos, vel, dt, eps):
    n_particles = pos.shape[1]
    new_pos = torch.empty_like(pos)
    new_vel = torch.empty_like(vel)
    
    grid = lambda META: (triton.cdiv(n_particles, META['BLOCK_SIZE']),)
    
    n_body_kernel[grid](
        pos, vel,
        new_pos, new_vel,
        n_particles, dt, eps,
        BLOCK_SIZE=1024
    )
    return new_pos, new_vel

def main(n_particles=32768, dt=0.01, eps=1e-6):
    pos = torch.randn(3, n_particles, device='cuda', dtype=torch.float32)
    vel = torch.randn(3, n_particles, device='cuda', dtype=torch.float32)

    rep = 100
    
    # Warm-up
    for _ in range(10):
        new_pos, new_vel = n_body_simulation(pos, vel, dt, eps)

    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    for _ in range(rep):
        new_pos, new_vel = n_body_simulation(pos, vel, dt, eps)
    end_time.record()
    torch.cuda.synchronize()

    triton_time = start_time.elapsed_time(end_time) / rep
    print(f"Triton N-Body Simulation time: {triton_time:.4f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton N-Body Simulation Benchmark")
    parser.add_argument("--n_particles", type=int, default=32768, help="Number of particles")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step")
    parser.add_argument("--eps", type=float, default=1e-6, help="Softening factor")
    args = parser.parse_args()
    
    main(args.n_particles, args.dt, args.eps) 