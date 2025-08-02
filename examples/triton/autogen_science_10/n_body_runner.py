#!/usr/bin/env python

import torch
import triton
import argparse

from n_body import n_body_kernel


def n_body_simulation(pos, vel, dt, eps, validate=False):
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

    if validate:
        dx = pos[0, :, None] - pos[0, None, :]
        dy = pos[1, :, None] - pos[1, None, :]
        dz = pos[2, :, None] - pos[2, None, :]
        dist_sq = dx**2 + dy**2 + dz**2 + eps
        inv_dist = 1.0 / torch.sqrt(dist_sq)
        inv_dist3 = inv_dist**3
        fx = torch.sum(-dx * inv_dist3, dim=1)
        fy = torch.sum(-dy * inv_dist3, dim=1)
        fz = torch.sum(-dz * inv_dist3, dim=1)
        new_vel_torch = vel.clone()
        new_vel_torch[0, :] += dt * fx
        new_vel_torch[1, :] += dt * fy
        new_vel_torch[2, :] += dt * fz
        new_pos_torch = pos + dt * new_vel_torch

        if torch.allclose(new_pos, new_pos_torch, atol=1e-1, rtol=0) and torch.allclose(new_vel, new_vel_torch, atol=1e-1, rtol=0):
            print("Validation Successful!")
        else:
            print("Validation Failed!")
            print(f"max diff pos: {(new_pos - new_pos_torch).abs().max().item()}")
            print(f"max diff vel: {(new_vel - new_vel_torch).abs().max().item()}")

    return new_pos, new_vel


def main(n_particles=32768, dt=0.01, eps=1e-6, validate=False):
    pos = torch.randn(3, n_particles, device='cuda', dtype=torch.float32)
    vel = torch.randn(3, n_particles, device='cuda', dtype=torch.float32)

    rep = 10

    for _ in range(10):
        n_body_simulation(pos, vel, dt, eps)

    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    for _ in range(rep):
        n_body_simulation(pos, vel, dt, eps)
    end_time.record()
    torch.cuda.synchronize()

    triton_time = start_time.elapsed_time(end_time) / rep
    print(f"Triton N-Body Simulation time: {triton_time:.4f} ms")

    if validate:
        n_body_simulation(pos, vel, dt, eps, validate=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton N-Body Simulation Benchmark")
    parser.add_argument("--n_particles", type=int, default=32768, help="Number of particles")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step")
    parser.add_argument("--eps", type=float, default=1e-6, help="Softening factor")
    parser.add_argument("--validate", action="store_true", help="Validate the Triton implementation against PyTorch.")
    args = parser.parse_args()

    main(args.n_particles, args.dt, args.eps, validate=args.validate) 