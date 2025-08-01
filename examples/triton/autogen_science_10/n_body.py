#!/usr/bin/env python

import triton
import triton.language as tl


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