#!/usr/bin/env python

import triton
import triton.language as tl


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