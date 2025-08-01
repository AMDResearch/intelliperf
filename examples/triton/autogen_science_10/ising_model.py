#!/usr/bin/env python

import triton
import triton.language as tl


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