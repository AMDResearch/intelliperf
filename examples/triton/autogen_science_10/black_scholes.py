#!/usr/bin/env python

import triton
import triton.language as tl


@triton.jit
def black_scholes_kernel(
    s_ptr, v_ptr,
    new_v_ptr,
    n_assets, n_timesteps,
    r, sigma, dt,
    stride_s, stride_t,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_assets

    s = tl.load(s_ptr + offsets, mask=mask)
    
    # Finite difference coefficients
    a = 0.5 * dt * (sigma*sigma*s*s - r*s)
    b = 1 - dt * (sigma*sigma*s*s + r)
    c = 0.5 * dt * (sigma*sigma*s*s + r*s)

    # Simplified step: in a real solver, this would be part of a loop
    # over timesteps, often working backwards from maturity.
    v_prev = tl.load(v_ptr + offsets - stride_s, mask=(offsets > 0) & mask)
    v_curr = tl.load(v_ptr + offsets, mask=mask)
    v_next = tl.load(v_ptr + offsets + stride_s, mask=(offsets < n_assets - 1) & mask)

    new_v = a * v_prev + b * v_curr + c * v_next
    
    tl.store(new_v_ptr + offsets, new_v, mask=mask) 