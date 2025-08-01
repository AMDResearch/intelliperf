#!/usr/bin/env python

import triton
import triton.language as tl


@triton.jit
def fdtd_2d_kernel(
    ex_ptr, ey_ptr, hz_ptr,
    Nx, Ny,
    stride_x, stride_y,
    BLOCK_SIZE_X: tl.constexpr, BLOCK_SIZE_Y: tl.constexpr
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)

    # Update H_z field
    hz_ptrs = hz_ptr + offsets_x[:, None] * stride_x + offsets_y[None, :] * stride_y
    mask_hz = (offsets_x[:, None] < Nx - 1) & (offsets_y[None, :] < Ny - 1)
    
    ex_here = tl.load(ex_ptr + offsets_x[:, None] * stride_x + (offsets_y[None, :] + 1) * stride_y, mask=mask_hz)
    ex_there = tl.load(ex_ptr + offsets_x[:, None] * stride_x + offsets_y[None, :] * stride_y, mask=mask_hz)
    ey_here = tl.load(ey_ptr + (offsets_x[:, None] + 1) * stride_x + offsets_y[None, :] * stride_y, mask=mask_hz)
    ey_there = tl.load(ey_ptr + offsets_x[:, None] * stride_x + offsets_y[None, :] * stride_y, mask=mask_hz)
    
    hz = tl.load(hz_ptrs, mask=mask_hz)
    hz = hz - 0.5 * (ey_here - ey_there - ex_here + ex_there)
    tl.store(hz_ptrs, hz, mask=mask_hz)

    # Update E_x field
    ex_ptrs = ex_ptr + offsets_x[:, None] * stride_x + offsets_y[None, :] * stride_y
    mask_ex = (offsets_x[:, None] < Nx) & (offsets_y[None, :] > 0) & (offsets_y[None, :] < Ny)
    
    hz_here = tl.load(hz_ptr + offsets_x[:, None] * stride_x + offsets_y[None, :] * stride_y, mask=mask_ex)
    hz_there = tl.load(hz_ptr + offsets_x[:, None] * stride_x + (offsets_y[None, :] - 1) * stride_y, mask=mask_ex)
    
    ex = tl.load(ex_ptrs, mask=mask_ex)
    ex = ex - 0.5 * (hz_here - hz_there)
    tl.store(ex_ptrs, ex, mask=mask_ex)
    
    # Update E_y field
    ey_ptrs = ey_ptr + offsets_x[:, None] * stride_x + offsets_y[None, :] * stride_y
    mask_ey = (offsets_x[:, None] > 0) & (offsets_x[:, None] < Nx) & (offsets_y[None, :] < Ny)

    hz_here = tl.load(hz_ptr + offsets_x[:, None] * stride_x + offsets_y[None, :] * stride_y, mask=mask_ey)
    hz_there = tl.load(hz_ptr + (offsets_x[:, None] - 1) * stride_x + offsets_y[None, :] * stride_y, mask=mask_ey)

    ey = tl.load(ey_ptrs, mask=mask_ey)
    ey = ey + 0.5 * (hz_here - hz_there)
    tl.store(ey_ptrs, ey, mask=mask_ey) 