#!/usr/bin/env python

import torch
import triton
import triton.language as tl
import argparse

@triton.jit
def jacobi_3d_kernel(
    grid_ptr, new_grid_ptr,
    Nx, Ny, Nz,
    stride_x, stride_y, stride_z,
    BLOCK_SIZE_X: tl.constexpr, BLOCK_SIZE_Y: tl.constexpr, BLOCK_SIZE_Z: tl.constexpr
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_z = tl.program_id(2)

    offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    offsets_z = pid_z * BLOCK_SIZE_Z + tl.arange(0, BLOCK_SIZE_Z)

    grid_ptrs = grid_ptr + (offsets_x[:, None, None] * stride_x + 
                            offsets_y[None, :, None] * stride_y + 
                            offsets_z[None, None, :] * stride_z)

    # Stencil operation for 3D Jacobi
    center = tl.load(grid_ptrs, mask=(offsets_x[:, None, None] < Nx) & 
                                     (offsets_y[None, :, None] < Ny) & 
                                     (offsets_z[None, None, :] < Nz), other=0.0)

    # Load neighbors, with boundary checks
    left = tl.load(grid_ptrs - stride_x, mask=(offsets_x[:, None, None] > 0), other=0.0)
    right = tl.load(grid_ptrs + stride_x, mask=(offsets_x[:, None, None] < Nx - 1), other=0.0)
    
    front = tl.load(grid_ptrs - stride_y, mask=(offsets_y[None, :, None] > 0), other=0.0)
    back = tl.load(grid_ptrs + stride_y, mask=(offsets_y[None, :, None] < Ny - 1), other=0.0)

    down = tl.load(grid_ptrs - stride_z, mask=(offsets_z[None, None, :] > 0), other=0.0)
    up = tl.load(grid_ptrs + stride_z, mask=(offsets_z[None, None, :] < Nz - 1), other=0.0)

    new_val = (center + left + right + front + back + down + up) / 7.0

    new_grid_ptrs = new_grid_ptr + (offsets_x[:, None, None] * stride_x + 
                                    offsets_y[None, :, None] * stride_y + 
                                    offsets_z[None, None, :] * stride_z)

    tl.store(new_grid_ptrs, new_val, mask=(offsets_x[:, None, None] < Nx) & 
                                        (offsets_y[None, :, None] < Ny) & 
                                        (offsets_z[None, None, :] < Nz))

def jacobi_3d_step(grid):
    Nx, Ny, Nz = grid.shape
    new_grid = torch.empty_like(grid)

    launch_grid = (triton.cdiv(Nx, 16), triton.cdiv(Ny, 16), triton.cdiv(Nz, 16))
    
    jacobi_3d_kernel[launch_grid](
        grid, new_grid,
        Nx, Ny, Nz,
        grid.stride(0), grid.stride(1), grid.stride(2),
        BLOCK_SIZE_X=16, BLOCK_SIZE_Y=16, BLOCK_SIZE_Z=4
    )
    return new_grid

def main(Nx=256, Ny=256, Nz=256, n_iters=10):
    grid = torch.randn(Nx, Ny, Nz, device='cuda', dtype=torch.float32)
    
    rep = 10
    
    # Warm-up
    for _ in range(5):
        current_grid = grid.clone()
        for _ in range(n_iters):
            current_grid = jacobi_3d_step(current_grid)

    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    for _ in range(rep):
        current_grid = grid.clone()
        for _ in range(n_iters):
            current_grid = jacobi_3d_step(current_grid)
    end_time.record()
    torch.cuda.synchronize()

    triton_time = start_time.elapsed_time(end_time) / (rep * n_iters)
    print(f"Triton 3D Jacobi Solver time per step: {triton_time:.4f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton 3D Jacobi Solver Benchmark")
    parser.add_argument("--Nx", type=int, default=256, help="Grid size in X dimension")
    parser.add_argument("--Ny", type=int, default=256, help="Grid size in Y dimension")
    parser.add_argument("--Nz", type=int, default=256, help="Grid size in Z dimension")
    parser.add_argument("--n_iters", type=int, default=10, help="Number of Jacobi iterations")
    args = parser.parse_args()
    
    main(args.Nx, args.Ny, args.Nz, args.n_iters) 