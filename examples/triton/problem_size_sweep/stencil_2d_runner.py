#!/usr/bin/env python

import torch
import triton
import triton.language as tl
import argparse

# Define the swizzled and unswizzled kernels
from stencil_2d_swizzle import stencil_2d_kernel as stencil_2d_swizzle
from stencil_2d_unswizzle import stencil_2d_kernel as stencil_2d_unswizzle

def run_stencil_2d(grid_size, version):
    grid = torch.randn((grid_size, grid_size), device='cuda', dtype=torch.float32)
    result = torch.empty_like(grid)
    
    if version == "swizzle":
        kernel_to_run = stencil_2d_swizzle
    elif version == "unswizzle":
        kernel_to_run = stencil_2d_unswizzle
    else:
        raise ValueError("Invalid version specified. Choose 'swizzle' or 'unswizzle'.")

    grid_ = (triton.cdiv(grid_size, 32), triton.cdiv(grid_size, 32))
    
    kernel_to_run[grid_](
        grid, result,
        grid_size, grid_size,
        grid.stride(0), grid.stride(1),
        BLOCK_SIZE_M=32, BLOCK_SIZE_N=32
    )
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 2D Stencil kernel with a specified grid size.")
    parser.add_argument('--grid_size', type=int, required=True, help='Size of the grid (grid_size x grid_size)')
    parser.add_argument('--version', type=str, required=True, choices=['swizzle', 'unswizzle'], help='Kernel version to run')
    
    args = parser.parse_args()
    
    run_stencil_2d(args.grid_size, args.version) 