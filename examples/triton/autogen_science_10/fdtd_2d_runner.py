#!/usr/bin/env python

import torch
import triton
import argparse

from fdtd_2d import fdtd_2d_kernel


def fdtd_2d_step(ex, ey, hz, validate=False):
    Nx, Ny = ex.shape

    grid = (triton.cdiv(Nx, 32), triton.cdiv(Ny, 32))

    ex_new = ex.clone()
    ey_new = ey.clone()
    hz_new = hz.clone()

    fdtd_2d_kernel[grid](
        ex_new, ey_new, hz_new,
        Nx, Ny,
        ex.stride(0), ex.stride(1),
        BLOCK_SIZE_X=32, BLOCK_SIZE_Y=32
    )

    if validate:
        hz_torch = hz - 0.5 * (torch.roll(ey, -1, 0) - ey - (torch.roll(ex, -1, 1) - ex))
        ex_torch = ex - 0.5 * (hz - torch.roll(hz, 1, 1))
        ey_torch = ey + 0.5 * (hz - torch.roll(hz, 1, 0))

        if torch.allclose(ex_new, ex_torch) and torch.allclose(ey_new, ey_torch) and torch.allclose(hz_new, hz_torch):
            print("Validation Successful!")
        else:
            print("Validation Failed!")
            print(f"max diff ex: {(ex_new - ex_torch).abs().max().item()}")
            print(f"max diff ey: {(ey_new - ey_torch).abs().max().item()}")
            print(f"max diff hz: {(hz_new - hz_torch).abs().max().item()}")

    return ex_new, ey_new, hz_new


def main(Nx=8192, Ny=8192, n_iters=100, validate=False):
    ex = torch.zeros(Nx, Ny, device='cuda', dtype=torch.float32)
    ey = torch.zeros(Nx, Ny, device='cuda', dtype=torch.float32)
    hz = torch.zeros(Nx, Ny, device='cuda', dtype=torch.float32)

    hz[Nx // 2, Ny // 2] = 1.0

    rep = 10

    for _ in range(5):
        current_ex, current_ey, current_hz = ex.clone(), ey.clone(), hz.clone()
        for _ in range(n_iters):
            current_ex, current_ey, current_hz = fdtd_2d_step(current_ex, current_ey, current_hz)

    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    for _ in range(rep):
        current_ex, current_ey, current_hz = ex.clone(), ey.clone(), hz.clone()
        for _ in range(n_iters):
            current_ex, current_ey, current_hz = fdtd_2d_step(current_ex, current_ey, current_hz)
    end_time.record()
    torch.cuda.synchronize()

    triton_time = start_time.elapsed_time(end_time) / (rep * n_iters)
    print(f"Triton 2D-FDTD time per step: {triton_time:.4f} ms")

    if validate:
        fdtd_2d_step(ex, ey, hz, validate=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton 2D-FDTD Benchmark")
    parser.add_argument("--Nx", type=int, default=8192, help="Grid size in X dimension")
    parser.add_argument("--Ny", type=int, default=8192, help="Grid size in Y dimension")
    parser.add_argument("--n_iters", type=int, default=100, help="Number of iterations")
    parser.add_argument("--validate", action="store_true", help="Validate the Triton implementation against PyTorch.")
    args = parser.parse_args()

    main(args.Nx, args.Ny, args.n_iters, validate=args.validate) 