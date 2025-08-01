import torch
import triton
import argparse

from gravity_potential import gravity_potential_kernel


def gravity_potential(masses, pos, Nx, Ny, validate=False):
    grid = torch.zeros(Nx, Ny, device='cuda', dtype=torch.float32)
    n_masses = masses.shape[0]

    grid_launcher = (triton.cdiv(Nx, 32), triton.cdiv(Ny, 32))

    gravity_potential_kernel[grid_launcher](
        grid, masses, pos,
        Nx, Ny, n_masses,
        grid.stride(0), grid.stride(1),
        BLOCK_SIZE_X=32, BLOCK_SIZE_Y=32
    )

    if validate:
        grid_x, grid_y = torch.meshgrid(torch.arange(Nx, device='cuda'), torch.arange(Ny, device='cuda'), indexing='ij')
        potential_torch = torch.zeros_like(grid_x, dtype=torch.float32)
        for i in range(n_masses):
            dist_sq = (grid_x - pos[0, i])**2 + (grid_y - pos[1, i])**2
            potential_torch -= masses[i] / torch.sqrt(dist_sq + 1e-6)
        if torch.allclose(grid, potential_torch, atol=1e-2, rtol=0):
            print("Validation Successful!")
        else:
            print("Validation Failed!")
            print(f"max diff: {(grid - potential_torch).abs().max().item()}")

    return grid


def main(Nx=4096, Ny=4096, n_masses=16384, validate=False):
    masses = torch.rand(n_masses, device='cuda', dtype=torch.float32) * 100
    pos = torch.rand(2, n_masses, device='cuda', dtype=torch.float32)
    pos[0, :] *= Nx
    pos[1, :] *= Ny

    rep = 100

    for _ in range(10):
        gravity_potential(masses, pos, Nx, Ny)

    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    for _ in range(rep):
        gravity_potential(masses, pos, Nx, Ny)
    end_time.record()
    torch.cuda.synchronize()

    triton_time = start_time.elapsed_time(end_time) / rep
    print(f"Triton Gravitational Potential time: {triton_time:.4f} ms")

    if validate:
        gravity_potential(masses, pos, Nx, Ny, validate=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton Gravitational Potential Benchmark")
    parser.add_argument("--Nx", type=int, default=4096, help="Grid size in X dimension")
    parser.add_argument("--Ny", type=int, default=4096, help="Grid size in Y dimension")
    parser.add_argument("--n_masses", type=int, default=16384, help="Number of masses")
    parser.add_argument("--validate", action="store_true", help="Validate the Triton implementation against PyTorch.")
    args = parser.parse_args()

    main(args.Nx, args.Ny, args.n_masses, validate=args.validate) 