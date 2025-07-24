#!/usr/bin/env python

import torch
import triton
import triton.language as tl
import argparse

@triton.jit
def lbm_d2q9_kernel(
    fin_ptr, fout_ptr,
    Nx, Ny,
    stride_q, stride_x, stride_y,
    omega,
    BLOCK_SIZE_X: tl.constexpr, BLOCK_SIZE_Y: tl.constexpr
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)

    # D2Q9 weights and velocities
    W = [4./9., 1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., 1./36.]
    CXS = [0, 1, 0, -1, 0, 1, -1, -1, 1]
    CYS = [0, 0, 1, 0, -1, 1, 1, -1, -1]
    
    # Manually unroll the loop for loading distributions
    mask = (offsets_x[None,:] < Nx) & (offsets_y[:,None] < Ny)
    f0_ptrs = fin_ptr + (0 * stride_q + offsets_x[None,:] * stride_x + offsets_y[:,None] * stride_y)
    f0 = tl.load(f0_ptrs, mask=mask, other=0.0)
    f1_ptrs = fin_ptr + (1 * stride_q + offsets_x[None,:] * stride_x + offsets_y[:,None] * stride_y)
    f1 = tl.load(f1_ptrs, mask=mask, other=0.0)
    f2_ptrs = fin_ptr + (2 * stride_q + offsets_x[None,:] * stride_x + offsets_y[:,None] * stride_y)
    f2 = tl.load(f2_ptrs, mask=mask, other=0.0)
    f3_ptrs = fin_ptr + (3 * stride_q + offsets_x[None,:] * stride_x + offsets_y[:,None] * stride_y)
    f3 = tl.load(f3_ptrs, mask=mask, other=0.0)
    f4_ptrs = fin_ptr + (4 * stride_q + offsets_x[None,:] * stride_x + offsets_y[:,None] * stride_y)
    f4 = tl.load(f4_ptrs, mask=mask, other=0.0)
    f5_ptrs = fin_ptr + (5 * stride_q + offsets_x[None,:] * stride_x + offsets_y[:,None] * stride_y)
    f5 = tl.load(f5_ptrs, mask=mask, other=0.0)
    f6_ptrs = fin_ptr + (6 * stride_q + offsets_x[None,:] * stride_x + offsets_y[:,None] * stride_y)
    f6 = tl.load(f6_ptrs, mask=mask, other=0.0)
    f7_ptrs = fin_ptr + (7 * stride_q + offsets_x[None,:] * stride_x + offsets_y[:,None] * stride_y)
    f7 = tl.load(f7_ptrs, mask=mask, other=0.0)
    f8_ptrs = fin_ptr + (8 * stride_q + offsets_x[None,:] * stride_x + offsets_y[:,None] * stride_y)
    f8 = tl.load(f8_ptrs, mask=mask, other=0.0)

    # Collision
    rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8
    ux = (f1 - f3 + f5 - f6 - f7 + f8) / rho
    uy = (f2 - f4 + f5 + f6 - f7 - f8) / rho

    # Unroll the loop for calculating fout
    cu0 = CXS[0] * ux + CYS[0] * uy
    feq0 = W[0] * rho * (1 + 3*cu0 + 4.5*cu0*cu0 - 1.5*(ux*ux + uy*uy))
    fout0 = f0 - omega * (f0 - feq0)
    cu1 = CXS[1] * ux + CYS[1] * uy
    feq1 = W[1] * rho * (1 + 3*cu1 + 4.5*cu1*cu1 - 1.5*(ux*ux + uy*uy))
    fout1 = f1 - omega * (f1 - feq1)
    cu2 = CXS[2] * ux + CYS[2] * uy
    feq2 = W[2] * rho * (1 + 3*cu2 + 4.5*cu2*cu2 - 1.5*(ux*ux + uy*uy))
    fout2 = f2 - omega * (f2 - feq2)
    cu3 = CXS[3] * ux + CYS[3] * uy
    feq3 = W[3] * rho * (1 + 3*cu3 + 4.5*cu3*cu3 - 1.5*(ux*ux + uy*uy))
    fout3 = f3 - omega * (f3 - feq3)
    cu4 = CXS[4] * ux + CYS[4] * uy
    feq4 = W[4] * rho * (1 + 3*cu4 + 4.5*cu4*cu4 - 1.5*(ux*ux + uy*uy))
    fout4 = f4 - omega * (f4 - feq4)
    cu5 = CXS[5] * ux + CYS[5] * uy
    feq5 = W[5] * rho * (1 + 3*cu5 + 4.5*cu5*cu5 - 1.5*(ux*ux + uy*uy))
    fout5 = f5 - omega * (f5 - feq5)
    cu6 = CXS[6] * ux + CYS[6] * uy
    feq6 = W[6] * rho * (1 + 3*cu6 + 4.5*cu6*cu6 - 1.5*(ux*ux + uy*uy))
    fout6 = f6 - omega * (f6 - feq6)
    cu7 = CXS[7] * ux + CYS[7] * uy
    feq7 = W[7] * rho * (1 + 3*cu7 + 4.5*cu7*cu7 - 1.5*(ux*ux + uy*uy))
    fout7 = f7 - omega * (f7 - feq7)
    cu8 = CXS[8] * ux + CYS[8] * uy
    feq8 = W[8] * rho * (1 + 3*cu8 + 4.5*cu8*cu8 - 1.5*(ux*ux + uy*uy))
    fout8 = f8 - omega * (f8 - feq8)

    # Streaming
    # Manually unroll the streaming step
    x_out0, y_out0 = offsets_x + CXS[0], offsets_y + CYS[0]
    mask_out0 = (x_out0[None,:] >= 0) & (x_out0[None,:] < Nx) & (y_out0[:,None] >= 0) & (y_out0[:,None] < Ny)
    fout0_ptrs = fout_ptr + (0 * stride_q + x_out0[None,:] * stride_x + y_out0[:,None] * stride_y)
    tl.store(fout0_ptrs, fout0, mask=mask_out0)
    x_out1, y_out1 = offsets_x + CXS[1], offsets_y + CYS[1]
    mask_out1 = (x_out1[None,:] >= 0) & (x_out1[None,:] < Nx) & (y_out1[:,None] >= 0) & (y_out1[:,None] < Ny)
    fout1_ptrs = fout_ptr + (1 * stride_q + x_out1[None,:] * stride_x + y_out1[:,None] * stride_y)
    tl.store(fout1_ptrs, fout1, mask=mask_out1)
    x_out2, y_out2 = offsets_x + CXS[2], offsets_y + CYS[2]
    mask_out2 = (x_out2[None,:] >= 0) & (x_out2[None,:] < Nx) & (y_out2[:,None] >= 0) & (y_out2[:,None] < Ny)
    fout2_ptrs = fout_ptr + (2 * stride_q + x_out2[None,:] * stride_x + y_out2[:,None] * stride_y)
    tl.store(fout2_ptrs, fout2, mask=mask_out2)
    x_out3, y_out3 = offsets_x + CXS[3], offsets_y + CYS[3]
    mask_out3 = (x_out3[None,:] >= 0) & (x_out3[None,:] < Nx) & (y_out3[:,None] >= 0) & (y_out3[:,None] < Ny)
    fout3_ptrs = fout_ptr + (3 * stride_q + x_out3[None,:] * stride_x + y_out3[:,None] * stride_y)
    tl.store(fout3_ptrs, fout3, mask=mask_out3)
    x_out4, y_out4 = offsets_x + CXS[4], offsets_y + CYS[4]
    mask_out4 = (x_out4[None,:] >= 0) & (x_out4[None,:] < Nx) & (y_out4[:,None] >= 0) & (y_out4[:,None] < Ny)
    fout4_ptrs = fout_ptr + (4 * stride_q + x_out4[None,:] * stride_x + y_out4[:,None] * stride_y)
    tl.store(fout4_ptrs, fout4, mask=mask_out4)
    x_out5, y_out5 = offsets_x + CXS[5], offsets_y + CYS[5]
    mask_out5 = (x_out5[None,:] >= 0) & (x_out5[None,:] < Nx) & (y_out5[:,None] >= 0) & (y_out5[:,None] < Ny)
    fout5_ptrs = fout_ptr + (5 * stride_q + x_out5[None,:] * stride_x + y_out5[:,None] * stride_y)
    tl.store(fout5_ptrs, fout5, mask=mask_out5)
    x_out6, y_out6 = offsets_x + CXS[6], offsets_y + CYS[6]
    mask_out6 = (x_out6[None,:] >= 0) & (x_out6[None,:] < Nx) & (y_out6[:,None] >= 0) & (y_out6[:,None] < Ny)
    fout6_ptrs = fout_ptr + (6 * stride_q + x_out6[None,:] * stride_x + y_out6[:,None] * stride_y)
    tl.store(fout6_ptrs, fout6, mask=mask_out6)
    x_out7, y_out7 = offsets_x + CXS[7], offsets_y + CYS[7]
    mask_out7 = (x_out7[None,:] >= 0) & (x_out7[None,:] < Nx) & (y_out7[:,None] >= 0) & (y_out7[:,None] < Ny)
    fout7_ptrs = fout_ptr + (7 * stride_q + x_out7[None,:] * stride_x + y_out7[:,None] * stride_y)
    tl.store(fout7_ptrs, fout7, mask=mask_out7)
    x_out8, y_out8 = offsets_x + CXS[8], offsets_y + CYS[8]
    mask_out8 = (x_out8[None,:] >= 0) & (x_out8[None,:] < Nx) & (y_out8[:,None] >= 0) & (y_out8[:,None] < Ny)
    fout8_ptrs = fout_ptr + (8 * stride_q + x_out8[None,:] * stride_x + y_out8[:,None] * stride_y)
    tl.store(fout8_ptrs, fout8, mask=mask_out8)
        
def lbm_d2q9_step(fin, omega):
    Nx, Ny = fin.shape[1], fin.shape[2]
    fout = torch.empty_like(fin)
    
    grid = (triton.cdiv(Nx, 32), triton.cdiv(Ny, 32))

    lbm_d2q9_kernel[grid](
        fin, fout,
        Nx, Ny,
        fin.stride(0), fin.stride(1), fin.stride(2),
        omega,
        BLOCK_SIZE_X=32, BLOCK_SIZE_Y=32
    )
    return fout

def main(Nx=2048, Ny=2048, n_iters=100, omega=1.0):
    # Initial state: 9 distributions, (Nx, Ny) grid
    fin = torch.ones(9, Nx, Ny, device='cuda', dtype=torch.float32) / 9.0
    
    rep = 10
    
    # Warm-up
    for _ in range(5):
        current_fin = fin.clone()
        for _ in range(n_iters):
            current_fin = lbm_d2q9_step(current_fin, omega)

    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    for _ in range(rep):
        current_fin = fin.clone()
        for _ in range(n_iters):
            current_fin = lbm_d2q9_step(current_fin, omega)
    end_time.record()
    torch.cuda.synchronize()

    triton_time = start_time.elapsed_time(end_time) / (rep * n_iters)
    print(f"Triton LBM-D2Q9 time per step: {triton_time:.4f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton LBM-D2Q9 Benchmark")
    parser.add_argument("--Nx", type=int, default=2048, help="Grid size in X dimension")
    parser.add_argument("--Ny", type=int, default=2048, help="Grid size in Y dimension")
    parser.add_argument("--n_iters", type=int, default=100, help="Number of iterations")
    parser.add_argument("--omega", type=float, default=1.0, help="Relaxation parameter")
    args = parser.parse_args()
    
    main(args.Nx, args.Ny, args.n_iters, args.omega) 