
import torch
import triton
import triton.language as tl

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
    w = tl.to_tensor([4./9., 1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., 1./36.], dtype=tl.float32)
    cxs = tl.to_tensor([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=tl.int32)
    cys = tl.to_tensor([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=tl.int32)

    # Pointers to the 9 distributions for a block
    fin_ptrs = fin_ptr + (offsets_x[None, :, None] * stride_x + 
                          offsets_y[None, None, :] * stride_y +
                          tl.arange(0, 9)[:, None, None] * stride_q)
    
    mask = (offsets_x[None, :, None] < Nx) & (offsets_y[None, None, :] < Ny)
    f = tl.load(fin_ptrs, mask=mask)

    # Collision step
    rho = tl.sum(f, axis=0)
    ux = tl.sum(f * cxs[:, None, None], axis=0) / rho
    uy = tl.sum(f * cys[:, None, None], axis=0) / rho
    
    feq = tl.zeros((9, BLOCK_SIZE_X, BLOCK_SIZE_Y), dtype=tl.float32)
    for i in range(9):
        cu = cxs[i] * ux + cys[i] * uy
        feq[i,:,:] = w[i] * rho * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * (ux * ux + uy * uy))
    
    fout = f - omega * (f - feq)
    
    # Streaming step (writing to fout)
    for i in range(9):
        x_out = (offsets_x + cxs[i])
        y_out = (offsets_y + cys[i])

        mask_out = (x_out >= 0) & (x_out < Nx) & (y_out >= 0) & (y_out < Ny)
        
        fout_ptrs = fout_ptr + (x_out[None, :, None] * stride_x +
                                y_out[None, None, :] * stride_y +
                                i * stride_q)
        tl.store(fout_ptrs, fout[i,:,:], mask=mask_out)


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

def main():
    Nx, Ny = 2048, 2048
    omega = 1.0 # Relaxation parameter
    
    # Initial state: 9 distributions, (Nx, Ny) grid
    fin = torch.ones(9, Nx, Ny, device='cuda', dtype=torch.float32) / 9.0
    
    n_iters = 100
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
    main() 