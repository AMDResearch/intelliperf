#!/usr/bin/env python

import torch
import triton
import triton.language as tl
import numpy as np
from scipy.sparse import csr_matrix
import argparse

@triton.jit
def spmv_kernel(
    y_ptr, x_ptr,
    data_ptr, indices_ptr, indptr_ptr,
    M,
    BLOCK_SIZE: tl.constexpr
):
    num_XCD = 8
    num_blocks = M
    pid = tl.program_id(0)
    
    # New swizzling pattern
    original_xcd = pid % num_XCD
    iteration = pid // num_XCD
    new_pid = (iteration + original_xcd * (num_blocks // num_XCD)) % num_blocks
    
    row_start = tl.load(indptr_ptr + new_pid)
    row_end = tl.load(indptr_ptr + new_pid + 1)
    
    acc = 0.0
    for i in range(row_start, row_end, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < row_end
        
        col_indices = tl.load(indices_ptr + offsets, mask=mask)
        data = tl.load(data_ptr + offsets, mask=mask)
        x_vals = tl.load(x_ptr + col_indices, mask=mask)
        
        acc += tl.sum(data * x_vals)
        
    tl.store(y_ptr + new_pid, acc)

def spmv(x, data, indices, indptr):
    M = indptr.size(0) - 1
    y = torch.empty(M, device=x.device, dtype=x.dtype)

    grid = (M,)
    
    spmv_kernel[grid](
        y, x,
        data, indices, indptr,
        M,
        BLOCK_SIZE=128
    )
    return y

def main(M=8192, N=8192, density=0.01):
    # Generate a random sparse matrix in COO format
    sparse_matrix = csr_matrix(np.random.randn(M, N) * (np.random.rand(M, N) < density))
    
    data = torch.from_numpy(sparse_matrix.data).to('cuda').to(torch.float16)
    indices = torch.from_numpy(sparse_matrix.indices).to('cuda').to(torch.int32)
    indptr = torch.from_numpy(sparse_matrix.indptr).to('cuda').to(torch.int32)
    
    x = torch.randn(N, device='cuda', dtype=torch.float16)

    rep = 100
    
    for _ in range(10):
        y_triton = spmv(x, data, indices, indptr)

    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    for _ in range(rep):
        y_triton = spmv(x, data, indices, indptr)
    end_time.record()
    torch.cuda.synchronize()

    triton_time = start_time.elapsed_time(end_time) / rep
    print(f"Triton SpMV time: {triton_time:.4f} ms")

    # For verification
    # dense_matrix = torch.from_numpy(sparse_matrix.toarray()).to('cuda').to(torch.float16)
    # y_torch = torch.mv(dense_matrix, x)
    # assert torch.allclose(y_triton, y_torch, atol=1e-1, rtol=0), "Triton and PyTorch results differ"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton SpMV Benchmark")
    parser.add_argument("--M", type=int, default=8192, help="Number of rows in the sparse matrix")
    parser.add_argument("--N", type=int, default=8192, help="Number of columns in the sparse matrix")
    parser.add_argument("--density", type=float, default=0.01, help="Density of the sparse matrix")
    args = parser.parse_args()
    
    main(args.M, args.N, args.density)