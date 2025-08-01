import torch
import triton
import numpy as np
from scipy.sparse import csr_matrix
import argparse

from spmv import spmv_kernel


def spmv(x, data, indices, indptr, sparse_matrix, validate=False):
    M = indptr.size(0) - 1
    y = torch.empty(M, device=x.device, dtype=x.dtype)

    grid = (M,)

    spmv_kernel[grid](
        y, x,
        data, indices, indptr,
        M,
        BLOCK_SIZE=128
    )

    if validate:
        dense_matrix = torch.from_numpy(sparse_matrix.toarray()).to('cuda').to(torch.float16)
        y_torch = torch.mv(dense_matrix, x)
        if torch.allclose(y, y_torch, atol=1e-1, rtol=0):
            print("Validation Successful!")
        else:
            print("Validation Failed!")
            print(f"max diff: {(y - y_torch).abs().max().item()}")

    return y


def main(M=8192, N=8192, density=0.01, validate=False):
    sparse_matrix = csr_matrix(np.random.randn(M, N) * (np.random.rand(M, N) < density))

    data = torch.from_numpy(sparse_matrix.data).to('cuda').to(torch.float16)
    indices = torch.from_numpy(sparse_matrix.indices).to('cuda').to(torch.int32)
    indptr = torch.from_numpy(sparse_matrix.indptr).to('cuda').to(torch.int32)

    x = torch.randn(N, device='cuda', dtype=torch.float16)

    rep = 100

    for _ in range(10):
        spmv(x, data, indices, indptr, sparse_matrix)

    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    for _ in range(rep):
        spmv(x, data, indices, indptr, sparse_matrix)
    end_time.record()
    torch.cuda.synchronize()

    triton_time = start_time.elapsed_time(end_time) / rep
    print(f"Triton SpMV time: {triton_time:.4f} ms")

    if validate:
        spmv(x, data, indices, indptr, sparse_matrix, validate=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton SpMV Benchmark")
    parser.add_argument("--M", type=int, default=8192, help="Number of rows in the sparse matrix")
    parser.add_argument("--N", type=int, default=8192, help="Number of columns in the sparse matrix")
    parser.add_argument("--density", type=float, default=0.01, help="Density of the sparse matrix")
    parser.add_argument("--validate", action="store_true", help="Validate the Triton implementation against PyTorch.")
    args = parser.parse_args()

    main(args.M, args.N, args.density, validate=args.validate) 