# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import torch

# Ensure we're using ROCm and the GPU
assert torch.version.hip is not None, "This script requires ROCm."
device = torch.device("cuda")

# Define matrix sizes
M, N, K = 128, 128, 128

# Initialize matrices A (MxK) and B (KxN)
A = torch.randn(M, K, device=device)
B = torch.randn(K, N, device=device)

# Perform GEMM: C = A @ B
C = A @ B

# Optional: verify on CPU
A_cpu = A.cpu()
B_cpu = B.cpu()
C_ref = A_cpu @ B_cpu
assert torch.allclose(C.cpu(), C_ref, atol=1e-5)

print("GEMM completed successfully on ROCm GPU.")
