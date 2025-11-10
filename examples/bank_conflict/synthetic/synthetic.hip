// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>
#include <iostream>

#define hip_try(expr)                                             \
  do {                                                            \
    hipError_t err = (expr);                                      \
    if (err != hipSuccess) {                                      \
      const char* msg = hipGetErrorString(err);                   \
      throw std::runtime_error(std::string("HIP error: ") + msg); \
    }                                                             \
  } while (0)


#define BLOCK_SIZE 256
#define LDS_SIZE 256

__global__ void bankConflictKernel(float* d_out) {
    __shared__ float lds[LDS_SIZE];

    int tid = threadIdx.x;
    lds[tid] = static_cast<float>(tid);

    // Synchronize to ensure all threads have written to LDS
    __syncthreads();

    // Intentionally cause bank conflicts by accessing LDS with a stride
    // that maps multiple threads to the same bank.
    int index = (tid * 32) % LDS_SIZE;


    float value = lds[index] * 2.0f;
    d_out[tid] = value;
}

int main() {
    // Allocate memory on the host
    float h_out[BLOCK_SIZE];

    // Allocate memory on the device
    float* d_out;
    hip_try(hipMalloc(&d_out, BLOCK_SIZE * sizeof(float)));

    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize(1);
    hipLaunchKernelGGL(bankConflictKernel, gridSize, blockSize, 0, 0, d_out);

    // Copy the result back to the host
    hip_try(hipMemcpy(h_out, d_out, BLOCK_SIZE * sizeof(float), hipMemcpyDeviceToHost));

    for (int i = 0; i < BLOCK_SIZE; ++i) {
        std::cout << "h_out[" << i << "] = " << h_out[i] << std::endl;
    }

    hip_try(hipFree(d_out));

    return 0;
}
