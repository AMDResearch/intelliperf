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


__global__ void vector_add(const float* a, const float* b, float* c, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

int main() {
    const size_t N = 1024 * 1024;
    const size_t size = N * sizeof(float);

    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_c = new float[N];

    // Initialize input vectors
    for (size_t i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = 2 * i;
    }

    float *d_a, *d_b, *d_c;
    hip_try(hipMalloc(&d_a, size));
    hip_try(hipMalloc(&d_b, size));
    hip_try(hipMalloc(&d_c, size));

    hip_try(hipMemcpy(d_a, h_a, size, hipMemcpyHostToDevice));
    hip_try(hipMemcpy(d_b, h_b, size, hipMemcpyHostToDevice));

    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Create HIP events
    hipEvent_t start, stop;
    hip_try(hipEventCreate(&start));
    hip_try(hipEventCreate(&stop));

    // Record start
    hip_try(hipEventRecord(start, 0));

    // Launch kernel
    hipLaunchKernelGGL(vector_add, dim3(blocks), dim3(threadsPerBlock), 0, 0, d_a, d_b, d_c, N);

    // Record stop
    hip_try(hipEventRecord(stop, 0));
    hip_try(hipEventSynchronize(stop));

    // Calculate elapsed time
    float milliseconds = 0;
    hip_try(hipEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    // Copy result back to host
    hip_try(hipMemcpy(h_c, d_c, size, hipMemcpyDeviceToHost));

    // Print some results
    for (size_t i = 0; i < 5; ++i)
        std::cout << i << ": " << h_a[i] << " + " << h_b[i] << " = " << h_c[i] << std::endl;

    // Cleanup
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    hip_try(hipFree(d_a));
    hip_try(hipFree(d_b));
    hip_try(hipFree(d_c));
    hip_try(hipEventDestroy(start));
    hip_try(hipEventDestroy(stop));

    return 0;
}
