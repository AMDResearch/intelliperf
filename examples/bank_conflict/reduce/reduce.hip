// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include <cstdio>
#include <hip/hip_runtime.h>

#define hip_try(expr)                                             \
  do {                                                            \
    hipError_t err = (expr);                                      \
    if (err != hipSuccess) {                                      \
      const char* msg = hipGetErrorString(err);                   \
      throw std::runtime_error(std::string("HIP error: ") + msg); \
    }                                                             \
  } while (0)


#define BLOCK_SIZE 256

__global__ void reduce_kernel(const float *d_in, float *d_out, int n) {
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (idx < n) ? d_in[idx] : 0.0f;
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) {
        int index = tid * 2 * s;
        if (index < blockDim.x && (index + s) < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_out[blockIdx.x] = sdata[0];
    }
}

int main() {
    const int numElements = 1 << 20;
    const size_t size = numElements * sizeof(float);

    float *h_in = (float *)malloc(size);
    float *h_out = (float *)malloc(sizeof(float) * (numElements / BLOCK_SIZE));

    for (int i = 0; i < numElements; i++) {
        h_in[i] = 1.0f;
    }

    float *d_in = nullptr, *d_out = nullptr;
    hip_try(hipMalloc((void **)&d_in, size));
    hip_try(hipMalloc((void **)&d_out, sizeof(float) * (numElements / BLOCK_SIZE)));

    hip_try(hipMemcpy(d_in, h_in, size, hipMemcpyHostToDevice));

    int gridSize = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    reduce_kernel<<<gridSize, BLOCK_SIZE>>>(d_in, d_out, numElements);
    hip_try(hipDeviceSynchronize());

    hip_try(hipMemcpy(h_out, d_out, sizeof(float) * gridSize, hipMemcpyDeviceToHost));

    printf("First block sum: %f\n", h_out[0]);

    // Free resources.
    hip_try(hipFree(d_in));
    hip_try(hipFree(d_out));
    free(h_in);
    free(h_out);

    return 0;
}
