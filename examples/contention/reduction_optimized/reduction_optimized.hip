// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <cstdint>
#include <iostream>

#define hip_try(expr)                                             \
  do {                                                            \
    hipError_t err = (expr);                                      \
    if (err != hipSuccess) {                                      \
      const char* msg = hipGetErrorString(err);                   \
      throw std::runtime_error(std::string("HIP error: ") + msg); \
    }                                                             \
  } while (0)


__global__ void reduction_kernel(const float* input, float* result, std::size_t count) {
  extern __shared__ float shared_data[];
  const auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  const auto local_id = threadIdx.x;

  float sum = 0.0;
  if (thread_id < count) {
    sum = input[thread_id] / (thread_id + 1);
  }
  shared_data[local_id] = sum;
  __syncthreads();

  for (std::size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (local_id < stride) {
      shared_data[local_id] += shared_data[local_id + stride];
    }
    __syncthreads();
  }

  if (local_id == 0) {
    atomicAdd(result, shared_data[0]);
  }
}

int main() {
  const std::size_t count{10000000};
  thrust::device_vector<float> input(count, 1);
  thrust::device_vector<float> result(1, 0);
  thrust::sequence(input.begin(), input.end(), 1);

  const std::size_t block_size{512};
  const std::size_t num_blocks{(count + block_size - 1) / block_size};

  std::cout << "input: " << input.data().get() << std::endl;
  std::cout << "result: " << result.data().get() << std::endl;

  reduction_kernel<<<num_blocks, block_size, block_size * sizeof(float)>>>(
      input.data().get(), result.data().get(), count);

  const auto status = hipDeviceSynchronize();
  if (status != hipSuccess || result[0] != count) {
    std::cout << "Kernel failed.";
    return -1;
  } else {
    std::cout << "Success!";
  }
  std::cout << std::endl;
  return static_cast<int>(result[0]);
}