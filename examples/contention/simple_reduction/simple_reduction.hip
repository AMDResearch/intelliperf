// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#define hip_try(expr)                                             \
  do {                                                            \
    hipError_t err = (expr);                                      \
    if (err != hipSuccess) {                                      \
      const char* msg = hipGetErrorString(err);                   \
      throw std::runtime_error(std::string("HIP error: ") + msg); \
    }                                                             \
  } while (0)


using data_t = double;

__global__ void reduction_kernel(const data_t* input, data_t* result, std::size_t count) {
  const auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  if (thread_id < count) {
    unsafeAtomicAdd(result, input[thread_id]);
  }
}

int main() {
  const std::size_t count{10000000};
  const std::size_t block_size{512};
  const std::size_t num_blocks{(count + block_size - 1) / block_size};

  data_t* d_input = nullptr;
  data_t* d_result = nullptr;

  hip_try(hipMalloc(&d_input, count * sizeof(data_t)));
  hip_try(hipMalloc(&d_result, sizeof(data_t)));

  std::vector<data_t> h_input(count, 1);
  hip_try(hipMemcpy(d_input, h_input.data(), count * sizeof(data_t), hipMemcpyHostToDevice));
  hip_try(hipMemset(d_result, 0, sizeof(data_t)));

  std::cout << "input: " << d_input << std::endl;
  std::cout << "result: " << d_result << std::endl;

  reduction_kernel<<<num_blocks, block_size>>>(d_input, d_result, count);
  hip_try(hipDeviceSynchronize());

  data_t h_result = 0;
  hip_try(hipMemcpy(&h_result, d_result, sizeof(data_t), hipMemcpyDeviceToHost));

  if (h_result != count) {
    std::cout << "Kernel failed. Expected: " << count << ", Got: " << h_result << "\n";
    hip_try(hipFree(d_input));
    hip_try(hipFree(d_result));
    return -1;
  } else {
    std::cout << "Success!";
  }
  std::cout << std::endl;

  hip_try(hipFree(d_input));
  hip_try(hipFree(d_result));
  return 0;
}
