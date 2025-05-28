/****************************************************************************
MIT License

Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
****************************************************************************/

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

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

  hipMalloc(&d_input, count * sizeof(data_t));
  hipMalloc(&d_result, sizeof(data_t));

  std::vector<data_t> h_input(count, 1);
  hipMemcpy(d_input, h_input.data(), count * sizeof(data_t), hipMemcpyHostToDevice);
  hipMemset(d_result, 0, sizeof(data_t));

  std::cout << "input: " << d_input << std::endl;
  std::cout << "result: " << d_result << std::endl;

  reduction_kernel<<<num_blocks, block_size>>>(d_input, d_result, count);
  hipDeviceSynchronize();

  data_t h_result = 0;
  hipMemcpy(&h_result, d_result, sizeof(data_t), hipMemcpyDeviceToHost);

  if (h_result != count) {
    std::cout << "Kernel failed. Expected: " << count << ", Got: " << h_result << "\n";
    hipFree(d_input);
    hipFree(d_result);
    return -1;
  } else {
    std::cout << "Success!";
  }
  std::cout << std::endl;

  hipFree(d_input);
  hipFree(d_result);
  return 0;
}
