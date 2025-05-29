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
#include <cassert>
#include <iostream>
#include <vector>

#define hip_try(expr)                                                          \
  do {                                                                         \
    hipError_t status = (expr);                                                \
    if (status != hipSuccess) {                                                \
      throw std::runtime_error(std::string("HIP error at ") + __FILE__ + ":" + \
                               std::to_string(__LINE__) + ": " +               \
                               hipGetErrorString(status));                     \
    }                                                                          \
  } while (0)

// Uncoalesced matrix transpose kernel
__global__ void matrix_transpose(const float* __restrict__ in,
                                 float* __restrict__ out,
                                 int width,
                                 int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;  // column
  int y = blockIdx.y * blockDim.y + threadIdx.y;  // row

  if (x < width && y < height) {
    out[x * height + y] = in[y * width + x];
  }
}

int main() {
  const int width = 1024;
  const int height = 1024;
  const int size = width * height;
  const size_t bytes = size * sizeof(float);

  std::vector<float> h_in(size);
  std::vector<float> h_out(size, 0.0f);

  // Initialize input matrix row-major: A[y][x] = y * width + x
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      h_in[y * width + x] = static_cast<float>(y * width + x);

  float *d_in, *d_out;
  hip_try(hipMalloc(&d_in, bytes));
  hip_try(hipMalloc(&d_out, bytes));

  hip_try(hipMemcpy(d_in, h_in.data(), bytes, hipMemcpyHostToDevice));

  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  hipLaunchKernelGGL(matrix_transpose, grid, block, 0, 0, d_in, d_out, width, height);

  hip_try(hipMemcpy(h_out.data(), d_out, bytes, hipMemcpyDeviceToHost));

  // Validate result: transpose A[y][x] -> B[x][y]
  bool correct = true;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float expected = h_in[y * width + x];
      float result = h_out[x * height + y];
      if (expected != result) {
        std::cerr << "Mismatch at (" << x << ", " << y << "): " << result
                  << " != " << expected << "\n";
        correct = false;
        break;
      }
    }
  }

  std::cout << (correct ? "Transpose correct ✅" : "Transpose incorrect ❌") << "\n";

  hipFree(d_in);
  hipFree(d_out);
  return 0;
}
