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
#include <iostream>
#include <numeric>
#include <vector>

#define TILE_DIM 16

template <typename T>
__global__ void matrixTransposeShared(T* out, const T* in, int width, int height) {
  __shared__ T tile[TILE_DIM][TILE_DIM];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;

  if (x < width && y < height) {
    tile[threadIdx.y][threadIdx.x] = in[y * width + x];
  }

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  if (x < height && y < width) {
    out[y * height + x] = tile[threadIdx.x][threadIdx.y];
  }
}

template <typename T>
__global__ void scaleMatrix(T* data, T scale, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] *= scale;
  }
}

template <typename T>
__global__ void vectorAddInPlace(T* data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] += data[idx];
  }
}

template <typename T>
void runKernels(int width, int height) {
  const int num_elements = width * height;
  std::vector<T> h_data(num_elements);
  std::vector<T> h_out(num_elements);

  // Fill with 0.0, 1.0, 2.0, ...
  std::iota(h_data.begin(), h_data.end(), static_cast<T>(0));

  T *d_data, *d_out;
  hipMalloc(&d_data, num_elements * sizeof(T));
  hipMalloc(&d_out, num_elements * sizeof(T));
  hipMemcpy(d_data, h_data.data(), num_elements * sizeof(T), hipMemcpyHostToDevice);

  // --- 1. Scale all elements
  int blockSize = 256;
  int gridSize = (num_elements + blockSize - 1) / blockSize;
  scaleMatrix<T><<<gridSize, blockSize>>>(d_data, static_cast<T>(2), num_elements);

  // --- 2. Add the buffer to itself
  vectorAddInPlace<T><<<gridSize, blockSize>>>(d_data, num_elements);

  // --- 3. Transpose into d_out
  dim3 blockDim(TILE_DIM, TILE_DIM);
  dim3 gridDim((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);
  matrixTransposeShared<T><<<gridDim, blockDim>>>(d_out, d_data, width, height);

  hipDeviceSynchronize();
  hipMemcpy(h_out.data(), d_out, num_elements * sizeof(T), hipMemcpyDeviceToHost);

  hipFree(d_data);
  hipFree(d_out);

  std::cout << "Result (first 10 elements):\n";
  for (int i = 0; i < 10; ++i)
    std::cout << h_out[i] << " ";
  std::cout << "\n";
}

int main() {
  runKernels<float>(1024, 1024);
  return 0;
}
