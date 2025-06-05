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
#include <vector>

#define TILE_DIM 16

__global__ void matrixTransposeShared_0(float* out,
                                        const float* in,
                                        int width,
                                        int height) {
  __shared__ float tile[TILE_DIM][TILE_DIM];

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

__global__ void matrixTransposeShared_1(float* out,
                                        const float* in,
                                        int width,
                                        int height) {
  __shared__ float tile[TILE_DIM][TILE_DIM];

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

__global__ void matrixTransposeShared_2(float* out,
                                        const float* in,
                                        int width,
                                        int height) {
  __shared__ float tile[TILE_DIM][TILE_DIM];

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

void runTranspose(int width, int height) {
  std::vector<float> h_in(width * height);
  std::vector<float> h_out(width * height);

  for (int i = 0; i < width * height; i++) {
    h_in[i] = static_cast<float>(i);
  }

  float* d_in;
  float* d_out;
  hipMalloc(&d_in, width * height * sizeof(float));
  hipMalloc(&d_out, width * height * sizeof(float));
  hipMemcpy(d_in, h_in.data(), width * height * sizeof(float), hipMemcpyHostToDevice);
  dim3 blockSize(TILE_DIM, TILE_DIM);
  dim3 gridSize((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);

  matrixTransposeShared_0<<<gridSize, blockSize>>>(d_out, d_in, width, height);
  matrixTransposeShared_1<<<gridSize, blockSize>>>(d_out, d_in, width, height);
  matrixTransposeShared_2<<<gridSize, blockSize>>>(d_out, d_in, width, height);

  const auto status = hipDeviceSynchronize();
  if (status != hipSuccess) {
    std::terminate();
  }
  hipMemcpy(h_out.data(), d_out, width * height * sizeof(float), hipMemcpyDeviceToHost);

  hipFree(d_in);
  hipFree(d_out);
}

int main() {
  runTranspose(1024, 1024);
  return 0;
}
