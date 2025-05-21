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
#include <iostream>

#define N 1024  // Matrix size N x N

__global__ void column_stats(const float* __restrict__ data,
                             float* __restrict__ mean,
                             float* __restrict__ stddev,
                             int width) {
  int col = threadIdx.x + blockIdx.x * blockDim.x;

  if (col < width) {
    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (int row = 0; row < width; ++row) {
      float val = data[row * width + col];
      sum += val;
      sum_sq += val * val;
    }

    float mu = sum / width;
    float sigma = sqrtf(sum_sq / width - mu * mu);

    mean[col] = mu;
    stddev[col] = sigma;
  }
}

int main() {
  size_t size = N * N * sizeof(float);
  float* h_data = new float[N * N];
  float* h_mean = new float[N];
  float* h_stddev = new float[N];

  // Initialize matrix with test values
  for (int i = 0; i < N * N; ++i) {
    h_data[i] = static_cast<float>(i % 100);
  }

  // Device allocations
  float *d_data, *d_mean, *d_stddev;
  hipMalloc(&d_data, size);
  hipMalloc(&d_mean, N * sizeof(float));
  hipMalloc(&d_stddev, N * sizeof(float));

  // Copy input data to device
  hipMemcpy(d_data, h_data, size, hipMemcpyHostToDevice);

  // Configure kernel launch
  dim3 blockSize(256);
  dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

  // Launch kernel with <<< >>> syntax
  column_stats<<<gridSize, blockSize>>>(d_data, d_mean, d_stddev, N);

  // Copy results back
  hipMemcpy(h_mean, d_mean, N * sizeof(float), hipMemcpyDeviceToHost);
  hipMemcpy(h_stddev, d_stddev, N * sizeof(float), hipMemcpyDeviceToHost);

  // Output result for column 0
  std::cout << "Mean of column 0: " << h_mean[0] << std::endl;
  std::cout << "Stddev of column 0: " << h_stddev[0] << std::endl;

  // Cleanup
  delete[] h_data;
  delete[] h_mean;
  delete[] h_stddev;

  hipFree(d_data);
  hipFree(d_mean);
  hipFree(d_stddev);

  return 0;
}
