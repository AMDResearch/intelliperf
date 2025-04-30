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
    hipMalloc(&d_a, size);
    hipMalloc(&d_b, size);
    hipMalloc(&d_c, size);

    hipMemcpy(d_a, h_a, size, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, size, hipMemcpyHostToDevice);

    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Create HIP events
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    // Record start
    hipEventRecord(start, 0);

    // Launch kernel
    hipLaunchKernelGGL(vector_add, dim3(blocks), dim3(threadsPerBlock), 0, 0, d_a, d_b, d_c, N);

    // Record stop
    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    hipEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    // Copy result back to host
    hipMemcpy(h_c, d_c, size, hipMemcpyDeviceToHost);

    // Print some results
    for (size_t i = 0; i < 5; ++i)
        std::cout << i << ": " << h_a[i] << " + " << h_b[i] << " = " << h_c[i] << std::endl;

    // Cleanup
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
    hipEventDestroy(start);
    hipEventDestroy(stop);

    return 0;
}
