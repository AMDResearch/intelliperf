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

#include <cstdio>
#include <hip/hip_runtime.h>

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
    hipMalloc((void **)&d_in, size);
    hipMalloc((void **)&d_out, sizeof(float) * (numElements / BLOCK_SIZE));

    hipMemcpy(d_in, h_in, size, hipMemcpyHostToDevice);

    int gridSize = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    reduce_kernel<<<gridSize, BLOCK_SIZE>>>(d_in, d_out, numElements);
    hipDeviceSynchronize();

    hipMemcpy(h_out, d_out, sizeof(float) * gridSize, hipMemcpyDeviceToHost);

    printf("First block sum: %f\n", h_out[0]);

    // Free resources.
    hipFree(d_in);
    hipFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}
