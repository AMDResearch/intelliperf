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
  
#define BLOCK_SIZE 256  
#define LDS_SIZE 256  
  
__global__ void bankConflictKernel(float* d_out) {  
    __shared__ float lds[LDS_SIZE];  
  
    int tid = threadIdx.x;  
    lds[tid] = static_cast<float>(tid);  
  
    // Synchronize to ensure all threads have written to LDS  
    __syncthreads();  
  
    // Intentionally cause bank conflicts by accessing LDS with a stride  
    // that maps multiple threads to the same bank.  
    int index = (tid * 32) % LDS_SIZE; 
  
     
    float value = lds[index] * 2.0f;   
    d_out[tid] = value;  
}  
  
int main() {  
    // Allocate memory on the host  
    float h_out[BLOCK_SIZE];  
  
    // Allocate memory on the device  
    float* d_out;  
    hipMalloc(&d_out, BLOCK_SIZE * sizeof(float));  
  
    dim3 blockSize(BLOCK_SIZE);  
    dim3 gridSize(1);  
    hipLaunchKernelGGL(bankConflictKernel, gridSize, blockSize, 0, 0, d_out);  
  
    // Copy the result back to the host  
    hipMemcpy(h_out, d_out, BLOCK_SIZE * sizeof(float), hipMemcpyDeviceToHost);  
  
    for (int i = 0; i < BLOCK_SIZE; ++i) {  
        std::cout << "h_out[" << i << "] = " << h_out[i] << std::endl;  
    }  
  
    hipFree(d_out);  
  
    return 0;  
}
