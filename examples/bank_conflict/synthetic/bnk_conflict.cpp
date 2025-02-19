#include <hip/hip_runtime.h>
#include <iostream>  
#include "dh_comms_dev.h"
  
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