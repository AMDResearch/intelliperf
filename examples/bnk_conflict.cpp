#include <hip/hip_runtime.h>  
#include <iostream>  
  
#define BLOCK_SIZE 256  
#define LDS_SIZE 256  
  
__global__ void bankConflictKernel(float* d_out) {  
    // Declare shared memory (LDS)  
    __shared__ float lds[LDS_SIZE];  
  
    // Calculate thread ID  
    int tid = threadIdx.x;  
  
    // Initialize LDS with some values  
    lds[tid] = static_cast<float>(tid);  
  
    // Synchronize to ensure all threads have written to LDS  
    __syncthreads();  
  
    // Intentionally cause bank conflicts by accessing LDS with a stride  
    // that maps multiple threads to the same bank.  
    // For example, if LDS has 32 banks, accessing with a stride of 32  
    // will cause all threads to access the same bank.  
    int index = (tid * 32) % LDS_SIZE; // Stride of 32 causes bank conflicts  
  
    // Perform some operation to highlight the conflict  
    float value = lds[index] * 2.0f;  
  
    // Write the result to global memory  
    d_out[tid] = value;  
}  
  
int main() {  
    // Allocate memory on the host  
    float h_out[BLOCK_SIZE];  
  
    // Allocate memory on the device  
    float* d_out;  
    hipMalloc(&d_out, BLOCK_SIZE * sizeof(float));  
  
    // Launch the kernel  
    dim3 blockSize(BLOCK_SIZE);  
    dim3 gridSize(1);  
    hipLaunchKernelGGL(bankConflictKernel, gridSize, blockSize, 0, 0, d_out);  
  
    // Copy the result back to the host  
    hipMemcpy(h_out, d_out, BLOCK_SIZE * sizeof(float), hipMemcpyDeviceToHost);  
  
    // Print the results  
    for (int i = 0; i < BLOCK_SIZE; ++i) {  
        std::cout << "h_out[" << i << "] = " << h_out[i] << std::endl;  
    }  
  
    // Free device memory  
    hipFree(d_out);  
  
    return 0;  
}