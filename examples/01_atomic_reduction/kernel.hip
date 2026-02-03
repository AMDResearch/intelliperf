#include <hip/hip_runtime.h>

#define BLOCK_SIZE 256
#define ELEMENTS_PER_THREAD 4

__global__ void atomic_reduce_kernel(const float* __restrict__ input, float* output, int n) {
    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + tid;

    // Shared memory for block-level reduction
    __shared__ float sdata[BLOCK_SIZE];

    float local_sum = 0.0f;

    // Each thread processes multiple elements
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int idx = global_idx * ELEMENTS_PER_THREAD + i;
        if (idx < n) {
            local_sum += input[idx];
        }
    }

    // Store in shared memory
    sdata[tid] = local_sum;
    __syncthreads();

    // Reduce within block
    for (int s = BLOCK_SIZE/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Only first thread in block does atomic add
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

void atomic_reduce(const float* d_input, float* d_output, int n, hipStream_t stream) {
    int threads = BLOCK_SIZE;
    int blocks = (n + threads * ELEMENTS_PER_THREAD - 1) / (threads * ELEMENTS_PER_THREAD);

    hipLaunchKernelGGL(atomic_reduce_kernel, dim3(blocks), dim3(threads), 0, stream,
                       d_input, d_output, n);
}

