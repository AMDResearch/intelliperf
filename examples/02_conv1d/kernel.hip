// Optimized causal 1D convolution kernel with TILING
#include <hip/hip_runtime.h>

// ============================================================
// TUNABLE PARAMETERS
// ============================================================
#define TILE_K 256
#define BLOCK_SIZE 256
#define BATCH_SIZE 16
#define MAX_WIDTH 4

// ============================================================
// KERNEL IMPLEMENTATION
// ============================================================
__global__ void causal_conv1d_kernel(
    float* __restrict__ out,
    const float* __restrict__ x,
    const float* __restrict__ weight,
    int batch,
    int dim,
    int seqlen,
    int width
) {
    int tid = threadIdx.x;
    int batch_id = blockIdx.x % batch;
    int channel_id = blockIdx.x / batch;

    if (channel_id >= dim) return;

    // Shared memory for tile
    __shared__ float tile[TILE_K + MAX_WIDTH]; // +MAX_WIDTH to handle boundary conditions

    // Use TILE_K for tiled processing
    for (int tile_start = (blockIdx.y * TILE_K) + tid; tile_start < seqlen; tile_start += blockDim.x * gridDim.y) {
        int tile_end = min(tile_start + TILE_K, seqlen);

        // Load data into shared memory
        for (int i = tile_start; i < tile_end; i += blockDim.x) {
            if (i < seqlen) {
                tile[i - tile_start] = x[batch_id * dim * seqlen + channel_id * seqlen + i];
            }
        }
        __syncthreads();

        for (int i = tile_start; i < tile_end; i += blockDim.x) {
            float sum = 0.0f;
            for (int w = 0; w < width; w++) {
                int idx = i - w;
                if (idx >= tile_start && idx < tile_end) {
                    sum += tile[idx - tile_start] * weight[w];
                } else if (idx >= 0) {
                    sum += x[batch_id * dim * seqlen + channel_id * seqlen + idx] * weight[w];
                }
            }
            if (i < seqlen) {
                out[batch_id * dim * seqlen + channel_id * seqlen + i] = sum;
            }
        }
        __syncthreads();
    }
}

// ============================================================
// LAUNCH WRAPPER - Configure grid/block here
// ============================================================
void conv1d(
    float* d_out,
    const float* d_x,
    const float* d_weight,
    int batch,
    int dim,
    int seqlen,
    int width,
    hipStream_t stream = 0
) {
    // Configure launch parameters
    dim3 grid(batch * dim);  // One block per (batch, channel)
    dim3 block(BLOCK_SIZE);   // Threads per block

    // Launch kernel
    hipLaunchKernelGGL(
        causal_conv1d_kernel,
        grid, block,
        0,      // shared memory (dynamically allocated)
        stream, // stream
        d_out, d_x, d_weight, batch, dim, seqlen, width
    );
}
