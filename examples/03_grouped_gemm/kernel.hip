// Grouped GEMM kernel with ACTUAL tiling
#define TILE_M 64
#define TILE_N 64
#define TILE_K 256

__global__ void grouped_gemm_kernel(
    float* __restrict__ Y,
    const float* __restrict__ X,
    const float* __restrict__ weights,
    const int* __restrict__ offsets,
    int num_groups,
    int K,
    int N
) {
    int group_id = blockIdx.x / ((N + TILE_N - 1) / TILE_N);
    if (group_id >= num_groups) return;

    int M = offsets[group_id + 1] - offsets[group_id];
    int local_row = blockIdx.y * blockDim.y + threadIdx.y;
    int local_col = (blockIdx.x % ((N + TILE_N - 1) / TILE_N)) * blockDim.x + threadIdx.x;
    
    if (local_row >= M || local_col >= N) return;

    int global_row = offsets[group_id] + local_row;
    
    float sum = 0.0f;
    
    // Tiled K-dimension using TILE_K
    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        int k_end = min(k_tile + TILE_K, K);
        
        for (int k = k_tile; k < k_end; k++) {
            float x_val = X[global_row * K + k];
            float w_val = weights[group_id * K * N + k * N + local_col];
            sum += x_val * w_val;
        }
    }
    
    Y[global_row * N + local_col] = sum;
}
