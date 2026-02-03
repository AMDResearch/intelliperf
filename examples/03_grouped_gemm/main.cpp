#include <hip/hip_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>

#include "kernel.hip"

int main() {
    const int num_groups = 8;
    const int K = 1024;
    const int N = 512;
    const int max_m = 2048;

    // Create offsets for groups
    std::vector<int> h_offsets(num_groups + 1);
    h_offsets[0] = 0;
    for (int i = 1; i <= num_groups; i++) {
        h_offsets[i] = h_offsets[i-1] + (max_m / num_groups);
    }
    int total_m = h_offsets[num_groups];

    // Allocate host memory
    std::vector<float> h_X(total_m * K, 1.0f);
    std::vector<float> h_weights(num_groups * K * N, 1.0f);
    std::vector<float> h_Y(total_m * N, 0.0f);

    // Allocate device memory
    float *d_X, *d_weights, *d_Y;
    int *d_offsets;
    hipMalloc(&d_X, total_m * K * sizeof(float));
    hipMalloc(&d_weights, num_groups * K * N * sizeof(float));
    hipMalloc(&d_Y, total_m * N * sizeof(float));
    hipMalloc(&d_offsets, (num_groups + 1) * sizeof(int));

    // Copy to device
    hipMemcpy(d_X, h_X.data(), total_m * K * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_weights, h_weights.data(), num_groups * K * N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_offsets, h_offsets.data(), (num_groups + 1) * sizeof(int), hipMemcpyHostToDevice);

    // Warmup
    dim3 block(256);
    dim3 grid((total_m + 31) / 32, (N + 31) / 32);

    for (int i = 0; i < 5; i++) {
        hipLaunchKernelGGL(grouped_gemm_kernel, grid, block, 0, 0,
                          d_Y, d_X, d_weights, d_offsets, num_groups, K, N);
    }
    hipDeviceSynchronize();

    // Benchmark
    const int num_runs = 10;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_runs; i++) {
        hipLaunchKernelGGL(grouped_gemm_kernel, grid, block, 0, 0,
                          d_Y, d_X, d_weights, d_offsets, num_groups, K, N);
    }
    hipDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count() / num_runs;

    // Copy result back
    hipMemcpy(h_Y.data(), d_Y, total_m * N * sizeof(float), hipMemcpyDeviceToHost);

    // Print result
    std::cout << "Grouped GEMM Benchmark" << std::endl;
    std::cout << "Time: " << elapsed << " ms" << std::endl;
    std::cout << "Output[0]: " << h_Y[0] << std::endl;

    // Cleanup
    hipFree(d_X);
    hipFree(d_weights);
    hipFree(d_Y);
    hipFree(d_offsets);

    return 0;
}

