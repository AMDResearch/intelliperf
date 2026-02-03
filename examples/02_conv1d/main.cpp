// Host-side benchmark for conv1d kernel
#include <hip/hip_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>

#include "kernel.hip"

int main() {
    // Problem size
    const int batch = 1;
    const int dim = 2048;
    const int seqlen = 2048;
    const int width = 4;
    const size_t size = batch * dim * seqlen;

    // ============================================================
    // HOST MEMORY ALLOCATION & INITIALIZATION
    // ============================================================
    std::vector<float> h_x(size, 1.0f);
    std::vector<float> h_weight(width, 1.0f);
    std::vector<float> h_out(size, 0.0f);

    // ============================================================
    // DEVICE MEMORY ALLOCATION
    // ============================================================
    float *d_x, *d_weight, *d_out;
    hipMalloc(&d_x, size * sizeof(float));
    hipMalloc(&d_weight, width * sizeof(float));
    hipMalloc(&d_out, size * sizeof(float));

    // ============================================================
    // HOST TO DEVICE TRANSFER
    // ============================================================
    hipMemcpy(d_x, h_x.data(), size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_weight, h_weight.data(), width * sizeof(float), hipMemcpyHostToDevice);

    // ============================================================
    // WARMUP
    // ============================================================
    for (int i = 0; i < 5; i++) {
        conv1d(d_out, d_x, d_weight, batch, dim, seqlen, width);
    }
    hipDeviceSynchronize();

    // ============================================================
    // BENCHMARK
    // ============================================================
    const int num_runs = 10;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_runs; i++) {
        conv1d(d_out, d_x, d_weight, batch, dim, seqlen, width);
    }
    hipDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count() / num_runs;

    // ============================================================
    // DEVICE TO HOST TRANSFER
    // ============================================================
    hipMemcpy(h_out.data(), d_out, size * sizeof(float), hipMemcpyDeviceToHost);

    // ============================================================
    // RESULTS
    // ============================================================
    std::cout << "Conv1D Benchmark" << std::endl;
    std::cout << "Time: " << elapsed << " ms" << std::endl;
    std::cout << "Output[0]: " << h_out[0] << std::endl;

    // ============================================================
    // CLEANUP
    // ============================================================
    hipFree(d_x);
    hipFree(d_weight);
    hipFree(d_out);

    return 0;
}

