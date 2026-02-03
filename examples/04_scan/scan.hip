#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>

// Simple scan kernel - processes one element per thread
__global__ void scan_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float sum = 0.0f;
        // Each thread computes prefix sum up to its index (naive O(n^2) approach)
        for (int i = 0; i <= idx; i++) {
            sum += input[i];
        }
        output[idx] = sum;
    }
}

// CPU reference implementation
void cpu_scan(const float* input, float* output, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += input[i];
        output[i] = sum;
    }
}

// Verification
bool verify(const float* gpu_output, const float* cpu_output, int n, float tolerance = 1e-3f) {
    for (int i = 0; i < n; i++) {
        if (std::abs(gpu_output[i] - cpu_output[i]) > tolerance) {
            std::cout << "Mismatch at index " << i << ": GPU=" << gpu_output[i]
                      << " CPU=" << cpu_output[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    // Problem size
    const int N = 1 << 20;  // 1M elements
    const int SIZE = N * sizeof(float);
    const int ITERATIONS = 1;

    std::cout << "Scan Benchmark" << std::endl;
    std::cout << "Elements: " << N << std::endl;
    std::cout << "Iterations: " << ITERATIONS << std::endl;

    // Allocate host memory
    std::vector<float> h_input(N);
    std::vector<float> h_output(N);
    std::vector<float> h_reference(N);

    // Initialize input
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;  // Simple case: all ones
    }

    // Allocate device memory
    float *d_input, *d_output;
    hipMalloc(&d_input, SIZE);
    hipMalloc(&d_output, SIZE);

    // Copy to device
    hipMemcpy(d_input, h_input.data(), SIZE, hipMemcpyHostToDevice);

    // Kernel configuration
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Warmup
    hipLaunchKernelGGL(scan_kernel, dim3(gridSize), dim3(blockSize), 0, 0,
                       d_input, d_output, N);
    hipDeviceSynchronize();

    // Timing
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        hipLaunchKernelGGL(scan_kernel, dim3(gridSize), dim3(blockSize), 0, 0,
                           d_input, d_output, N);
    }
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float milliseconds = 0;
    hipEventElapsedTime(&milliseconds, start, stop);
    float avg_time = milliseconds / ITERATIONS;

    std::cout << "Average kernel time: " << avg_time << " ms" << std::endl;

    // Copy result back
    hipMemcpy(h_output.data(), d_output, SIZE, hipMemcpyDeviceToHost);

    // Verify
    cpu_scan(h_input.data(), h_reference.data(), N);
    bool correct = verify(h_output.data(), h_reference.data(), N);

    if (correct) {
        std::cout << "✓ Results verified!" << std::endl;
    } else {
        std::cout << "✗ Verification failed!" << std::endl;
    }

    // Cleanup
    hipFree(d_input);
    hipFree(d_output);
    hipEventDestroy(start);
    hipEventDestroy(stop);

    return correct ? 0 : 1;
}
