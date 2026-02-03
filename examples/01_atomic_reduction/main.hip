#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

extern void atomic_reduce(const float* d_input, float* d_output, int n, hipStream_t stream);

#define HIP_CHECK(cmd) \
    do { \
        hipError_t error = (cmd); \
        if (error != hipSuccess) { \
            std::cerr << "HIP error: " << hipGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

int main() {
    const int N = 1 << 20; // 1M elements
    const int WARMUP = 5;
    const int ITERATIONS = 100;

    // Allocate host memory
    std::vector<float> h_input(N);
    float h_output = 0.0f;

    // Initialize input
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;
    }
    float expected = static_cast<float>(N);

    // Allocate device memory
    float *d_input, *d_output;
    HIP_CHECK(hipMalloc(&d_input, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_output, sizeof(float)));

    // Copy to device
    HIP_CHECK(hipMemcpy(d_input, h_input.data(), N * sizeof(float), hipMemcpyHostToDevice));

    // Create stream
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // Warmup
    for (int i = 0; i < WARMUP; i++) {
        HIP_CHECK(hipMemset(d_output, 0, sizeof(float)));
        atomic_reduce(d_input, d_output, N, stream);
        HIP_CHECK(hipStreamSynchronize(stream));
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < ITERATIONS; i++) {
        HIP_CHECK(hipMemset(d_output, 0, sizeof(float)));
        atomic_reduce(d_input, d_output, N, stream);
    }

    HIP_CHECK(hipStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();

    // Copy result back
    HIP_CHECK(hipMemcpy(&h_output, d_output, sizeof(float), hipMemcpyDeviceToHost));

    // Calculate timing
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_time_ms = elapsed_ms / ITERATIONS;

    // Verify result
    float error = std::abs(h_output - expected) / expected;
    bool correct = error < 1e-5;

    std::cout << "Atomic Reduction Benchmark" << std::endl;
    std::cout << "Elements: " << N << std::endl;
    std::cout << "Average time: " << avg_time_ms << " ms" << std::endl;
    std::cout << "Result: " << h_output << " (expected: " << expected << ")" << std::endl;
    std::cout << "Correctness: " << (correct ? "PASS" : "FAIL") << std::endl;

    // Cleanup
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipStreamDestroy(stream));

    return correct ? 0 : 1;
}

