#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <random>
#include <vector>

using index_type = std::size_t;
using real_type = double;

namespace kernels {
// Histogram kernel courtesy of Kasia Swirydowicz
__global__ void histogram_kernel(const index_type n,
                                 const index_type k,
                                 const index_type* labels,
                                 const index_type* flip,
                                 const real_type* input,
                                 real_type* output) {
  index_type idx = blockIdx.x * blockDim.x + threadIdx.x;
  while (idx < n) {
    real_type val = input[idx];
    if (flip[idx] != 1) {
      val *= -1.0;
    }
    atomicAdd(&output[labels[idx]], val);
    idx += blockDim.x * gridDim.x;
  }
}
}  // namespace kernels
void histogram(index_type n,
               index_type k,
               index_type* labels,
               index_type* flip,
               real_type* input,
               real_type* output) {
  hipLaunchKernelGGL(kernels::histogram_kernel,
                     dim3(10000),
                     dim3(1024),
                     0,
                     0,
                     n,
                     k,
                     labels,
                     flip,
                     input,
                     output);
}

int main(int argc, char* argv[]) {
  index_type n = 238072;
  index_type k = 1858;

  std::vector<index_type> h_labels(n);
  std::vector<index_type> h_flip(n);
  std::vector<real_type> h_input(n);
  std::vector<real_type> h_output(k);

  // initial with random values
  // Set the seed for the random number generator
  std::srand(12345);

  for (index_type i = 0; i < n; i++) {
    h_labels[i] = std::rand() % k;
    h_flip[i] = std::rand() % 2;
    h_input[i] = std::rand() % 100;
  }

  for (index_type i = 0; i < k; i++) {
    h_output[i] = 0.0;
  }

  // allocate device memory
  index_type* d_labels;
  index_type* d_flip;
  real_type* d_input;
  real_type* d_output;

  hipMalloc(&d_labels, n * sizeof(index_type));
  hipMalloc(&d_flip, n * sizeof(index_type));
  hipMalloc(&d_input, n * sizeof(real_type));
  hipMalloc(&d_output, k * sizeof(real_type));

  hipMemcpy(d_labels, h_labels.data(), n * sizeof(index_type), hipMemcpyHostToDevice);
  hipMemcpy(d_flip, h_flip.data(), n * sizeof(index_type), hipMemcpyHostToDevice);
  hipMemcpy(d_input, h_input.data(), n * sizeof(real_type), hipMemcpyHostToDevice);
  hipMemcpy(d_output, h_output.data(), k * sizeof(real_type), hipMemcpyHostToDevice);

  histogram(n, k, d_labels, d_flip, d_input, d_output);

  hipDeviceSynchronize();
}