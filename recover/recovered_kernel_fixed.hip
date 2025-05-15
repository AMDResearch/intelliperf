#include <hip/hip_runtime.h>
#include <thrust/device_vector.h>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>
#include "/maestro/src/formulas/code_gen/../../csrc/vmem_allocator.hpp"

#define TILE_DIM 32

__global__ void matrixTransposeShared(const float* in,
                                      float* out,
                                      int width,
                                      int height) {
  __shared__ float tile[TILE_DIM][TILE_DIM];
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  if (x < width && y < height) {
    tile[threadIdx.y][threadIdx.x] = in[y * width + x];
  }
  __syncthreads();
  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;
  if (x < height && y < width) {
    out[y * height + x] = tile[threadIdx.x][threadIdx.y];
  }
}

template <typename T>
std::vector<T> load_arg(const char* path, int index) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error(std::string("Failed to open file: ") + path);
  }

  constexpr std::size_t BEGIN_SIZE = 6;  // "BEGIN\n"
  constexpr std::size_t END_SIZE = 4;    // "END\n"
  char begin_buf[BEGIN_SIZE] = {};
  char end_buf[END_SIZE] = {};

  std::vector<T> results;
  int current_index = 0;

  while (file.read(begin_buf, BEGIN_SIZE)) {
    if (std::string(begin_buf, BEGIN_SIZE) != "BEGIN\n") {
      throw std::runtime_error("Expected BEGIN marker not found");
    }

    std::size_t ptr_size = 0;
    file.read(reinterpret_cast<char*>(&ptr_size), sizeof(std::size_t));
    if (!file)
      throw std::runtime_error("Failed to read ptr_size");

    if (ptr_size % sizeof(T) != 0) {
      throw std::runtime_error("ptr_size is not a multiple of sizeof(T)");
    }

    std::size_t num_elements = ptr_size / sizeof(T);
    std::vector<T> temp(num_elements);
    std::cout << "Reading " << ptr_size << " bytes" << std::endl;
    file.read(reinterpret_cast<char*>(temp.data()), ptr_size);

    std::cout << "First 10 values\n";
    for (std::size_t i = 0; i < 10; i++) {
      std::cout << i << " " << temp[i] << std::endl;
    }

    if (!file)
      throw std::runtime_error("Failed to read value block");

    file.read(end_buf, END_SIZE);
    if (!file) {
      throw std::runtime_error("Failed to read END marker");
    }

    std::string end_str(end_buf, END_SIZE);
    if (end_str != "END\n") {
      std::cerr << "Found unexpected END marker: ";
      std::cerr << end_str;
      std::cerr << std::endl;
      throw std::runtime_error("Expected END marker not found");
    }

    if (current_index == index) {
      return temp;
    }
    ++current_index;
  }

  if (index != -1 && index >= current_index) {
    throw std::out_of_range("Requested index exceeds number of values in file");
  }

  return results;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    printf("Usage: %s <path-to-args-to-load>\n", argv[0]);
    return 1;
  }

  auto h_arg_0 = load_arg<float>(argv[1], 0);
  auto d_arg_0 = thrust::device_vector<float>(h_arg_0);
  auto h_arg_1 = load_arg<float>(argv[1], 1);
  auto d_arg_1 = thrust::device_vector<float>(h_arg_1);
  auto h_arg_2 = load_arg<int>(argv[1], 2);
  int d_arg_2 = h_arg_2[0];
  auto h_arg_3 = load_arg<int>(argv[1], 3);
  int d_arg_3 = h_arg_3[0];

  matrixTransposeShared<<<8, 1024>>>(
      d_arg_0.data().get(), d_arg_1.data().get(), d_arg_2, d_arg_3);

  return hipDeviceSynchronize();
}