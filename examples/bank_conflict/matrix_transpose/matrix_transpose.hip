#include "dh_comms_dev.h"
#include <hip/hip_runtime.h>


#define TILE_DIM 16

__global__ void matrixTransposeShared(float* out,
                                      const float* in,
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

void runTranspose(int width, int height) {
  std::vector<float> h_in(width * height);
  std::vector<float> h_out(width * height);

  for (int i = 0; i < width * height; i++) {
    h_in[i] = static_cast<float>(i);
  }

  float* d_in;
  float* d_out;
  hipMalloc(&d_in, width * height * sizeof(float));
  hipMalloc(&d_out, width * height * sizeof(float));
  hipMemcpy(d_in, h_in.data(), width * height * sizeof(float), hipMemcpyHostToDevice);
  dim3 blockSize(TILE_DIM, TILE_DIM);
  dim3 gridSize((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);

  matrixTransposeShared<<<gridSize, blockSize>>>(d_out, d_in, width, height);

  const auto status = hipDeviceSynchronize();
  if(status != hipSuccess){
    std::terminate();
  }


  hipMemcpy(h_out.data(), d_out, width * height * sizeof(float), hipMemcpyDeviceToHost);
  hipFree(d_in);
  hipFree(d_out);
}


int main() {
  runTranspose(1024, 1024);
  return 0;
}
