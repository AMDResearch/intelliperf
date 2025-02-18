#include <hip/hip_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

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
  thrust::host_vector<float> h_in(width * height);
  thrust::host_vector<float> h_out(width * height);

  for (int i = 0; i < width * height; i++) {
    h_in[i] = static_cast<float>(i);
  }

  thrust::device_vector<float> d_in = h_in;
  thrust::device_vector<float> d_out(width * height);

  dim3 blockSize(TILE_DIM, TILE_DIM);
  dim3 gridSize((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);

  matrixTransposeShared<<<gridSize, blockSize>>>(
      d_out.data().get(), d_in.data().get(), width, height);

  const auto status = hipDeviceSynchronize();
  if(status != hipSuccess){
    std::terminate();
  }

  h_out = d_out;
}

int main() {
  runTranspose(1024, 1024);
  return 0;
}
