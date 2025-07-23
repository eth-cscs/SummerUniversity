// https://people.maths.ox.ac.uk/~gilesm/cuda/lecs/lec4.pdf

// https://pages.mini.pw.edu.pl/~kaczmarskik/gpca/resources/Part4-new-cuda-features.pdf

/*

TODO: 1) Eliminate race condition
      2) Add reduction between blocks

*/

#include <iostream>

#define N 20000

__global__ void sum(float *d_data, float *d_sum) {
  extern __shared__  float temp[];
  int tid = threadIdx.x, tidw;
  float value = d_data[tid + blockIdx.x * blockDim.x];
  for (int i = 1; i < warpSize; i <<= 1)
    value += __shfl_xor_sync((unsigned)-1, value, i);
  tidw = tid / warpSize;
  if (!(tid % warpSize)) {
    temp[tidw] = value;
  }
  for (int d = blockDim.x / warpSize / 2; d > 0; d >>= 1) {
    if (!(tid % warpSize) && tidw < d) {
      temp[tidw] += temp[tidw + d];
    }
  }
}

int main() {
  float d_data_host[N], *d_data, d_sum_host, *d_sum;
  for (int i = 0; i < N; i++) {
    d_data_host[i] = i;
  }
  cudaMalloc(&d_data, N * sizeof(float));
  cudaMalloc(&d_sum, 1 * sizeof(float));
  cudaMemcpy(d_data, d_data_host, N * sizeof(float), cudaMemcpyHostToDevice);
  sum<<<(N - 1) / 128 + 1, 128, 128 * sizeof(float)>>>(d_data, d_sum);
  cudaMemcpy(&d_sum_host, d_sum, 1 * sizeof(float), cudaMemcpyDeviceToHost);
  std::cout << d_sum_host << std::endl;
}
