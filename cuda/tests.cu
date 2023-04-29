#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "eval.cuh"

__global__ void idx_kernel(int *ans, int r, int c, int size) {
  *ans = idx(r, c, size);
}

int idx_wrapper(int r, int c, int size){
  int h_ans;
  int *d_ans;
  cudaMalloc(&d_ans, sizeof(int));
  idx_kernel<<<1, 1>>>(d_ans, r, c, size);
  cudaMemcpy(&h_ans, d_ans, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_ans);
  return h_ans;
}

void test_idx() {
  assert(idx_wrapper(1, 4, 5) == 9);
  assert(idx_wrapper(0, 0, 5) == 0);
  assert(idx_wrapper(0, 7, 12) == 7);
  assert(idx_wrapper(8, 3, 10) == 83);
  printf("idx_tests sucessful\n");
}

int main() {
  test_idx();
  return 0;
}
