#include "eval.cuh"
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
using namespace std;

__global__ void idx_kernel(int *ans, int r, int c, int size) {
  *ans = idx(r, c, size);
}

int idx_wrapper(int r, int c, int size) {
  int h_ans;
  int *d_ans;
  cudaMalloc(&d_ans, sizeof(int));
  idx_kernel<<<1, 1>>>(d_ans, r, c, size);
  cudaMemcpy(&h_ans, d_ans, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_ans);
  return h_ans;
}

char *to_device(char *h_arr, int size) {
  char *d_arr;
  cudaMalloc(&d_arr, size);
  cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);
  return d_arr;
}

char *to_host(char *d_arr, int size) {
  char *h_arr = new char[size];
  cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);
  return h_arr;
}

__global__ void scan_horizontal_kernel(int size, char *board, char *x_scratch, char *o_scratch, int dc){
  scan_horizontal(size, board, x_scratch, o_scratch, dc);
}

void scan_horizontal_wrapper(int size, char *board, char *x_scratch,
                             char *o_scratch, int dc) {
  size_t board_mem_size = size * size * sizeof(char);
  char *d_board = to_device(board, board_mem_size);
  char *d_x_scratch = to_device(x_scratch, board_mem_size);
  char *d_o_scratch = to_device(o_scratch, board_mem_size);
  scan_horizontal_kernel<<<1, 1>>>(size, d_board, d_x_scratch, d_o_scratch, dc);
  cudaMemcpy(board, d_board, board_mem_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(x_scratch, d_x_scratch, board_mem_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(o_scratch, d_o_scratch, board_mem_size, cudaMemcpyDeviceToHost);
}

void test_scan_horizontal_helper(string name, int size, char *board, char *x_scratch, char *o_scratch,
                                 char *expected_board, char *expected_x_scratch,
                                 char *expected_o_scratch, int dc) {
  cout << "Running test: " << name << endl;
  scan_horizontal_wrapper(size, board, x_scratch, o_scratch, dc);
  for (int i = 0; i < size * size; i++) {
    cout << "Got: " << board[i] << (int)x_scratch[i] << (int)o_scratch[i] << endl;
    cout << "Expect: " << expected_board[i] << (int)expected_x_scratch[i] << (int)expected_o_scratch[i] << endl;
    assert(board[i] == expected_board[i]);
    assert(x_scratch[i] == expected_x_scratch[i]);
    assert(o_scratch[i] == expected_o_scratch[i]);
  }
}

void test_idx() {
  assert(idx_wrapper(1, 4, 5) == 9);
  assert(idx_wrapper(0, 0, 5) == 0);
  assert(idx_wrapper(0, 7, 12) == 7);
  assert(idx_wrapper(8, 3, 10) == 83);
}


void test_scan_horizontal() {
  char board[9] = {'x', 'x', '.',
                   '.', 'o', '.',
                   'x', '.', 'o'};
  char x_scratch[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  char o_scratch[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  char expected_board[9] = {'x', 'x', '.',
                            '.', 'o', '.',
                            'x', '.', 'o'};
  char expected_x_scratch[9] = {0, 0, 2,
                                0, 0, 0,
                                0, 1, 0};
  char expected_o_scratch[9] = {0, 0, 0,
                                0, 0, 1,
                                0, 0, 0};
  test_scan_horizontal_helper("scan right", 3, board, x_scratch, o_scratch,
    expected_board, expected_x_scratch, expected_o_scratch, 1);

  char e_xs[9] = {0, 0, 2, 0, 0, 0, 0, 1, 0};
  char e_os[9] = {0, 0, 0, 1, 0, 1, 0, 1, 0};
  // scan same board with previous scratch but from left
  test_scan_horizontal_helper("scan left", 3, board, x_scratch, o_scratch,
    expected_board, e_xs, e_os, -1);

}

int main() {
  test_idx();
  test_scan_horizontal();
  return 0;
}
