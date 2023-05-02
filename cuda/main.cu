#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "eval.cuh"
#include "engine_board.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void print_gpu() {
  int nDevices = 0;
  cudaGetDeviceCount(&nDevices);
  printf("Number of GPUs: %d\n", nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
    prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
    prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
    2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  }
}

__global__ void hello_world() {
   printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main(int argc, char** argv )
{
   print_gpu();
   cudaDeviceSynchronize();

  string easy_win = R"(
x o . . .
x o . . .
x o . . .
x o . . .
. . . . .)";
  Engine_Board board(easy_win, 'x', 5);

   int move = board.cuda_recommendation(1);
   cout << "move: " << move << endl;
   return 0;
}
