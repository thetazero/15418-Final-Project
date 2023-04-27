#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "board.h"

__global__ void test(){
  printf("Hi Cuda World\n");
  Board b(12);
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

int main(int argc, char** argv )
{
   test<<<1,1>>>();
   cudaDeviceSynchronize();
   print_gpu();
   return 0;
}