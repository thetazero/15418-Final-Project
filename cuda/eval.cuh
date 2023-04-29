#ifndef EVAL
#define EVAL
__device__ __inline__ 
int idx(int r, int c, int size);

__device__ __inline__ 
void update_scratch(int state, int i, char *x_scratch, char *o_scratch);

__device__ __inline__
int update_state(int r, int c, int size, int state, char *board, char *x_scratch, char *o_scratch);

__device__ __inline__
void scan_horizontal(int size, char *board, char *x_scratch, char *o_scratch);

__global__
void eval(int size, char *board, char *x_scratch, char *o_scratch) {

}
#endif

