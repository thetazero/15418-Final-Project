#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#ifndef CUDA_EVAL
#define CUDA_EVAL

// Get index of cell at row r and column c in a board of size size
__device__ __inline__
int idx(int r, int c, int size) { return r * size + c; }

// Given the current state, (where 0 = empty, +n = n x's in a row, -n = n o's in a row)
// and the index of a cell, if the cell is empty, update the scratch array
// The scratch arrays are used to keep track of the longest streak of x's and o's
// adjacent to each empty cell
__device__ __inline__
void update_scratch(int state, int i, char *x_scratch, char *o_scratch) {
  if (state == 0) return;
  char *scratch = state > 0 ? x_scratch : o_scratch;
  state = abs(state);
  scratch[i] = max(scratch[i], state);
}

/* Given the current state, (where 0 = empty, +n = n x's in a row, -n = n o's in a row)
 *  Process the cell at row r and column c in a board of size size
 * If the cell is empty, update the scratch array
 * Otherwise, update the state to keep track of the streak of x's or o's in the current direction
*/
__device__ __inline__
int update_state(int r, int c, int size, int state, char *board, char *x_scratch, char *o_scratch) {
  int i = idx(r, c, size);
  if (board[i] == '.') {
    update_scratch(state, i, x_scratch, o_scratch);
  } else {
    bool new_x = board[i] == 'x';
    bool old_x = state >= 1;
    int dir = new_x ? 1 : -1;
    if (new_x != old_x) {
      state = dir;
    } else {
      state += dir;
    }
  }
  return state;
}

/* Scan accross the board horizontaly (dc indicates direction.
 * Update the scratch arrays for each empty cell, to keep track of the longest streak of x's and o's
*/
__device__ __inline__
void scan_horizontal(int size, char *board, char *x_scratch, char *o_scratch, int dc) {
  int state = 0;
  // 0 = empty, +n = n x's in a row, -n = n o's in a row
  int r0 = 0;
  int c0 = dc > 0 ? 0 : size-1;
  int c_terminate = dc > 0 ? size : -1;
  for (int r = r0; r < size; r++) {
    for (int c = c0; c != c_terminate; c+=dc) {
      state = update_state(r, c, size, state, board, x_scratch, o_scratch);
    }
    state = 0;
  }
}

/* Scan accross the board vertically (dr indicates direction).
 * Update the scratch arrays for each empty cell, to keep track of the longest streak of x's and o's
*/
__device__ __inline__
void scan_vertical(int size, char *board, char *x_scratch, char *o_scratch, int dr) {
  int r0 = dr > 0 ? 0 : size-1;
  int c0 = 0;
  int r_terminate = dr > 0 ? size : -1;
  for (int c = c0; c < size; c++) {
    for (int r = r0; r != r_terminate; r+dr) {
      state = update_state(r, c, size, state, board, x_scratch, o_scratch);
    }
    state = 0;
  }
}

#endif
