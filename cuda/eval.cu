__inline__ int idx(int r, int c, int size) { return r * size + c; }

__inline__ void update_scratch(int state, int i, char *x_scratch, char *o_scratch) {
  if (state == 0) return;
  char *scratch = state > 0 ? x_scratch : o_scratch;
  state = abs(state);
  scratch[i] = max(scratch[i], state);
}

__inline__ int update_state(int r, int c, int state, char *board, char *x_scratch, char *o_scratch) {
  int i = idx(r, c, size); 
  if (state[i] == '.') {
    update_scratch(state, i, x_scratch, o_scratch);
  } else {
    bool new_x = state[i] == 'x';
    bool old_x = state >= 1;
    int dir = new_x ? 1 : -1;
    if (new_x != old_x) {
      state = dir;
    } else {
      state += dir; 
    }
  }
}

// Clearly not going to make functions for each direction, writing these to help myself think
__inline__ void scan_right(int size, char *board, char *x_scratch, char *o_scratch) {
  int state = 0; 
  // 0 = empty, +n = n x's in a row, -n = n o's in a row
  int r0 = 0, c0 = 0;
  for (int r = r0; r < size; r++) {
    for (int c = c0; c < size; c++) {
      state = update_state(r, c, state, board, x_scratch, o_scratch);
    }
    state = 0;
  }
}

__inline__ void scan_left(int size, char *board, char *x_scratch, char *o_scratch) {
  int state = 0;
  int r0 = 0, c0 = 0;
  for (int c=c0; c < size; c++){
    for (int r=r0; r < size; r++) {
      state = update_state(r, c, state, board, x_scratch, o_scratch);
    }
    state = 0;
  }
}

__global__
int eval(int size, char *board, char *x_scratch, char *o_scratch) {

}


