#define GAME_OVER_EVAL 1000
typedef int32 board_t; // test if 32 bit is faster than 8 bit as the ispc reference says

static inline int idx(int r, int c, int size) { return r * size + c; }

static inline uniform int sum_5(int r, int c, int dr, int dc, board_t *board, int size) {
  uniform int sum;
  int s;
  foreach (i = 0 ... 5) {
    s += board[idx(r + i * dr, r + i * dc, size)];
  }
  sum = reduce_add(s);
  return sum;
}

static int game_over(int r, int c, int r_max, int c_max, board_t *board,
                     int size) {
  int r_valid = (r_max - r >= 4);
  int c_valid = (c_max - c >= 4);
  int d1_valid = r_valid && c_valid;
  int d2_valid = r_valid && c >= 4;

  int rr = 0, cc = 0, d1 = 0, d2 = 0; // counts in each direction
  // int rrt = 0, cct = 0, d1t = 0, d2t = 0;
  // if (d1_valid) {
  //   d1 = sum_5(r, c, 1, 1, board, size);
  // }
  // if (d2_valid) {
  //   d2 = sum_5(r, c, 1, -1, board, size);;
  // }
  // if (r_valid) {
  //   rr = sum_5(r, c, 1, 0, board, size);
  // }
  // if (c_valid) {
  //   cc = sum_5(r, c, 0, 1, board, size);
  // }
  

  if (d1_valid) {
    d1 = (board[idx(r, c, size)] + board[idx(r + 1, c + 1, size)] +
          board[idx(r + 2, c + 2, size)] + board[idx(r + 3, c + 3, size)] +
          board[idx(r + 4, c + 4, size)]);
  }
  if (d2_valid) {
    d2 = (board[idx(r, c, size)] + board[idx(r + 1, c - 1, size)] +
          board[idx(r + 2, c - 2, size)] + board[idx(r + 3, c - 3, size)] +
          board[idx(r + 4, c - 4, size)]);
  }
  if (r_valid) {
    rr = (board[idx(r, c, size)] + board[idx(r + 1, c, size)] +
          board[idx(r + 2, c, size)] + board[idx(r + 3, c, size)] +
          board[idx(r + 4, c, size)]);
  }
  if (c_valid) {
    cc = (board[idx(r, c, size)] + board[idx(r, c + 1, size)] +
          board[idx(r, c + 2, size)] + board[idx(r, c + 3, size)] +
          board[idx(r, c + 4, size)]);
  }

  if (d1 == 5 || d2 == 5 || rr == 5 || cc == 5) {
    return GAME_OVER_EVAL;
  }

  if (d1 == -5 || d2 == -5 || rr == -5 || cc == -5) {
    return -1 * GAME_OVER_EVAL;
  }

  return 0;
}

static bool check_direction(int r, int c, int dr, int dc, int &count,
                            board_t *board, int size, int r_min, int r_max,
                            int c_min, int c_max) {
  int start_tile =
      board[idx(r + dr, c + dc, size)]; // the first tile in the sequence
  for (int i = 0; i < 4; i++) {
    // move to the next tile in direction
    r += dr;
    c += dc;
    // if out of bounds, not an open edge
    if (r > r_max || r < r_min || c > c_max || c < c_min) {
      return false;
    }

    int tile = board[idx(r, c, size)];

    if (tile == 0) {
      return true;
    }
    if (tile != start_tile) {
      return false;
    }
    count += tile;
  }

  return false;
}

static inline int my_max(int n1, int n2) {
  if (n1 > n2) {
    return n1;
  } else {
    return n2;
  }
}

static inline int my_min(int n1, int n2) {
  if (n1 > n2) {
    return n2;
  } else {
    return n1;
  }
}

static int process_counts(int counts[]) {
  int max_x = 0, max_o = 0;
  for (int i = 0; i < 4; i++) {
    int c1 = counts[2 * i];
    int c2 = counts[2 * i + 1];

    if (c1 < 0 && c2 < 0) {
      max_o = my_max(-c1 - c2, max_o);
    } else if (c1 >= 0 && c2 >= 0) {
      max_x = my_max(c1 + c2, max_x);
    } else {
      max_x = my_max(max_x, my_max(c1, c2));
      max_o = my_max(max_o, my_max(-c1, -c2));
    }
  }
  max_x = my_min(max_x, 4);
  max_o = my_min(max_o, 4);

  int diff = max_x - max_o;
  int sign = diff < 0 ? -1 : 1;
  return sign * diff * diff;
}

static int check_5_straight(int r, int c, board_t *board, int &x_4_counts,
                            int &o_4_counts, int &critical_x_4, int &critical_o_4, int &critical_3,
                            int size, int r_min, int r_max, int c_min,
                            int c_max) {
  // consecutive counts in each direction
  int down = 0, up = 0, right = 0, left = 0;
  int down_right = 0, up_left = 0, down_left = 0, up_right = 0;

  bool down_open = check_direction(r, c, 1, 0, down, board, size, r_min, r_max,
                                   c_min, c_max);
  bool up_open =
      check_direction(r, c, -1, 0, up, board, size, r_min, r_max, c_min, c_max);
  bool right_open = check_direction(r, c, 0, 1, right, board, size, r_min,
                                    r_max, c_min, c_max);
  bool left_open = check_direction(r, c, 0, -1, left, board, size, r_min, r_max,
                                   c_min, c_max);
  bool down_right_open = check_direction(r, c, 1, 1, down_right, board, size,
                                         r_min, r_max, c_min, c_max);
  bool up_left_open = check_direction(r, c, -1, -1, up_left, board, size, r_min,
                                      r_max, c_min, c_max);
  bool down_left_open = check_direction(r, c, 1, -1, down_left, board, size,
                                        r_min, r_max, c_min, c_max);
  bool up_right_open = check_direction(r, c, -1, 1, up_right, board, size,
                                       r_min, r_max, c_min, c_max);

  critical_x_4 =
      ((down == 4) || (up == 4) || (left == 4) || (right == 4) ||
       (down_left == 4) || (up_right == 4) || (down_right == 4) ||
       (up_left == 4) || (down + up >= 4) || (down_right + up_left >= 4) ||
       (right + left >= 4) || (down_left + up_right >= 4));
  critical_o_4 =
      ((down == -4) || (up == -4) || (left == -4) || (right == -4) ||
       (down_left == -4) || (up_right == -4) || (down_right == -4) ||
       (up_left == -4) || (down + up <= -4) || (down_right + up_left <= -4) ||
       (right + left <= -4) || (down_left + up_right <= -4));

  // add to totals if found a live 3 or 4
  x_4_counts += critical_x_4;
  o_4_counts += critical_o_4;
  
  // TODO: this stuff is not correct yet, double counting issues
  int critical_x_3 =
      ((down == 3 && down_open) || (left == 3 && left_open) ||
       (right == 3 && right_open) || (up == 3 && up_open) ||
       (down_left == 3 && down_left_open) ||
       (down_right == 3 && down_right_open) || (up_left == 3 && up_left_open) ||
       (up_right == 3 && up_right_open));

  int critical_special_x_3 =
      ((down + up == 3 && down_open && up_open) ||
       (left + right == 3 && left_open && right_open) ||
       (down_right + up_left == 3 && down_right_open && up_left_open) ||
       (down_left + up_right == 3 && down_left_open && up_right_open));

  int critical_o_3 =
      ((down == -3 && down_open) || (left == -3 && left_open) ||
       (right == -3 && right_open) || (up == -3 && up_open) ||
       (down_left == -3 && down_left_open) ||
       (down_right == -3 && down_right_open) ||
       (up_left == -3 && up_left_open) || (up_right == -3 && up_right_open));

  int critical_special_o_3 =
      ((down + up == -3 && down_open && up_open) ||
       (left + right == -3 && left_open && right_open) ||
       (down_right + up_left == -3 && down_right_open && up_left_open) ||
       (down_left + up_right == -3 && down_left_open && up_right_open));

  // add the the set of highest importance
  int critical_4 = critical_x_4 || critical_o_4;
  critical_3 = !critical_4 && (critical_x_3 || critical_o_3 || critical_special_o_3 ||
             critical_special_o_3);

  int counts[8] = {down,      up,       left,       right,
                   down_left, up_right, down_right, up_left};
  int tile_score = process_counts(counts);
  return tile_score;
}

export uniform int eval_ispc(uniform int r_min, uniform int r_max,
                             uniform int c_min, uniform int c_max,
                             uniform int size, uniform board_t board[],
                             uniform int &win_x, uniform int &win_o,
                             uniform int &x_4_count, uniform int &o_4_count,
                             uniform int critical_4[],
                             uniform int critical_3[]) {

  uniform int eval;
  int partial_eval = 0;
  int partial_x_4_count = 0;
  int partial_o_4_count = 0;
  int partial_win_x = 0, partial_win_o = 0;
  foreach (r = r_min... r_max + 1, c = c_min... c_max + 1) {
    // check for 5 in a row if square occupied
    int i = idx(r, c, size);
    
    int critical_x_4, critical_o_4, critical_3_mark;
    int tile = board[i];
    int old_x_4_count = partial_x_4_count, old_o_4_count = partial_o_4_count;
    int local_eval = check_5_straight(
          r, c, board, partial_x_4_count, partial_o_4_count, critical_x_4, critical_o_4,
          critical_3_mark, size, r_min, r_max, c_min, c_max);
    
    if (tile == 0) {
      partial_eval += local_eval;
      critical_4[i] = critical_x_4 || critical_o_4;
      critical_3[i] = critical_3_mark;
    }
    else {
      partial_win_x = tile == 1 && critical_x_4;
      partial_win_o = tile == -1 && critical_o_4;
      partial_x_4_count = old_x_4_count;
      partial_o_4_count = old_o_4_count;
    }
  }
  eval = reduce_add(partial_eval);
  win_x = reduce_max(partial_win_x);
  win_o = reduce_max(partial_win_o);
  x_4_count = reduce_add(partial_x_4_count);
  o_4_count = reduce_add(partial_o_4_count);
  return eval;
}