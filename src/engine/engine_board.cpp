#include "engine_board.h"
#include "objs/eval_ispc.h"
#include "timing.h"
#include <algorithm> // std::reverse
#include <cassert>
#include <climits>
#include <stdio.h>
#include <utility>

using namespace std;

// void print_line(MinimaxResult &line) {
//   cout << line.score << ": ";
//   for (auto &move : line.moves) {
//     int r = move.first;
//     int c = move.second;
//     cout << "(" << r << "," << c << ") ";
//   }
//   cout << endl;
// }

Engine_Board::Engine_Board(int board_size = 19) : Board(board_size) {
  r_min = size;
  c_min = size;
  r_max = -1;
  c_max = -1;
}

Engine_Board::Engine_Board(string filename) : Board(filename) {
  r_min = size;
  c_min = size;
  r_max = -1;
  c_max = -1;
  // establish bounds of pieces
  for (int r = 0; r < size; r++) {
    for (int c = 0; c < size; c++) {
      if (board[idx(r, c)] != 0) {
        update_bounds(r, c);
      }
    }
  }
}

Engine_Board::Engine_Board(Engine_Board &b) : Board(b) {
  r_min = b.r_min;
  c_min = b.c_min;
  r_max = b.r_max;
  c_max = b.c_max;
  critical_4 = b.critical_4;
  critical_3 = b.critical_3;
}

vector<int> Engine_Board::get_candidate_moves() {
  // most basic: return vector of empty spots within the search space
  // smarter: first keep track of forced move squares (e.g. to stop live 4's and
  // 3's)
  vector<int> moves;

  // if empty board, just return the center square
  if (r_max < 0) {
    moves.push_back(idx(size / 2, size / 2));
    return moves;
  }

  // first put all critical 4 and 3 squares at the front of moves
  for (int i : critical_4) {
    if (!board[i])
      moves.push_back(i);
  }
  for (int i : critical_3) {
    if (!board[i])
      moves.push_back(i);
  }

  for (int r = r_min; r <= r_max; r++) {
    for (int c = c_min; c <= c_max; c++) {
      int i = idx(r, c);
      // add empty squares that haven't already been added
      if (board[i] == 0 && !critical_4.count(i) && !critical_3.count(i)) {
        moves.push_back(i);
      }
    }
  }
  return moves;
}

bool Engine_Board::check_direction(int r, int c, int dr, int dc, int &count) {
  int start_tile = board[idx(r + dr, c + dc)]; // the first tile in the sequence
  for (int i = 0; i < 4; i++) {
    // move to the next tile in direction
    r += dr;
    c += dc;
    // if out of bounds, not an open edge
    if (r > r_max || r < r_min || c > c_max || c < c_min) {
      return false;
    }

    int tile = board[idx(r, c)];

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

int Engine_Board::process_counts(int *counts) {
  int max_x = 0, max_o = 0;
  for (int i = 0; i < 4; i++) {
    int c1 = counts[2 * i];
    int c2 = counts[2 * i + 1];

    if (c1 < 0 && c2 < 0) {
      max_o = max(-c1 - c2, max_o);
    } else if (c1 >= 0 && c2 >= 0) {
      max_x = max(c1 + c2, max_x);
    } else {
      max_x = max(max_x, max(c1, c2));
      max_o = max(max_o, max(-c1, -c2));
    }
  }
  max_x = min(max_x, 4);
  max_o = min(max_o, 4);

  int diff = max_x - max_o;
  int sign = diff < 0 ? -1 : 1;
  return sign * diff * diff;
}

// TODO: keep track of critical squares?
// search 5 straight dots fomr (r, c) down the row, column, and two diagonals
// update counts of live 4's and live 3's for both colors
// live 4: 4 out of 5 squares are one piece, last square empty
// live 3: this specific pattern: [. x x x .]
int Engine_Board::check_5_straight(int r, int c, int &x_4_count,
                                   int &o_4_count) {
  // consecutive counts in each direction
  int down = 0, up = 0, right = 0, left = 0;
  int down_right = 0, up_left = 0, down_left = 0, up_right = 0;

  bool down_open = check_direction(r, c, 1, 0, down);
  bool up_open = check_direction(r, c, -1, 0, up);
  bool right_open = check_direction(r, c, 0, 1, right);
  bool left_open = check_direction(r, c, 0, -1, left);
  bool down_right_open = check_direction(r, c, 1, 1, down_right);
  bool up_left_open = check_direction(r, c, -1, -1, up_left);
  bool down_left_open = check_direction(r, c, 1, -1, down_left);
  bool up_right_open = check_direction(r, c, -1, 1, up_right);

  int critical_x_4 =
      ((down == 4) || (up == 4) || (left == 4) || (right == 4) ||
       (down_left == 4) || (up_right == 4) || (down_right == 4) ||
       (up_left == 4) || (down + up >= 4) || (down_right + up_left >= 4) ||
       (right + left >= 4) || (down_left + up_right >= 4));
  int critical_o_4 =
      ((down == -4) || (up == -4) || (left == -4) || (right == -4) ||
       (down_left == -4) || (up_right == -4) || (down_right == -4) ||
       (up_left == -4) || (down + up <= -4) || (down_right + up_left <= -4) ||
       (right + left <= -4) || (down_left + up_right <= -4));

  // add to totals if found a live 3 or 4
  x_4_count += critical_x_4;
  o_4_count += critical_o_4;

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

  // add to totals, avoiding double counting

  // x_3_count += critical_x_3;
  // if (!critical_x_3) {
  //   special_x_3_count += critical_special_x_3;
  // }

  // o_3_count += critical_o_3;
  // if (!critical_o_3) {
  //   special_o_3_count += critical_special_o_3;
  // }

  // add the the set of highest importance
  if (critical_x_4 || critical_o_4) {
    critical_4.insert(idx(r, c));
  } else if (critical_x_3 || critical_o_3 || critical_special_o_3 ||
             critical_special_o_3) {
    critical_3.insert(idx(r, c));
  }

  int counts[8] = {down,      up,       left,       right,
                   down_left, up_right, down_right, up_left};
  int tile_score = process_counts(counts);
  return tile_score;
}

int Engine_Board::game_over(int r, int c) {
  int r_valid = (r_max - r >= 4);
  int c_valid = (c_max - c >= 4);
  int d1_valid = r_valid && c_valid;
  int d2_valid = r_valid && c >= 4;

  int rr = 0, cc = 0, d1 = 0, d2 = 0; // counts in each direction
  if (d1_valid) {
    d1 = (board[idx(r, c)] + board[idx(r + 1, c + 1)] +
          board[idx(r + 2, c + 2)] + board[idx(r + 3, c + 3)] +
          board[idx(r + 4, c + 4)]);
  }
  if (d2_valid) {
    d2 = (board[idx(r, c)] + board[idx(r + 1, c - 1)] +
          board[idx(r + 2, c - 2)] + board[idx(r + 3, c - 3)] +
          board[idx(r + 4, c - 4)]);
  }
  if (r_valid) {
    rr = (board[idx(r, c)] + board[idx(r + 1, c)] + board[idx(r + 2, c)] +
          board[idx(r + 3, c)] + board[idx(r + 4, c)]);
  }
  if (c_valid) {
    cc = (board[idx(r, c)] + board[idx(r, c + 1)] + board[idx(r, c + 2)] +
          board[idx(r, c + 3)] + board[idx(r, c + 4)]);
  }

  if (d1 == 5 || d2 == 5 || rr == 5 || cc == 5) {
    return GAME_OVER_EVAL;
  }

  if (d1 == -5 || d2 == -5 || rr == -5 || cc == -5) {
    return -1 * GAME_OVER_EVAL;
  }

  return 0;
}

int Engine_Board::game_over() {
  for (int r = r_min; r <= r_max; r++) {
    for (int c = c_min; c <= c_max; c++) {
      // check for 5 in a row if square occupied
      if (board[idx(r, c)] != 0) {
        int winner = game_over(r, c);
        if (winner != 0) {
          return 1;
        }
      }
    }
  }
  return 0;
}

int Engine_Board::sequential_eval(int &x_4_count, int &o_4_count) {
  int eval = 0;
  for (int r = r_min; r <= r_max; r++) {
    for (int c = c_min; c <= c_max; c++) {
      // check for 5 in a row if square occupied
      if (board[idx(r, c)] != 0) {
        int winner = game_over(r, c);
        if (winner != 0) {
          return winner;
        }
      } else { // check if filling in the spot would form live 4 or 3
        eval += check_5_straight(r, c, x_4_count, o_4_count);
      }
    }
  }

  return eval;
}

int Engine_Board::ispc_eval(int &x_4_count, int &o_4_count) {
  int total = size * size;
  int *critical_4_mark = new int[total];
  int *critical_3_mark = new int[total];
  int win_x, win_o;
  memset(critical_4_mark, 0, total);
  memset(critical_3_mark, 0, total);

  int eval =
      ispc::eval_ispc(r_min, r_max, c_min, c_max, size, board, win_x, win_o,
                      x_4_count, o_4_count, critical_4_mark, critical_3_mark);

  if (win_x) {
    eval = GAME_OVER_EVAL;
  } 
  if (win_o) {
    eval = -1 * GAME_OVER_EVAL;
  }
  // add to critical set
  for (int r = r_min; r <= r_max; r++) {
    for (int c = c_min; c <= c_max; c++) {
      int i = idx(r, c);
      if (critical_4_mark[i]) {
        critical_4.insert(i);
      } else if (critical_3_mark[i]) {
        critical_3.insert(i);
      }
    }
  }
  delete critical_4_mark, critical_3_mark;
  return eval;
}

int Engine_Board::eval() {
  md.eval_count++;
  Timer t;
  // number of live 4's for x and o
  int x_4_count = 0, o_4_count = 0;

  // clear critical square set
  critical_4.clear();
  critical_3.clear();

  int eval = sequential_eval(x_4_count, o_4_count);
  // int eval = ispc_eval(x_4_count, o_4_count);
  // cout << "4 counts: " << x_4_count << ", " << o_4_count << endl;
  if (eval == GAME_OVER_EVAL || eval == -1 * GAME_OVER_EVAL) {
    md.eval_time += t.elapsed();
    return eval;
  }

  // TODO: this stuff is not correct yet with the 3 counts, double counting
  // cout << "4 counts: " << x_4_count << ", " << o_4_count << endl;

  // if you have a live 4 and it is your turn, you will win
  if ((x_4_count > 0 && turn == 1) || (o_4_count > 0 && turn == -1)) {
    return turn * INEVITABLE_WIN_4_EVAL;
  }

  // should only have max 1 person have 1 or more live 4's
  assert(!(x_4_count > 0 && o_4_count > 0));

  // if you have more than 1 live 4, you will win regardless of who's turn
  if (x_4_count > 1)
    return INEVITABLE_WIN_4_EVAL;
  if (o_4_count > 1)
    return -1 * INEVITABLE_WIN_4_EVAL;

  // one player has single live 4 and opponent is forced to block
  if (x_4_count == 1) {

  } else if (o_4_count == 1) {

  } else { // no one has live 4
  }

  // if (x_3_count >= 1) {
  //   if (turn == 1) { // your turn and you have at least a live 3
  //     if (o_4_count == 0) { // if they don't have immediate live 4, you win
  //       return INEVITABLE_WIN_3_EVAL;
  //     }
  //   }
  //   else { // their turn and you have at least a live 3

  //   }
  // }
  // if (o_3_count >= 1) {
  //   if (x_4_count == 0 && turn == -1) {
  //     return -1*INEVITABLE_WIN_3_EVAL;
  //   }
  // }
  md.eval_time += t.elapsed();
  return eval;
}

// update the list of moves
int Engine_Board::make_move(int r, int c) {
  int res = Board::make_move(r, c);
  if (res < 0)
    return res;
  update_bounds(r, c);
  return res;
}

int Engine_Board::make_move(int i) {
  int res = Board::make_move(i);
  if (res < 0)
    return res;
  int r = i / size;
  int c = i % size;
  update_bounds(r, c);
  return res;
}

int Engine_Board::undo_move(int i) {
  int res = Board::undo_move(i);
  if (res < 0)
    return res;
  int r = i / size;
  int c = i % size;
  update_bounds(r, c);
  return res;
}

void Engine_Board::update_bounds(int r, int c) {
  r_min = max(min(r - 1, (int)r_min), 0);
  c_min = max(min(c - 1, (int)c_min), 0);
  r_max = min(max(r + 1, (int)r_max), size - 1);
  c_max = min(max(c + 1, (int)c_max), size - 1);
  // print_bounds();
}

void Engine_Board::print_bounds() {
  cout << "(" << (int)r_min << ", " << (int)c_min << ") "
       << "(" << (int)r_max << ", " << (int)c_max << ")\n";
}

// helper functions to sort lines
bool compare_lines_gt(MinimaxResult &l1, MinimaxResult &l2) {
  return (l1.score > l2.score) ||
         (l1.score == l2.score && l1.moves.size() < l2.moves.size());
}

bool compare_lines_lt(MinimaxResult &l1, MinimaxResult &l2) {
  return (l1.score < l2.score) ||
         (l1.score == l2.score && l1.moves.size() < l2.moves.size());
}

vector<MinimaxResult>
Engine_Board::engine_recommendation(int depth, int num_lines, bool prune) {
  md.eval_count = 0;
  md.eval_time = 0;
  md.prune_count.clear();

  Timer t;
  bool isMax = turn == 1;
  vector<MinimaxResult> lines;

  MinimaxResult result =
      minimax(depth, 0, lines, isMax, INT_MIN, INT_MAX, true);

  md.total_time = t.elapsed();
  md.print();

  // sort lines in order of evaluation, and keep only top lines
  if (turn == 1)
    sort(lines.begin(), lines.end(), compare_lines_gt);
  else
    sort(lines.begin(), lines.end(), compare_lines_lt);
  lines.resize(num_lines);

  for (int i = 0; i < lines.size(); i++) {
    auto &line = lines[i];
    reverse(line.moves.begin(), line.moves.end()); // moves are stored backwards
  }

  return lines;
}

MinimaxResult Engine_Board::minimax(int max_depth, int depth,
                                    vector<MinimaxResult> &lines, bool isMax,
                                    int alpha, int beta, bool prune) {
  if (depth == max_depth) {
    return MinimaxResult{eval(), vector<pair<int, int>>()};
  }

  MinimaxResult best_move;
  int e = eval();
  if (e == GAME_OVER_EVAL || e == -1 * GAME_OVER_EVAL) {
    return MinimaxResult{eval(), vector<pair<int, int>>()};
  }
  vector<int> moves = get_candidate_moves();

  best_move.score = isMax ? INT_MIN : INT_MAX;

  for (int i = 0; i < moves.size(); i++) {
    int8_t old_r_min = r_min, old_c_min = c_min;
    int8_t old_r_max = r_max, old_c_max = c_max;
    make_move(moves[i]);

    if (game_over()) {
      int e = eval();
      undo_move(moves[i]);
      r_min = old_r_min;
      c_min = old_c_min;
      r_max = old_r_max;
      c_max = old_c_max;
      return MinimaxResult{e, vector<pair<int, int>>(1, rc(moves[i]))};
    }

    MinimaxResult res =
        minimax(max_depth, depth + 1, lines, !isMax, alpha, beta, prune);
    res.moves.push_back(rc(moves[i]));

    // add to set of lines if at root node
    if (depth == 0) {
      lines.push_back(res);
    }

    if (isMax) {
      if (res.score > best_move.score) {
        best_move.score = res.score;
        best_move.moves = res.moves;
      }
      alpha = max(alpha, best_move.score);
    } else {
      if (res.score < best_move.score) {
        best_move.score = res.score;
        best_move.moves = res.moves;
      }
      beta = min(beta, best_move.score);
    }

    undo_move(moves[i]);
    r_min = old_r_min;
    c_min = old_c_min;
    r_max = old_r_max;
    c_max = old_c_max;

    if (prune && (beta < alpha)) {
      if (md.prune_count.count(depth) > 0) {
        md.prune_count[depth].second += 1;
      } else {
        md.prune_count[depth].second = 1;
      }
      break;
    } else {
      if (md.prune_count.count(depth) > 0) {
        md.prune_count[depth].first += 1;
      } else {
        md.prune_count[depth].first = 1;
      }
    }
  }

  return best_move;
}