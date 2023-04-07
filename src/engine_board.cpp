#include "engine_board.h"
#include "timing.h"
#include <cassert>
#include <climits>
#include <stdio.h>

using namespace std;
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

void Engine_Board::check_direction(int r, int c, int dr, int dc, 
                                   int &count, int &x_3, int &o_3) {
  count = (board[idx(r + 1*dr, c + 1*dc)] + board[idx(r + 2*dr, c + 2*dc)] + 
           board[idx(r + 3*dr, c + 3*dc)] + board[idx(r + 4*dr, c + 4*dc)]);
  int empty_edge = board[idx(r + 4*dr, c + 4*dc)] == 0;
  x_3 = empty_edge && (count == 3);
  o_3 = empty_edge && (count == -3);
  // printf("(%d, %d) (%d, %d): count: %d, (%d, %d)\n", r, c, dr, dc, count, x_3, o_3);
}

// TODO: keep track of critical squares?
// search 5 straight dots fomr (r, c) down the row, column, and two diagonals
// update counts of live 4's and live 3's for both colors
// live 4: 4 out of 5 squares are one piece, last square empty
// live 3: this specific pattern: [. x x x .]
void Engine_Board::check_5_straight(int r, int c, int &x_4_count, int &o_4_count,
                                   int &x_3_count, int &o_3_count) {
  int valid_down = (r_max - r >= 4);
  int valid_up = (r - r_min >= 4);
  int valid_right = (c_max - c >= 4);
  int valid_left = (c - c_min >= 4);
  int valid_down_right = valid_down && valid_right;
  int valid_up_left = valid_up && valid_left;
  int valid_down_left = valid_down && valid_left;
  int valid_up_right = valid_up && valid_right;

  // consecutive counts in each direction
  int down = 0, up = 0, right = 0, left = 0;
  int down_right = 0, up_left = 0, down_left = 0, up_right = 0; 
  
  // track live 3's for x and o
  int down_x_3 = 0, up_x_3 = 0, right_x_3 = 0, left_x_3 = 0;
  int down_right_x_3 = 0, up_left_x_3 = 0, down_left_x_3 = 0, up_right_x_3 = 0;
  int down_o_3 = 0, up_o_3 = 0, right_o_3 = 0, left_o_3 = 0;
  int down_right_o_3 = 0, up_left_o_3 = 0, down_left_o_3 = 0, up_right_o_3 = 0;

  if (valid_down) {
    check_direction(r, c, 1, 0, down, down_x_3, down_o_3);
  }
  if (valid_up) {
    check_direction(r, c, -1, 0, up, up_x_3, up_o_3);
  }
  if (valid_right) {
    check_direction(r, c, 0, 1, right, right_x_3, right_o_3);
  }
  if (valid_left) {
    check_direction(r, c, 0, -1, left, left_x_3, left_o_3);
  }
  if (valid_down_right) {
    check_direction(r, c, 1, 1, down_right, down_right_x_3, down_right_o_3);
  }
  if (valid_up_left) {
    check_direction(r, c, -1, -1, up_left, up_left_x_3, up_left_o_3);
  }
  if (valid_down_left) {
    check_direction(r, c, 1, -1, down_left, down_left_x_3, down_left_o_3);
  }
  if (valid_up_right) {
    check_direction(r, c, -1, 1, up_right, up_right_x_3, up_right_o_3);
  }

  int critical_x_4 = (down == 4 || up == 4 || right == 4|| left == 4 ||
                      down_right == 4 || up_left == 4 || 
                      down_left == 4 || up_right == 4);
  int critical_o_4 = (down == -4 || up == -4 || right == -4|| left == -4 ||
                      down_right == -4 || up_left == -4 || 
                      down_left == -4 || up_right == -4);
  // add to totals if found a live 3 or 4
  x_4_count += critical_x_4;
  o_4_count += critical_o_4;

  int critical_x_3 = (down_x_3 || up_x_3 || right_x_3 || left_x_3 || 
                      down_right_x_3 || up_left_x_3 || 
                      down_left_x_3 || up_right_x_3);
  int critical_o_3 = (down_o_3 || up_o_3 || right_o_3 || left_o_3 || 
                      down_right_o_3 || up_left_o_3 || 
                      down_left_o_3 || up_right_o_3);
  x_3_count += critical_x_3;
  o_3_count += critical_o_3;

  // add the the set of highest importance
  if (critical_x_4 || critical_o_4) {
    critical_4.insert(idx(r, c));
  }
  else if (critical_x_3 || critical_o_3){
    critical_3.insert(idx(r, c));
  }
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

void Engine_Board::check_special_3(int r, int c, int &x_3_count, int &o_3_count) {
  int valid_up = (r_max - r >= 2) && (r - r_min >= 3);
  int valid_down = (r_max - r >= 3) && (r - r_min >= 2);
  int valid_left = (c_max - c >= 2) && (c - c_min >= 3);
  int valid_right = (c_max - c >= 3) && (c - c_min >= 2);
  int valid_down_left = valid_down && valid_left;
  int valid_up_right = valid_up && valid_right;
  int valid_down_right = valid_down && valid_right;
  int valid_up_left = valid_up && valid_left;

  int up_x = 0, down_x = 0, left_x = 0, right_x = 0;
  int up_right_x = 0, down_left_x = 0, up_left_x = 0, down_right_x = 0;
  int up_o = 0, down_o = 0, left_o = 0, right_o = 0;
  int up_right_o = 0, down_left_o = 0, up_left_o = 0, down_right_o = 0;

  if (valid_up) {
    check_special(r, c, -1, 0, up_x, up_o);
  }
  if (valid_down) {
    check_special(r, c, 1, 0, down_x, down_o);
  }
  if (valid_left) {
    check_special(r, c, 0, -1, left_x, left_o);
  }
  if (valid_right) {
    check_special(r, c, 0, 1, right_x, right_o);
  }
  if (valid_down_left) {
    check_special(r, c, 1, -1, down_left_x, down_left_o);
  }
  if (valid_up_right) {
    check_special(r, c, -1, 1, up_right_x, up_right_o);
  }
  if (valid_down_right) {
    check_special(r, c, 1, 1, down_right_x, down_right_o);
  }
  if (valid_up_left) {
    check_special(r, c, -1, -1, up_left_x, up_left_o);
  }

  int critical_x_3 = (up_x || down_x || right_x || left_x || 
                      down_left_x || up_right_x || down_right_x || up_left_x);
  int critical_o_3 = (up_o || down_o || right_o || left_o || 
                      down_left_o || up_right_o || down_right_o || up_left_o);
  x_3_count += critical_x_3;
  o_3_count += critical_o_3;

  if (critical_x_3 || critical_o_3) {
    critical_3.insert(idx(r, c));
  }
}

void Engine_Board::check_special(int r, int c, int dr, int dc,
                                 int &x_3, int &o_3) {
  int sum = (board[idx(r + 1*dr, c + 1*dc)] + board[idx(r + 2*dr, c + 2*dc)] + 
             board[idx(r - 1*dr, c - 1*dc)]);
  int open = !board[idx(r - 2*dr, c - 2*dc)] && !board[idx(r + 3*dr, c + 3*dc)];
  x_3 += (sum == 3) && open;
  o_3 += (sum == -3) && open;
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

int Engine_Board::eval() {
  eval_count++;
  // number of live 4's for x and o
  int x_4_count = 0, o_4_count = 0;
  // number of live 3's for x and o
  int x_3_count = 0, o_3_count = 0;

  // clear critical square set
  critical_4.clear();
  critical_3.clear();

  for (int r = r_min; r <= r_max; r++) {
    for (int c = c_min; c <= c_max; c++) {
      // check for 5 in a row if square occupied
      if (board[idx(r, c)] != 0) {
        int winner = game_over(r, c);
        if (winner != 0) {
          return winner;
        }
      }
      else { // check if filling in the spot would form live 4 or 3
        check_5_straight(r, c, x_4_count, o_4_count, x_3_count, o_3_count);
        check_special_3(r, c, x_3_count, o_3_count);
      }
    }
  }

  // cout << "4 counts: " << x_4_count << ", " << o_4_count << endl;
  // cout << "3 counts: " << x_3_count << ", " << o_3_count << endl;

  // if you have a live 4 and it is your turn, you will win
  if ((x_4_count > 0 && turn == 1) || (o_4_count > 0 && turn == -1)) {
    return turn * INEVITABLE_WIN_EVAL;
  }

  // should only have max 1 person have 1 or more live 4's
  assert(!(x_4_count > 0 && o_4_count > 0));

  // if you have more than 1 live 4, you will win regardless of who's turn
  if (x_4_count > 1)
    return INEVITABLE_WIN_EVAL;
  if (o_4_count > 1)
    return -1 * INEVITABLE_WIN_EVAL;

  // one player has single live 4 and opponent is forced to block
  if (x_4_count == 1) {

  } else if (o_4_count == 1) {

  } else { // no one has live 4
  }

  if (x_3_count >= 1 && turn == 1) {

  }
  if (o_3_count >= 1 && turn == -1) {

  }
  return x_3_count - o_3_count;
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

MinimaxResult Engine_Board::engine_recommendation(int depth, bool prune) {
  eval_count = 0;
  MinimaxResult result;
  Timer t;
  bool isMax = turn == 1;
  vector<vector<int>> lines;
  if (prune) {
    result = minimax_alpha_beta(depth, 0, isMax, INT_MIN, INT_MAX);
  }    
  else {
    result = minimax(depth, 0, isMax);
  }
  double elapsed = t.elapsed();
  cout << "Eval Count: " << eval_count  << ", Time: " << elapsed << endl;
  return result;
    
}

MinimaxResult Engine_Board::minimax(int max_depth, int depth, bool isMax) {
  if (depth == max_depth) {
    return MinimaxResult{eval(), -1};
  }
  MinimaxResult best_move;
  vector<int> moves = get_candidate_moves();
  if (isMax) {
    best_move.score = -999999;
    for (int i = 0; i < 3; i++) {
      make_move(moves[i]);
      print();
      if (game_over()) {
        cout << eval() << endl;
        return MinimaxResult{eval(), 0};
      }
      MinimaxResult res = minimax(max_depth, depth + 1, !isMax);
      if (res.score > best_move.score) {
        best_move.score = res.score;
        best_move.move = moves[i];
      }
      undo_move(moves[i]);
    }
  } else {
    best_move.score = 999999;
    for (int i = 0; i < 3; i++) {
      make_move(moves[i]);
      print();
      if (game_over()) {
        return MinimaxResult{eval(), 0};
      }
      MinimaxResult res = minimax(max_depth, depth + 1, !isMax);
      if (res.score < best_move.score) {
        best_move.score = res.score;
        best_move.move = moves[i];
      }
      undo_move(moves[i]);
    }
  }
  return best_move;
}

MinimaxResult Engine_Board::minimax_alpha_beta(int max_depth, int depth, 
                                               bool isMax, int alpha, int beta) {
  if (depth == max_depth) {
    // cout << "Depth Reached: " << eval() << endl;
    return MinimaxResult{eval(), -1};
  }
  MinimaxResult best_move;
  int e = eval();
  vector<int> moves = get_candidate_moves();
  
  best_move.score = isMax ? INT_MIN : INT_MAX;

  for (int i = 0; i < moves.size(); i++) {
    char old_r_min = r_min, old_c_min = c_min;
    char old_r_max = r_max, old_c_max = c_max;
    make_move(moves[i]);
    
    if (game_over()) {
      int e = eval();
      undo_move(moves[i]);
      return MinimaxResult{e, 0};
    }

    MinimaxResult res = minimax_alpha_beta(max_depth, depth + 1, !isMax, alpha, beta);
    if (isMax) {
      if (res.score > best_move.score) {
        best_move.score = res.score;
        best_move.move = moves[i];
      }
      alpha = max(alpha, best_move.score);
    }
    else {
      if (res.score < best_move.score) {
        best_move.score = res.score;
        best_move.move = moves[i];
      }
      beta = min(beta, best_move.score);
    }

    undo_move(moves[i]);
    r_min = old_r_min;
    c_min = old_c_min;
    r_max = old_r_max;
    c_max = old_c_max;
    
    if (alpha == GAME_OVER_EVAL || beta == -1*GAME_OVER_EVAL || beta <= alpha)
      break;
  }

  return best_move;
}