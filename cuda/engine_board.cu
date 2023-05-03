#include "engine_board.cuh"
#include <cassert>
#include <climits>
#include <stdio.h>
#include <utility>

pair<int, int> readable_move(int i, int size) {
  return make_pair(i / size, i % size);
}

using namespace std;

Engine_Board::Engine_Board() : Engine_Board(19) {}

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

Engine_Board::Engine_Board(string board_state, char board_turn, int board_size)
    : Board(board_state, board_turn, board_size) {
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

  for (int r = r_min; r <= r_max; r++) {
    for (int c = c_min; c <= c_max; c++) {
      int i = idx(r, c);
      if (board[i] == 0) {
        moves.push_back(i);
      }
    }
  }
  return moves;
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
          return winner;
        }
      }
    }
  }
  return 0;
}

bool Engine_Board::in_bounds(int r, int c) {
  return r >= 0 && r < size && c >= 0 && c < size;
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

int Engine_Board::cuda_recommendation(int max_depth) {
  size_t board_mem_size = size * size;
  size_t potential_boards = 1;
  size_t max_board_width = max_depth * 2 + 1;
  for (int i =0; i < max_depth; i++) {
    potential_boards *= (max_board_width * max_board_width - 1 -i);
  }
  char *boards = new char[potential_boards * board_mem_size];

  int i = 0;
  bool isMax = turn == 1;

  cuda_minimax_stage(max_depth, 0, isMax, boards, &i);
  int *evals = new int[potential_boards];

  eval_wrapper(size, boards, evals, potential_boards);

  cuda_minimax(max_depth, 0, isMax, evals, &i);
  return fast_root_best_move;
}

void Engine_Board::cuda_minimax_stage(int max_depth, int depth,
                                      bool isMax, char *boards, int *board_idx) {
  if (depth == max_depth) {
    size_t board_offset = *board_idx * size * size;
    for (int r = 0; r < size; r++) {
      for (int c = 0; c < size; c++) {
        boards[board_offset + idx(r, c)] = board[idx(r, c)];
      }
    }
    *board_idx = *board_idx + 1;
    return;
  }
  vector<int> moves = get_candidate_moves();

  for (int i = 0; i < moves.size(); i++) {
    int old_r_min = r_min, old_c_min = c_min;
    int old_r_max = r_max, old_c_max = c_max;
    make_move(moves[i]);

    cuda_minimax_stage(max_depth, depth + 1, !isMax, boards, board_idx);

    undo_move(moves[i]);
    r_min = old_r_min;
    c_min = old_c_min;
    r_max = old_r_max;
    c_max = old_c_max;
  }
}

int Engine_Board::cuda_minimax(const int max_depth, const int depth,
                               const bool isMax, int *evals, int *eval_idx) {
  if (game_over() != 0) {
    return game_over();
  }
  if (depth == max_depth) {
    int e = evals[*eval_idx];
    *eval_idx = *eval_idx + 1;
    return e;
  }

  vector<int> moves = get_candidate_moves();
  int best_move = isMax ? INT_MIN : INT_MAX;

  for (int i = 0; i < moves.size(); i++) {
    int old_r_min = r_min, old_c_min = c_min;
    int old_r_max = r_max, old_c_max = c_max;
    make_move(moves[i]);
    int res = cuda_minimax(max_depth, depth + 1, !isMax, evals, eval_idx);

    if (isMax) {
      if (res > best_move) {
        best_move = res;
        if (depth == 0)
          fast_root_best_move = moves[i];
      }
    } else {
      if (res < best_move) {
        best_move = res;
        if (depth == 0)
          fast_root_best_move = moves[i];
      }
    }

    undo_move(moves[i]);
    r_min = old_r_min;
    c_min = old_c_min;
    r_max = old_r_max;
    c_max = old_c_max;
  }
  return best_move;
}
