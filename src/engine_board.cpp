#include "engine_board.h"
#include <cassert>

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

void Engine_Board::get_candidate_moves() {
  // most basic: return vector of empty spots within the search space
  // smarter: first keep track of forced move squares (e.g. to stop live 4's and 3's)
}

// TODO: keep track of critical squares?
// search 5 straight dots fomr (r, c) down the row, column, and two diagonals
// return winner eval if someone has 5 in a row
// update counts of live 4's and live 3's for both colors
// live 4: 4 out of 5 squares are one piece, last square empty
// live 3: this specific pattern: [. x x x .]
int Engine_Board::check_5_straight(int r, int c, 
                                   int &x_4_count, int &o_4_count,
                                   int &x_3_count, int &o_3_count) {
  int r_valid = (r_max - r >= 4);
  int c_valid = (c_max - c >= 4);
  int d1_valid = r_valid && c_valid;
  int d2_valid = r_valid && c >= 4;
  
  int rr = 0, cc = 0, d1 = 0, d2 = 0; // counts in each direction
  int rr_3_x = 0, cc_3_x = 0, d1_3_x = 0, d2_3_x = 0; // track live 3's for player x
  int rr_3_o = 0, cc_3_o = 0, d1_3_o = 0, d2_3_o = 0; // track live 3's for player o

  if (d1_valid) {
    d1 = (board[idx(r, c)] + board[idx(r + 1, c + 1)] + board[idx(r + 2, c + 2)] +
          board[idx(r + 3, c + 3)] + board[idx(r + 4, c + 4)]);
    int empty_edge = (board[idx(r, c)] == 0) && (board[idx(r + 4, c + 4)] == 0);
    d1_3_x = empty_edge && (d1 == 3);
    d1_3_o = empty_edge && (d1 == -3);
  }
  if (d2_valid) {
    d2 = (board[idx(r, c)] + board[idx(r + 1, c - 1)] + board[idx(r + 2, c - 2)] +
          board[idx(r + 3, c - 3)] + board[idx(r + 4, c - 4)]);
    int empty_edge = (board[idx(r, c)] == 0) && (board[idx(r + 4, c - 4)] == 0);
    d2_3_x = empty_edge && (d2 == 3);
    d2_3_o = empty_edge && (d2 == -3);
  }
  if (r_valid) {
    rr = (board[idx(r, c)] + board[idx(r + 1, c)] + board[idx(r + 2, c)] +
          board[idx(r + 3, c)] + board[idx(r + 4, c)]);
    int empty_edge = (board[idx(r, c)] == 0) && (board[idx(r + 4, c)] == 0);
    rr_3_x = empty_edge && (rr == 3);
    rr_3_o = empty_edge && (rr == -3);
  }
  if (c_valid) {
    cc = (board[idx(r, c)] + board[idx(r, c + 1)] + board[idx(r, c + 2)] +
          board[idx(r, c + 3)] + board[idx(r, c + 4)]);
    int empty_edge = (board[idx(r, c)] == 0) && (board[idx(r, c + 4)] == 0);
    cc_3_x = empty_edge && (cc == 3);
    cc_3_o = empty_edge && (cc == -3);
  }

  if (d1 == 5 || d2 == 5 || rr == 5 || cc == 5) {
    return GAME_OVER_EVAL;
  }

  if (d1 == -5 || d2 == -5 || rr == -5 || cc == -5) {
    return -1 * GAME_OVER_EVAL;
  }

  x_4_count += ((d1 == 4) + (d2 == 4) + (rr == 4) + (cc == 4));
  o_4_count += ((d1 == -4) + (d2 == -4) + (rr == -4) + (cc == -4));

  x_3_count += (d1_3_x + d2_3_x + rr_3_x + cc_3_x);
  o_3_count += (d1_3_o + d2_3_o + rr_3_o + cc_3_o);

  return 0;
}

int Engine_Board::eval() {
  // number of live 4's for x and o
  int x_4_count = 0, o_4_count = 0;
  // number of live 3's for x and o
  int x_3_count = 0, o_3_count = 0;
  
  for (int r = r_min; r <= r_max; r++) {
    for (int c = c_min; c <= c_max; c++) {
      // check for 5 in a row: game over
      int winner = check_5_straight(r, c, x_4_count, o_4_count, x_3_count, o_3_count);
      if (winner != 0) {
        return winner;
      }
      // add check 6 straight to count live 3's like [. x . x x .]
    }
  }

  cout << "4 counts: " << x_4_count << ", " << o_4_count << endl;
  cout << "3 counts: " << x_3_count << ", " << o_3_count << endl;
  // if you have a live 4 and it is your turn, you will win
  if ((x_4_count > 0 && turn == 1) || (o_4_count > 0 && turn == -1)) {
    return turn * INEVITABLE_WIN_EVAL;
  }

  // should only have max 1 person have 1 or more live 4's
  assert(!(x_4_count > 0 && o_4_count > 0)); 

  // if you have more than 1 live 4, you will win regardless of who's turn
  if (x_4_count > 1) return INEVITABLE_WIN_EVAL;
  if (o_4_count > 1) return -1 * INEVITABLE_WIN_EVAL;

  // one player has single live 4 and opponent is forced to block
  if (x_4_count == 1) {

  }
  else if (o_4_count == 1) {

  }
  else { // no one has live 4

  }
  
  return 0;
}

// update the list of moves
int Engine_Board::make_move(int r, int c) {
  int res = Board::make_move(r, c);
  if (res < 0)
    return res;
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