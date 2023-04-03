#include <cassert>
#include <iostream>
#include <string>
#include <limits>

#include "board.h"
#include "engine_board.h"
#include "timing.h"

using namespace std;
string file_name = "./boards/19x19.txt";

void test_empty_board() {
  Engine_Board b(5);
  b.print();
  for (int i = 0; i < 25; i++) {
    auto moves = b.get_candidate_moves();
    for (auto &m : moves) {
      cout << "(" << m / b.get_size() << "," << m % b.get_size() << ") ";
    }
    cout << endl;
    b.make_move(moves[0]);
    b.print();
  }
  b.save_board(file_name);
}

void load_board_and_move() {
  Board b(file_name);
  b.make_move(0, 2);
  b.print();
  b.make_move(0, 1);
  assert(b.make_move(0, 2) < 0);
  b.print();
  b.save_board(file_name);
}

void test_engine_board() {
  Engine_Board e(5);
  e.print();
  int eval = e.eval();
  e.make_move(0, 1);
  e.make_move(0, 2);
  cout << "Eval: " << eval << endl;
  e.print();
}

void rng_vs_minimax() {
  const int size = 7;
  Engine_Board b(size);
  b.print();
  for (int i = 0; i < size * size; i++) {
    auto moves = b.get_candidate_moves();
    for (auto &m : moves) {
      cout << "(" << m / b.get_size() << "," << m % b.get_size() << ") ";
    }
    cout << endl;
    if (i % 2 == 0) {
      MinimaxResult best_move = b.engine_recomendation();
      b.make_move(best_move.move);
      cout << "Best move: " << best_move.move << endl;
      cout << "Best score: " << best_move.score << endl;
    } else {
      b.make_move(moves[0]);
    }
    b.print();
    if (b.game_over()) {
      cout << "Game over!\n";
      break;
    }
  }
  b.save_board(file_name);
}
int main() {
  // test_empty_board();
  // load_board_and_move();
  // test_engine_board();
  rng_vs_minimax();
  return 0;
}