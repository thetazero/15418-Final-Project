#include <cassert>
#include <iostream>
#include <limits>
#include <string>

#include "board.h"
#include "engine_board.h"
#include "timing.h"

using namespace std;

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
  Engine_Board e(file_name);
  e.print();
  int eval = e.eval();
  // e.save_board(file_name);
  cout << "Eval: " << eval << endl;
  // e.print();
}

void print_line(MinimaxResult &line) {
  cout << line.score << ": ";
  for (auto &move : line.moves) {
    int r = move.first;
    int c = move.second;
    cout << "(" << r << "," << c << ") ";
  }
  cout << endl;
}

void rng_vs_minimax(int depth) {
  const int size = 19;
  Engine_Board b(size);
  b.print();
  for (int i = 0; i < size * size; i++) {
    if (i % 2 == 0) {
      vector<MinimaxResult> lines = b.engine_recommendation(depth, 3, true);
      for (auto &line : lines) {
        print_line(line);
      }
      cout << endl;
      cout << "Eval at current position: " << b.eval() << endl;
    } else {
      auto moves = b.get_candidate_moves();
      b.make_move(moves[0]);
    }
    b.print();
    int eval = b.eval();
    if (eval == 10000 || eval == -10000) {
      cout << "Number of evals: " << b.eval_count << endl;
      cout << "Game over!\n";
      break;
    }
  }
  b.save_board(file_name);
}

int main(int argc, char *argv[]) {
  int depth;
  if (argc == 1) {
    depth = 5;
  } else if (argc == 2) {
    depth = atoi(argv[1]);
  } else {
    printf("Usage: ./engine_vs_random <depth>\n");
    return;
  }
  printf("Random Game with Depth %d Engine\n");
  rng_vs_minimax(depth);
  return 0;
}