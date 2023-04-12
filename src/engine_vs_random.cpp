#include <cassert>
#include <iostream>
#include <limits>
#include <stdlib.h>
#include <string>

#include "engine/board.h"
#include "engine/engine_board.h"
#include "engine/timing.h"

using namespace std;

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
      int r = lines[0].moves[0].first;
      int c = lines[0].moves[0].second;
      b.make_move(r, c);
      cout << endl;
      cout << "Eval at current position: " << b.eval() << endl;
    } else {
      auto moves = b.get_candidate_moves();
      int r = rand() % moves.size();
      b.make_move(moves[r]);
    }
    b.print();
    int eval = b.eval();
    if (eval == 10000 || eval == -10000) {
      cout << "Number of evals: " << b.md.eval_count << endl;
      cout << "Game over!\n";
      break;
    }
  }
}

int main(int argc, char *argv[]) {
  int depth;
  if (argc == 1) {
    depth = 5;
  } else if (argc == 2) {
    depth = atoi(argv[1]);
  } else {
    printf("Usage: ./engine_vs_random <depth>\n");
    return 0;
  }
  printf("Random Game with Depth %d Engine\n", depth);
  rng_vs_minimax(depth);
  return 0;
}