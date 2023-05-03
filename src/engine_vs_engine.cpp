#include <cassert>
#include <iostream>
#include <sstream>
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

void engine_vs_engine(int depth_x, int depth_o) {
  const int size = 19;
  Engine_Board b(size);
  b.set_parallel_eval_mode(true);
  b.set_parallel_search_mode(true);
  b.set_fast_mode(true);
  b.print();
  for (int i = 0; i < size * size; i++) {
    int depth = (i % 2 == 0) ? depth_x : depth_o;
    vector<MinimaxResult> lines = b.engine_recommendation(depth, 3, true);
    for (auto &line : lines) {
      print_line(line);
    }
    int r = lines[0].moves[0].first;
    int c = lines[0].moves[0].second;
    b.make_move(r, c);
    cout << endl;
    cout << "Eval at current position: " << b.eval() << endl;
    ostringstream fname;
    fname << "./boards/sim_depth_x" << depth_x << "_o" << depth_o << "_mv_" << i << ".board";
    string bfile = fname.str();
    cout << bfile << endl;
    b.print();
    b.save_board(bfile);
    int eval = b.eval();
    if (eval == 1000 || eval == -1000) {
      cout << "Number of evals: " << b.md.eval_count << endl;
      cout << "Game over!\n";
      break;
    }
  }
}

int main(int argc, char *argv[]) {
  int depth_x, depth_o;
  if (argc == 3) {
    depth_x = atoi(argv[1]);
    depth_o = atoi(argv[2]);
  } else {
    depth_x = 5;
    depth_o = 5;
  }

  printf("Engine Game: X Depth %d vs O Depth %d\n", depth_x, depth_o);
  engine_vs_engine(depth_x, depth_o);
  return 0;
}