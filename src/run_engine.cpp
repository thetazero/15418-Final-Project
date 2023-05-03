#include <cassert>
#include <iostream>
#include <limits>
#include <string>

#include "engine/board.h"
#include "engine/engine_board.h"
#include "engine/timing.h"

using namespace std;

#define MAX_DEPTH 7

void print_line(MinimaxResult &line) {
  cout << line.score << ": ";
  for (auto &move : line.moves) {
    int r = move.first;
    int c = move.second;
    cout << "(" << r << "," << c << ") ";
  }
  cout << endl;
}

void search_depth(Engine_Board &b, int d, bool parallel_search, bool parallel_eval, bool fast, bool prune) {
  Engine_Board b_tmp(b);
  b_tmp.set_parallel_eval_mode(parallel_eval);
  b_tmp.set_parallel_search_mode(parallel_search);
  b_tmp.set_fast_mode(fast);
  vector<MinimaxResult> lines = b_tmp.engine_recommendation(d, 3, prune);
  cout << "Depth: " << d << ", Turn: " << (b_tmp.get_turn() == 1 ? "x" : "o")
       << " Parallel Search/Eval: " << parallel_search << "/" << parallel_eval << endl;
  b_tmp.md.print();
  for (auto &line : lines) {
    print_line(line);
  }
  cout << endl;
  int eval = b_tmp.eval();
  cout << "Eval at current position: " << eval << endl;
}


void search_position(string file_name, bool prune) {
  Engine_Board b(file_name);
  // cout << "Current Eval: " << b.eval() << endl;
  // auto moves = b.get_candidate_moves();
  // for (auto &m : moves) {
  //   cout << "(" << m / b.get_size() << "," << m % b.get_size() << ") ";
  // }
  cout << endl;
  b.print();
  for (int d = 1; d <= MAX_DEPTH; d++) {
    search_depth(b, d, false, true, false, prune);
    search_depth(b, d, false, false, false, prune);
    // search_depth(b, d, false, true, true, prune);
    // search_depth(b, d, false, false, true, prune);
    // search_depth(b, d, true, true, true, prune);
  }
}


int main(int argc, char *argv[]) {
  bool prune = true;
  if (argc != 2) {
    printf("Usage: ./run_engine <board_file.board>\n");
    return 0;
  }
  search_position(string(argv[1]), prune);
  return 0;
}