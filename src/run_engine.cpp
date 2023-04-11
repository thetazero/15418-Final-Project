#include <cassert>
#include <iostream>
#include <limits>
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

void search_position(string file_name) {
  Engine_Board b(file_name);
  cout << "Current Eval: " << b.eval() << endl;
  auto moves = b.get_candidate_moves();
  for (auto &m : moves) {
    cout << "(" << m / b.get_size() << "," << m % b.get_size() << ") ";
  }
  cout << endl;
  for (int d = 1; d <= 7; d++) {
    Engine_Board b_tmp(b);
    vector<MinimaxResult> lines = b.engine_recommendation(d, 3, true);
    cout << "Depth: " << d << ", Turn: " << (b.get_turn() == 1 ? "x" : "o")
         << endl;
    for (auto &line : lines) {
      print_line(line);
    }
    cout << endl;
    cout << "Eval at current position: " << b.eval() << endl;
  }
}
int main(int argc, char *argv[]) {
  cout << argc << endl;
  if (argc != 2) {
    printf("Usage: ./run_engine <board_file.txt>\n");
    return 0;
  }
  search_position(string(argv[1]));
  return 0;
}