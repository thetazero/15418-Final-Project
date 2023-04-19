#include <cassert>
#include <iostream>
#include <limits>
#include <string>

#include "engine/board.h"
#include "engine/engine_board.h"

using namespace std;

void get_eval(string file_name) {
  Engine_Board b(file_name);
  b.set_parallel_eval_mode(false);
  int eval;

  b.md.eval_count = 0;
  b.md.eval_time = 0;
  eval = b.eval();
  cout << "Sequential Eval: " << eval << endl;  
  b.md.print();

  b.set_parallel_eval_mode(true);
  b.md.eval_count = 0;
  b.md.eval_time = 0;
  eval = b.eval();
  cout << "Parallel Eval: " << eval << endl;
  b.md.print();
  
}

int main(int argc, char *argv[]) {
  bool prune = true;
  if (argc != 2) {
    printf("Usage: ./eval_board <board_file.txt>\n");
    return 0;
  }
  get_eval(string(argv[1]));
  return 0;
}