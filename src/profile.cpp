#include <cassert>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <unistd.h>

#include "engine/board.h"
#include "engine/engine_board.h"
#include "engine/timing.h"

using namespace std;


void search_depth(Engine_Board &b, int d, int max_depth, bool parallel_search, bool parallel_eval, bool prune) {
  Engine_Board b_tmp(b);
  b_tmp.set_parallel_eval_mode(parallel_eval);
  b_tmp.set_parallel_search_mode(parallel_search);
  vector<MinimaxResult> lines = b_tmp.engine_recommendation(d, 3, prune);
  printf("%d,%d,%d,%d,%0.5f,%0.5f", d, parallel_search, parallel_eval, 
          b_tmp.md.eval_count, b_tmp.md.total_time, b_tmp.md.eval_time);
  for (int d = 0; d < max_depth; d++) {
    int searched = 0, pruned = 0;
    if (b_tmp.md.prune_count.count(d)) {
      auto &data = b_tmp.md.prune_count.at(d);
      searched = data.first;
      pruned = data.second;
    }
    printf(",%d,%d", searched, pruned);
  }
  printf("\n");
}

void search_position(string file_name, int max_depth, bool prune) {
  Engine_Board b(file_name);
  for (int d = 1; d <= max_depth; d++) {
    search_depth(b, d, max_depth, false, true, prune);
    search_depth(b, d, max_depth, false, false, prune);
    search_depth(b, d, max_depth, true, true, prune);
    search_depth(b, d, max_depth, true, false, prune);
  }
}

int main(int argc, char *argv[]) {
  bool prune = true;

  if (argc != 3) {
    printf("Usage: ./profile <board_file.txt> <max_depth>\n");
    return 0;
  }
  search_position(string(argv[1]), atoi(argv[2]), prune);

  return 0;
}