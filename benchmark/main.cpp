#include "engine_board.h"
#include "timing.h"

#include <algorithm>
#include <argp.h>
#include <filesystem>
#include <functional>
#include <iostream>
#include <set>

using namespace std;

// benchmark two engine recommendation functions at a given depth
void benchmark(const int depth,
               function<void(Engine_Board &, int)> engine_rec) {
  vector<Engine_Board> boards;
  size_t i = 0;
  float total_time = 0;
  
  set<filesystem::path> board_files;
  for (const auto &board_file : filesystem::directory_iterator("../boards/")) {
    board_files.insert(board_file.path());
  }
  for (const auto &board_file : board_files) {
    cout << board_file << ", ";
    Engine_Board b(board_file);

    Timer timer;
    engine_rec(b, depth);
    float time = timer.elapsed();

    total_time += time;

    cout << time << endl;
  }
  cout << "Total time: " << total_time << endl;
}

pair<int, int> rm(int i, int size) { return make_pair(i / size, i % size); }

void engine_recommendation_prune(Engine_Board &b, int depth) {
  b.fast_engine_recommendation(depth);
}

void engine_recommendation_no_prune(Engine_Board &b, int depth) {
  b.engine_recommendation(depth, 1, false);
}

void omp_engine_recommendation(Engine_Board &b, int depth) {
  b.fast_engine_recommendation_omp(depth);
}

const char *argp_program_version = "Connect Five Benchmark 1.0";
const char *argp_program_bug_address = "<aseletsk@andrew.cmu.edu>";
static char doc[] = "Your program description.";
static char args_doc[] = "[FILENAME]...";
static struct argp_option options[] = {
    {"engine", 'e', "default", 0,
     "Set engine being bechmarked. (default, no_prune, omp, ispc, omp_ispc)"},
    {"depth", 'd', "3", 0, "Set depth of minimax search."},
    {0}};

enum engine_t { DEFAULT, OMP, NO_PRUNE };
struct arguments {
  engine_t engine;
  int depth;
};

static error_t parse_opt(int key, char *arg, struct argp_state *state) {
  struct arguments *arguments = static_cast<struct arguments *>(state->input);

  if (key == 'e') {
    if (strcmp(arg, "omp") == 0) {
      arguments->engine = OMP;
    } else if (strcmp(arg, "no_prune") == 0) {
      arguments->engine = NO_PRUNE;
    } else {
      arguments->engine = DEFAULT;
    }
  } else if (key == 'd') {
    arguments->depth = atoi(arg);
  } else {
    return ARGP_ERR_UNKNOWN;
  }
  return 0;
}

static struct argp argp = {options, parse_opt, args_doc, doc};

int main(int argc, char **argv) {
  struct arguments arguments;
  arguments.engine = DEFAULT;
  arguments.depth = 3;
  argp_parse(&argp, argc, argv, 0, 0, &arguments);


  function<void(Engine_Board &, int)> engine_rec;

  if (arguments.engine == DEFAULT) {
    engine_rec = engine_recommendation_prune;
    cout << "Default engine" << endl;
  } else if (arguments.engine == OMP) {
    engine_rec = omp_engine_recommendation;
    cout << "OMP engine" << endl;
  } else if (arguments.engine == NO_PRUNE) {
    engine_rec = engine_recommendation_no_prune;
    cout << "No prune default engine" << endl;
  } else {
    cout << "Invalid engine" << endl;
    return 1;
  }
  cout << "Depth: " << arguments.depth << endl;

  benchmark(arguments.depth, engine_rec);
}