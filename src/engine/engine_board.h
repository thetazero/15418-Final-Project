#ifndef ENGINE_BOARD_H
#define ENGINE_BOARD_H

#include "board.h"
#include <map>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace std;

const int GAME_OVER_EVAL = 1000;
const int INEVITABLE_WIN_4_EVAL = 800;
const int INEVITABLE_WIN_3_EVAL = 600;
const int SEARCH_BOUND_PADDING = 2; // number of tiles outside of last chip to search

struct Search_Metadata {
  int eval_count;
  double eval_time, total_time, ispc_time;
  // map depth to (searched/pruned branches)
  map<int, pair<int, int>> prune_count; 

  void print() {
    cout << "---------------------- Search Statistics -------------------------" << endl;
    cout << "Total Time: " << total_time << " Eval Time: " << eval_time
         << " Eval Count: " << eval_count << " ISPC Time: " << ispc_time << endl;
    for (auto i = prune_count.begin(); i != prune_count.end(); i++) {
      cout << "Depth: " << i->first << " Searched/Pruned: " << i->second.first
           << "/" << i->second.second << endl;
    }
    cout << "------------------------------------------------------------------" << endl;
  }
};

typedef struct MinimaxResult {
  int score;
  vector<pair<int, int>> moves; // store the variation
} MinimaxResult;

class Engine_Board : public Board {
public:
  Engine_Board(int board_size);
  Engine_Board(string filename);
  Engine_Board(Engine_Board &b);

  ~Engine_Board();

  // returns the index of board positions for the moves
  // call make_move(idx) to make one of the moves
  void get_candidate_moves(vector<int> &moves);

  // returns the evaluation of any given position
  int eval();
  void set_parallel_eval_mode(bool parallel);
  void set_parallel_search_mode(bool parallel);
  void set_fast_mode(bool fast);
  
  // move at board position (r,c)
  int make_move(int r, int c);
  // move at board index i
  int make_move(int i);
  // undo move at board index i
  int undo_move(int i);
  int game_over();
  // recomend a move after searching for max_depth of depth
  vector<MinimaxResult> engine_recommendation(int depth, int num_lines,
                                              bool prune);

  Search_Metadata md;

protected:
  // bounds of search, give a buffer of 1 row/col on each side where possible
  // e.g. if pieces are between (1,1)->(5,5), search bounds are (0,0)->(6,6)
  int r_min, c_min, r_max, c_max;
  bool parallel_eval, parallel_search, fast_mode;

private:
  // track critical squares
  // unordered_set<int> critical_4, critical_3;
  int *critical_4, *critical_3;

  void update_bounds(int r, int c);
  void print_bounds();

  int game_over(int r, int c);
  // check for 5 in a row from (r,c) in rows, cols, diags
  // also check for live 4's and 3's
  int check_5_straight(int r, int c, int &x_4_count, int &o_4_count);

  // returns true if the edge is empty, false otherwise
  // empty edge means the last tile that stops a consecutive sequence of
  // the same player's chip is empty and not the opposite player's chip
  bool check_direction(int r, int c, int dr, int dc, int &count);

  int sequential_eval(int &x_4_count, int &o_4_count);
  int ispc_eval(int &x_4_count, int &o_4_count);
  int process_counts(int *counts);
  
  MinimaxResult minimax(int max_depth, int depth, vector<MinimaxResult> &lines,
                        bool isMax, int alpha, int beta, bool prune);
  
  MinimaxResult fast_engine_recommendation(int depth);
  // fast implementation of minimax that only searches for the best move
  int fast_minimax(int max_depth, int depth, bool isMax, int alpha, int beta);
  // fast implementation of minimax that only searches for the best move, using openmp
  int fast_minimax_omp(int max_depth, int depth, bool isMax, int alpha, int beta);
  // fast_minimax sets this to the best move from the root
  volatile int fast_root_best_move;
};

#endif