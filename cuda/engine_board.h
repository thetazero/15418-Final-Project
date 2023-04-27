#ifndef ENGINE_BOARD_H
#define ENGINE_BOARD_H

#include "board.h"
#include <unordered_set>
#include <utility>
#include <vector>


using namespace std;

const int GAME_OVER_EVAL = 100000;
const int INEVITABLE_WIN_EVAL = 10000;

struct Piece {
  char r;
  char c;
  char player; // 1 or -1
  Piece(char pr, char pc, char p) : r(pr), c(pc), player(p) {}
};

struct TileSummary {
  int x;
  int o;
  bool operator==(const TileSummary &other) const {
    return x == other.x && o == other.o;
  }
  string to_string() {
    return "<" + std::to_string(x) + "," + std::to_string(o) + ">";
  }
  TileSummary(int x, int o) : x(x), o(o) {}
};

const int directions[][2] = {
    {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1},
};

// half the directions for checking in directions in parallel pairs
const int half_directions[][2] = {
    {1, 0}, {0, 1}, {1, 1}, {1, -1},
};

typedef struct MinimaxResult {
  int score;
  vector<pair<int, int>> moves; // store the variation
} MinimaxResult;

class Engine_Board : public Board {
public:
  Engine_Board(int board_size);
  Engine_Board(string filename);
  Engine_Board(string board_state, char turn, int board_size);
  Engine_Board(Engine_Board &b);
  Engine_Board();

  // returns the index of board positions for the moves
  // call make_move(idx) to make one of the moves
  vector<int> get_candidate_moves();

  // returns the evaluation of any given position
  int eval();

  // move at board position (r,c)
  int make_move(int r, int c);
  // move at board index i
  int make_move(int i);
  // undo move at board index i
  int undo_move(int i);
  int game_over();
  // recomend a move after searching for max_depth of depth
  std::vector<MinimaxResult> engine_recommendation(int depth, int num_lines, bool prune);
  std::vector<MinimaxResult> engine_recommendation_omp(int depth, int num_lines, bool prune);
  // max speed, only returns the best move
  int fast_engine_recommendation(int depth);
  // with omp
  int fast_engine_recommendation_omp(int depth);

  int eval_count = 0;

  bool in_bounds(int r, int c);
  // count the number of pieces in a direction from (r,c)
  // positive if counting x's and negative if counting o's
  int count_direction(int r, int c, int dr, int dc);
  // compute the max formed straight lines (x's and o's) by placing a piece at (r,c)
  TileSummary summarize_empty_tile(int r, int c);
  // give a score for an empty tile based of its summary
  int summary_score(TileSummary ts);

  // bounds of search, give a buffer of 1 row/col on each side where possible
  // e.g. if pieces are between (1,1)->(5,5), search bounds are (0,0)->(6,6)
  char r_min, c_min, r_max, c_max;

protected:

private:
  // track critical squares
  unordered_set<int> critical_4, critical_3;

  void update_bounds(int r, int c);
  void print_bounds();

  int game_over(int r, int c);
  // check for 5 in a row from (r,c) in rows, cols, diags
  // also check for live 4's and 3's
  void check_5_straight(int r, int c, int &x_4_count, int &o_4_count,
                        int &x_3_count, int &o_3_count);

  void check_direction(int r, int c, int dr, int dc, int &count, int &x_3,
                       int &o_3);
  void check_special_3(int r, int c, int &x_3_count, int &o_3_count);
  void check_special(int r, int c, int dr, int dc, int &x_3, int &o_3);

  MinimaxResult minimax(int max_depth, int depth, vector<MinimaxResult> &lines,
                        bool isMax, int alpha, int beta, bool prune);
  MinimaxResult minimax_omp(int max_depth, int depth, vector<MinimaxResult> &lines,
                        bool isMax, int alpha, int beta, bool prune);

  // fast implementation of minimax that only searches for the best move
  int fast_minimax(int max_depth, int depth, bool isMax, int alpha, int beta);
  // fast implementation of minimax that only searches for the best move, using openmp
  int fast_minimax_omp(int max_depth, int depth, bool isMax, int alpha, int beta);

  // fast_minimax sets this to the best move from the root
  volatile int fast_root_best_move;
};

#endif