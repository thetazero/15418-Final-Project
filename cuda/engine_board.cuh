#ifndef ENGINE_BOARD_H
#define ENGINE_BOARD_H

#include "board.cuh"
#include "eval.cuh"
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

  // move at board position (r,c)
  int make_move(int r, int c);
  // move at board index i
  int make_move(int i);
  // undo move at board index i
  int undo_move(int i);

  int game_over();
  int cuda_recommendation(int depth);
  bool in_bounds(int r, int c);

  // bounds of search, give a buffer of 1 row/col on each side where possible
  // e.g. if pieces are between (1,1)->(5,5), search bounds are (0,0)->(6,6)
  char r_min, c_min, r_max, c_max;

protected:

private:
  void update_bounds(int r, int c);
  void print_bounds();

  int game_over(int r, int c);
  // check for 5 in a row from (r,c) in rows, cols, diags
  int cuda_minimax(int max_depth, int depth, bool isMax, int *evals, int *i);
  void cuda_minimax_stage(int max_depth, int depth, bool isMax, char *boards, int *i);

  // fast_minimax sets this to the best move from the root
  volatile int fast_root_best_move;
};

#endif
