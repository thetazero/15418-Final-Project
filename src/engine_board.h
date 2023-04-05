#ifndef ENGINE_BOARD_H
#define ENGINE_BOARD_H

#include "board.h"
#include <vector>

using namespace std;

const int GAME_OVER_EVAL = 10000;
const int INEVITABLE_WIN_EVAL = 1000;

struct Piece {
    char r;
    char c;
    char player; // 1 or -1
    Piece(char pr, char pc, char p): r(pr), c(pc), player(p) {} 
};

typedef struct MinimaxResult {
  int score;
  int move;
} MinimaxResult;

class Engine_Board : public Board {
public:
    Engine_Board(int board_size);
    Engine_Board(string filename);
    
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
    // recomend a move
    MinimaxResult engine_recomendation(bool use_alpha_beta);
    // return if the game is over
    bool game_over();
    int empty_eval();
    int eval_count = 0;

protected:
    // bounds of search, give a buffer of 1 row/col on each side where possible
    // e.g. if pieces are between (1,1)->(5,5), search bounds are (0,0)->(6,6)
    char r_min, c_min, r_max, c_max; 

private:
    void update_bounds(int r, int c);
    void print_bounds();

    // check for 5 in a row from (r,c) in rows, cols, diags
    // also check for live 4's and 3's
    int check_5_straight(int r, int c, 
                         int &x_4_count, int &o_4_count, 
                         int &x_3_count, int &o_3_count);
    

    MinimaxResult minimax(int depth, bool isMax);
    MinimaxResult minimax_alpha_beta(int depth, bool isMax, int alpha, int beta);

    int line_len(int r, int c, int dr, int dc);
    int tile_value(int r, int c);
};

#endif