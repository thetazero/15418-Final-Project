#ifndef ENGINE_BOARD_H
#define ENGINE_BOARD_H

#include "board.h"
#include <vector>
#include <unordered_set>

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
    Engine_Board(Engine_Board &b);
    
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
    MinimaxResult engine_recommendation(int depth, bool prune);
    
    int eval_count = 0;

protected:
    // bounds of search, give a buffer of 1 row/col on each side where possible
    // e.g. if pieces are between (1,1)->(5,5), search bounds are (0,0)->(6,6)
    char r_min, c_min, r_max, c_max; 

private:
    // track critical squares
    unordered_set<int> critical_4, critical_3;

    void update_bounds(int r, int c);
    void print_bounds();
    
    int game_over(int r, int c);
    // check for 5 in a row from (r,c) in rows, cols, diags
    // also check for live 4's and 3's
    void check_5_straight(int r, int c, 
                         int &x_4_count, int &o_4_count, 
                         int &x_3_count, int &o_3_count);

    void check_direction(int r, int c, int dr, int dc, 
                         int &count, int &x_3, int &o_3); 
    void check_special_3(int r, int c, int &x_3_count, int &o_3_count);
    void check_special(int r, int c, int dr, int dc, int &x_3, int &o_3);   

    MinimaxResult minimax(int max_depth, int depth, bool isMax);
    MinimaxResult minimax_alpha_beta(int max_depth, int depth, bool isMax, int alpha, int beta);
};

#endif