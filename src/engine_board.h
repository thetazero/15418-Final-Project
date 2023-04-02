#ifndef ENGINE_BOARD_H
#define ENGINE_BOARD_H

#include "board.h"
using namespace std;

const int GAME_OVER_EVAL = 10000;
const int INEVITABLE_WIN_EVAL = 1000;

struct Piece {
    char r;
    char c;
    char player; // 1 or -1
    Piece(char pr, char pc, char p): r(pr), c(pc), player(p) {} 
};

class Engine_Board : public Board {
public:
    Engine_Board(int board_size);
    Engine_Board(string filename);
    
    void get_candidate_moves();
    
    // returns the evaluation of any given position
    int eval();

    // updates the list of moves
    int make_move(int r, int c);

protected:
    // bounds of search, give a buffer of 1 row/col on each side where possible
    // e.g. if pieces are between (1,1)->(5,5), search bounds are (0,0)->(6,6)
    char r_min, c_min, r_max, c_max; 

private:
    void update_bounds(int r, int c);
    void print_bounds();

    // check for 5 in a row from (r,c) in rows, cols, diags
    // also ch
    int check_5_straight(int r, int c, 
                         int &x_4_count, int &o_4_count, 
                         int &x_3_count, int &o_3_count);
};

#endif