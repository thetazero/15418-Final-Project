#ifndef BOARD_H
#define BOARD_H

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>

using namespace std;

/* Board Details:
Boards are square, size x size
There are 2 players: x and o. Empty position is represented by a .
Each position at a board is either:
x - represented by 1 in our array
o - represented by -1 in our array
. - represented by 0 in our array

In board files, the format is:
- first line is the size
- second line is the who's turn
- followed by a space-separated grid of the board
E.g.
3
x
. x o
x o .
o o x
*/

class Board {
public:
  // initialize an empty board
  Board(int board_size);

  // initialize a board from a file
  Board(string filename);

  // Initialize board from board state string, turn char, and size int
  Board(string board_state, char board_turn, int board_size);

  // copy constructor
  Board(Board &b);

  int get_turn() { return turn; }
  // make move for whoever's turn is up on board[r,c]
  virtual int make_move(int r, int c);
  virtual int make_move(int i);
  virtual int undo_move(int i);

  // save the existing board to a file
  void save_board(string filename);

  // print the board to console
  void print();

  // get board size
  int get_size();

  ~Board();
  int turn;

protected:
  char *board;
  int size;

  inline int idx(int r, int c) { return r * size + c; }
  inline pair<int, int> rc(int i) { return make_pair(i / size, i % size); }
};
#endif
