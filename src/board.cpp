#include "board.h"

Board::Board() : Board(19) {}

Board::Board(int board_size = 19) {
  size = board_size;
  turn = 1;
  int total = size * size;
  board = new char[total];
  memset(board, 0, total);
}

// initialize a board from a file
Board::Board(string filename) {
  fstream board_file;
  board_file.open(filename, fstream::in);
  if (!board_file.is_open()) {
    cout << "Failed to open file: " << filename << endl;
  }

  string line;

  // get board size
  getline(board_file, line);
  size = stoi(line);

  // get who's turn
  getline(board_file, line);
  if (line[0] == 'x')
    turn = 1;
  else
    turn = -1;

  int total = size * size;
  board = new char[total];
  int i = 0;
  while (getline(board_file, line)) {
    if (line.size() != (2 * size - 1)) {
      return;
    }

    for (int c = 0; c < 2 * size; c += 2) {
      int n;
      if (line[c] == 'x')
        n = 1;
      else if (line[c] == 'o')
        n = -1;
      else if (line[c] == '.')
        n = 0;
      else {
        cout << "invalid piece marker: " << line[c] << endl;
        return;
      }
      board[i + c / 2] = n;
    }

    i += size;
    if (i > total) {
      cout << "Too many squares\n";
      return;
    }
  }

  board_file.close();
}

Board::Board(string board_state, char board_turn, int board_size) {
  turn = board_turn == 'x' ? 1 : -1;
  size = board_size;
  int total = board_size * board_size;
  board = new char[total];
  int count = 0;
  for (int i = 0; i < board_state.length(); i++) {
    if (board_state[i] == 'x')
      board[count] = 1;
    else if (board_state[i] == 'o')
      board[count] = -1;
    else if (board_state[i] == '.')
      board[count] = 0;
    else if (board_state[i] == ' ' || board_state[i] == '\n') {
      continue;
    } else {
      cout << "invalid piece marker: " << board_state[i] << endl;
      return;
    }
    count += 1;
  }
}

Board::Board(Board &b) {
  size = b.size;
  turn = b.turn;
  int total = size * size;
  board = new char[total];
  memcpy(board, b.board, total);
}

// make move n on board[r,c]
int Board::make_move(int r, int c) { return make_move(idx(r, c)); }

int Board::make_move(int i) {
  if (i < 0 || i >= size * size) {
    return -1;
  }
  if (board[i] != 0) {
    return -1;
  }

  board[i] = turn;
  turn *= -1;
  return 0;
}

int Board::undo_move(int i) {
  if (i < 0 || i >= size * size) {
    return -1;
  }
  if (board[i] == 0) {
    return -1;
  }

  board[i] = 0;
  turn *= -1;
  return 0;
}

// save the existing board to a file
void Board::save_board(string filename) {
  fstream board_file;
  board_file.open(filename, fstream::out | fstream::trunc);
  if (!board_file.is_open()) {
    cout << "Failed to open file: " << filename << endl;
  }

  board_file << size << "\n";
  if (turn == 1)
    board_file << "x\n";
  else
    board_file << "o\n";

  for (int r = 0; r < size; r++) {
    for (int c = 0; c < size; c++) {
      int n = board[idx(r, c)];
      if (n == 1)
        board_file << "x";
      else if (n == -1)
        board_file << "o";
      else
        board_file << ".";

      if (c == size - 1)
        board_file << "\n";
      else
        board_file << " ";
    }
  }

  board_file.close();
}

string Board::to_string() {
  string str = "";
  if (turn == 1)
    str += "Next Up: x\n";
  else
    str += "Next Up: o\n";
  for (int r = 0; r < size; r++) {
    for (int c = 0; c < size; c++) {
      int n = board[idx(r, c)];
      if (n == 1)
        str += "x";
      else if (n == -1)
        str += "o";
      else
        str += ".";

      if (c == size - 1){
        if (r != size - 1) str += "\n";
      }
      else
        str += " ";
    }
  }
  return str;
}

// print the board to console
void Board::print() {
  cout << to_string() << endl;
}

// get board size
int Board::get_size() { return size; }

Board::~Board() { delete[] board; }