#include <iostream>
#include <string>
#include "board.h"
#include "engine_board.h"
#include "timing.h"
#include <cassert>

using namespace std;
string file_name = "./boards/19x19.txt";

void test_empty_board() {
    Engine_Board b(19);
    b.print();
    for (int i = 0; i < 5; i++) {
        auto moves = b.get_candidate_moves();
        for (auto &m : moves) {
            cout << "(" << m / b.get_size() << "," << m % b.get_size() << ") ";
        }
        cout << endl;
        b.make_move(moves[0]);
        b.print();
    }
    b.save_board(file_name);
}

void load_board_and_move() {
    Board b(file_name);
    b.make_move(0, 2);
    b.print();
    b.make_move(0, 1);
    assert(b.make_move(0, 2) < 0);
    b.print();
    b.save_board(file_name);
}

void test_engine_board() {
    Engine_Board e(file_name);
    e.print();
    int eval = e.eval();
    e.make_move(0, 1);
    e.make_move(0, 2);
    cout << "Eval: " << eval << endl;
    e.print();
}
int main() {
    test_empty_board();
    // load_board_and_move();  
    // test_engine_board(); 
    return 0;
}