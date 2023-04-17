#include <cassert>
#include <iostream>
#include <limits>
#include <string>

#include "board.h"
#include "engine_board.h"
#include "timing.h"
#include <filesystem>
#include <omp.h>

using namespace std;
string file_name = "./boards/19x19.txt";

void test_empty_board() {
  Engine_Board b(5);
  b.print();
  for (int i = 0; i < 25; i++) {
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
  // e.save_board(file_name);
  cout << "Eval: " << eval << endl;
  // e.print();
}

// void rng_vs_minimax() {
//   const int size = 19;
//   Engine_Board b(size);
//   b.print();
//   for (int i = 0; i < size * size; i++) {
//     auto moves = b.get_candidate_moves();
//     // for (auto &m : moves) {
//     //   cout << "(" << m / b.get_size() << "," << m % b.get_size() << ") ";
//     // }
//     // cout << endl;
//     if (i % 2 == 0) {
//       MinimaxResult best_move = b.engine_recommendation(3, true);
//       int r = best_move.move / b.get_size();
//       int c = best_move.move % b.get_size();
//       cout << "Best move: (" << r << "," << c << ")" << endl;
//       cout << "Best score: " << best_move.score << endl;
//       cout << "Eval at current position: " << b.eval() << endl;
//       b.make_move(best_move.move);
      
//     } else {
//       b.make_move(moves[0]);
//     }
//     b.print();
//     int eval = b.eval();
//     if (eval == 10000 || eval == -10000) {
//       cout << "Number of evals: " << b.eval_count << endl;
//       cout << "Game over!\n";
//       break;
//     }
//   }
//   b.save_board(file_name);
// }

// void search_position() {
//   Engine_Board b(file_name);
//   cout << "Current Eval: " << b.eval() << endl;
//   auto moves = b.get_candidate_moves();
//   for (auto &m : moves) {
//     cout << "(" << m / b.get_size() << "," << m % b.get_size() << ") ";
//   }
//   cout << endl;
//   for (int d = 1; d <= 7; d++) {
//     Engine_Board b_tmp(b);
//     MinimaxResult best_move = b.engine_recommendation(d, true);
//     cout << "Depth: " << d << endl;
//     for (auto &move : best_move.moves) {
//       int r = move.first;
//       int c = move.second;
//       cout << "(" << r << "," << c << ")" << endl;
//       b_tmp.make_move(r, c);
//       // b_tmp.print();
//     }
//     // cout << endl;
//     cout << "Best score: " << best_move.score << endl;
//     cout << "Eval at current position: " << b.eval() << endl; 
//     cout << endl;
//     int r = best_move.moves[0].first, c = best_move.moves[0].second; 
    
//   }
    
//   // b.make_move(r, c);
//   // b.print();
// }

void omp_vs_normal_benchmark(const int depth, const bool prune){
  vector<Engine_Board> boards;
  size_t i = 0;
  float normal_time_total = 0, omp_time_total = 0;
  for (const auto & board_file : std::filesystem::directory_iterator("../test/boards/")) {
    cout << "Board " << board_file.path() << endl;
    Engine_Board b(board_file.path());
    Timer t_omp;
    b.engine_recommendation_omp(depth, 1, prune);
    float omp_time = t_omp.elapsed();
    Timer t_normal;
    b.engine_recommendation(depth, 1, prune);
    float normal_time = t_normal.elapsed();

    omp_time_total += omp_time;
    normal_time_total += normal_time;

    cout << "Normal time: " << normal_time << endl;
    cout << "Omp time: " << omp_time << endl;
  }
  cout << "Total time" << endl;
  cout << "Normal time: " << normal_time_total << endl;
  cout << "Omp time: " << omp_time_total << endl;
}

int main() {
  // test_empty_board();
  // load_board_and_move();
  // test_engine_board();
  // rng_vs_minimax();
  // search_position();
  omp_vs_normal_benchmark(4, true);
  return 0;
}