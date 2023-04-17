#include "engine_board.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <filesystem>


TEST(ENGINE_BOARD, game_over) {
  Engine_Board empty(19);
  EXPECT_EQ(empty.game_over(), 0);

  Engine_Board empty_tiny(1);
  EXPECT_EQ(empty_tiny.game_over(), 0);

  Engine_Board x_won(R"(
x o o o o 
o x . x .
o . x . .
o x . x .
o x . o x
  )", 'x', 5);
  EXPECT_EQ(x_won.game_over(), GAME_OVER_EVAL);

  Engine_Board o_won(R"(
x o . . . 
x o . . .
. o x . .
. o . x .
. o . . x
  )", 'o', 5);
  EXPECT_EQ(o_won.game_over(), -GAME_OVER_EVAL);
}

TEST(ENGINE_BOARD, LoadBoard) {
  string simple_board = R"(xx..
                           oo..
                           ....
                           ....)";
  Engine_Board simple(simple_board, 'x', 4);

  EXPECT_EQ(simple.game_over(), 0);
  EXPECT_EQ(simple.turn, 1);
  string simple_str = simple.to_string();
  string simple_str_expected = R"(Next Up: x
x x . .
o o . .
. . . .
. . . .)";
  EXPECT_EQ(simple_str, simple_str_expected);
}

TEST(ENGINE_BOARD, get_candidate_moves) {
  Engine_Board small(R"(...
                        .x.
                        ...)", 'o', 3);
  vector<int> moves = small.get_candidate_moves();
  EXPECT_EQ(moves.size(), 8);
  std::sort(moves.begin(), moves.end());
  vector<int> expected_moves = {0, 1, 2, 3, 5, 6, 7, 8};
  for (int i = 0; i < moves.size(); i++) {
    EXPECT_EQ(moves[i], expected_moves[i]);
  }
}

TEST(ENGINE_BOARD, in_bounds){
  Engine_Board small(3);
  EXPECT_EQ(small.in_bounds(0, 0), true);
  EXPECT_EQ(small.in_bounds(0, 2), true);
  EXPECT_EQ(small.in_bounds(2, 0), true);
  EXPECT_EQ(small.in_bounds(1, 2), true);
  EXPECT_EQ(small.in_bounds(2, 2), true);
  EXPECT_EQ(small.in_bounds(-1, 3), false);
  EXPECT_EQ(small.in_bounds(0, 4), false);
  EXPECT_EQ(small.in_bounds(3, -1), false);
  EXPECT_EQ(small.in_bounds(-1, 3), false);
  EXPECT_EQ(small.in_bounds(3, 4), false);

}

TEST(ENGINE_BOARD, count_direction){
  string b1 = R"(
  .......
  ..xxx..
  ..xoo.x
  ..oxo..
  ....xo.
  .......
  .......
  )";
  Engine_Board board1(b1, 'x', 7);
  EXPECT_EQ(board1.count_direction(1, 1, 0, 1), 3);
  EXPECT_EQ(board1.count_direction(0, 1, 0, 1), 0);
  EXPECT_EQ(board1.count_direction(2, 5, 0, -1), -2);
  EXPECT_EQ(board1.count_direction(3, 1, 0, 1), -1);
  EXPECT_EQ(board1.count_direction(0, 2, 1, 0), 2);
  EXPECT_EQ(board1.count_direction(0, 3, 1, 0), 1);
  EXPECT_EQ(board1.count_direction(3, 5, -1, -1), -1);

  string b2 = R"(
  .o.....
  ..xxx..
  ..x.o.x
  ..oxo..
  ..x.xo.
  ..o..o.
  ..xxx..
  )";
  Engine_Board board2(b2, 'x', 7);
  EXPECT_EQ(board2.count_direction(0, 0, 0, 1), -1);
  EXPECT_EQ(board2.count_direction(0, 0, 0, -1), 0);
  EXPECT_EQ(board2.count_direction(0, 0, 1, 0), 0);
  EXPECT_EQ(board2.count_direction(0, 0, 1, 1), 0);
  EXPECT_EQ(board2.count_direction(0, 0, -1, 0), 0);
  EXPECT_EQ(board2.count_direction(0, 0, 1, -1), 0);
  EXPECT_EQ(board2.count_direction(0, 0, -1, -1), 0);
  EXPECT_EQ(board2.count_direction(0, 0, -1, 1), 0);
}

TEST(ENGINE_BOARD, sumarize_empty_tile){
  string b1 = R"(
  .o.....
  ..xxx..
  ..x.o.x
  ..oxo..
  ..x.xo.
  .oo..x.
  .oxxx..
  )";
  Engine_Board board1(b1, 'x', 7);
  EXPECT_EQ(board1.summarize_empty_tile(0, 0), TileSummary(0, 1));
  EXPECT_EQ(board1.summarize_empty_tile(1, 0), TileSummary(0, 1));
  EXPECT_EQ(board1.summarize_empty_tile(2, 0), TileSummary(0, 0));
  EXPECT_EQ(board1.summarize_empty_tile(1, 1), TileSummary(4, 1));
  EXPECT_EQ(board1.summarize_empty_tile(4, 3), TileSummary(2, 3));
  EXPECT_EQ(board1.summarize_empty_tile(3, 1), TileSummary(2, 1));
  EXPECT_EQ(board1.summarize_empty_tile(3, 5), TileSummary(2, 1));
  EXPECT_EQ(board1.summarize_empty_tile(5, 3), TileSummary(2, 2));
  EXPECT_EQ(board1.summarize_empty_tile(2, 3), TileSummary(2, 2));
  EXPECT_EQ(board1.summarize_empty_tile(0, 6), TileSummary(0, 0));
  EXPECT_EQ(board1.summarize_empty_tile(6, 0), TileSummary(0, 1));
  EXPECT_EQ(board1.summarize_empty_tile(6, 6), TileSummary(4, 0));
}

TEST(ENGINE_BOARD, e_function){
  string b1 = R"(
. . . 
. x .
. . .
  )";
  Engine_Board board1(b1, 'o', 3);
  EXPECT_EQ(board1.eval(), 8);

  string b2 = R"(
x o x
o . o
x o x
  )";
  Engine_Board board2(b2, 'x', 3);
  EXPECT_EQ(board2.eval(), 0);

  string b3 = R"(
x o x o
o x o x
x x . o 
o o x x)";
  Engine_Board board3(b3, 'x', 3);
  EXPECT_EQ(board3.eval(), 0);
}

TEST(ENGINE_BOARD, bounds){
  string easy_win = R"(
x o . . . 
x o . . .
x o . . .
x o . . .
. . . . .)";
  Engine_Board board(easy_win, 'x', 5);
  EXPECT_EQ(board.r_min, 0);
  EXPECT_EQ(board.r_max, 4);
  EXPECT_EQ(board.c_min, 0);
  EXPECT_EQ(board.c_max, 2);
}

TEST(ENGINE_BOARD, size){
  string easy_win = R"(
x o . . . 
x o . . .
x o . . .
x o . . .
. . . . .)";
  Engine_Board board(easy_win, 'x', 5);
  EXPECT_EQ(board.size, 5);

}

TEST(ENGINE_BOARD, engine_recommendation) {
  string easy_win = R"(
x o . . . 
x o . . .
x o . . .
x o . . .
. . . . .)";
  Engine_Board board(easy_win, 'x', 5);
  MinimaxResult result = board.engine_recommendation(1, 1, true)[0];
  pair<int,int> expected_move = make_pair(4, 0);
  EXPECT_EQ(result.moves[0], expected_move);

  string easy_win2 = R"(
. . . . . .
. . . . . .
. o x o . .
. o x . . .
. . x . . .
. . . . . .)";
  Engine_Board board2(easy_win2, 'x', 6);
  MinimaxResult result2 = board2.engine_recommendation(3, 1, true)[0];
  vector<pair<int,int>> expected_moves = {
    make_pair(1, 2),
    make_pair(5, 2),
  };
  EXPECT_TRUE(
    std::find(expected_moves.begin(), expected_moves.end(), result2.moves[0]) != expected_moves.end()
  ) << "Expected (1, 2) or (5, 2), got (" << result2.moves.back().first << ", " << result2.moves.front().second << ")";

  string easy_win3 = R"(
x o . . .
x o . . .
x o . . .
x o x . . 
. . . . .)";
  Engine_Board board3(easy_win3, 'x', 5);
  MinimaxResult result3 = board3.engine_recommendation(1, 1, true)[0];
  pair<int,int> expected_move3 = make_pair(4, 1);
  EXPECT_EQ(result3.moves[0], expected_move3);
}

TEST(ENGINE_BOARD, prunning_consistency) {
  string path = "boards/";
  for (const auto & board_file : std::filesystem::directory_iterator(path)) {
    Engine_Board board(board_file.path());
    EXPECT_EQ(board.engine_recommendation(3, 1, true)[0].moves[0], board.engine_recommendation(3, 1, false)[0].moves[0]);
  }
}

TEST(ENGINE_BOARD, engine_recommendation_with_omp) {
  string path = "boards/";
  for (const auto & board_file : std::filesystem::directory_iterator(path)) {
    Engine_Board board(board_file.path());
    int move = board.engine_recommendation(3, 1, false)[0].score;
    int omp_move = board.engine_recommendation_omp(3, 1, false)[0].score;
    EXPECT_EQ(move, omp_move) << "Board: " << board_file.path() << 
    " gave a different score with and without OpenMP." << 
    " Expected: " << move << " Got: " << omp_move << endl;
  }
}

TEST(ENGINE_BOARD, engine_recommendation_with_omp_pruning) {
  string path = "boards/";
  for (const auto & board_file : std::filesystem::directory_iterator(path)) {
    Engine_Board board(board_file.path());
    int move = board.engine_recommendation(3, 1, true)[0].score;
    int omp_move = board.engine_recommendation_omp(3, 1, true)[0].score;
    EXPECT_EQ(move, omp_move) << "Board: " << board_file.path() << 
    " gave a different score with and without OpenMP." << 
    " Expected: " << move << " Got: " << omp_move << endl;
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}