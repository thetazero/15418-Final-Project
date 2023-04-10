#include "engine_board.h"
#include <gtest/gtest.h>
#include <algorithm>

TEST(ENGINE_BOARD, game_over) {
  Engine_Board empty(19);
  EXPECT_EQ(empty.game_over(), 0);

  Engine_Board empty_tiny(1);
  EXPECT_EQ(empty_tiny.game_over(), 0);
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
  EXPECT_EQ(board1.summarize_empty_tile(5, 3), TileSummary(2, 2));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}