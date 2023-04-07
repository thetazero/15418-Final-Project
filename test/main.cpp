#include "engine_board.h"
#include <gtest/gtest.h>

TEST(GameOver, EmptyBoard) {
  Engine_Board empty(19);
  EXPECT_EQ(empty.game_over(), 0);

  Engine_Board empty_tiny(1);
  EXPECT_EQ(empty_tiny.game_over(), 0);
}
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}