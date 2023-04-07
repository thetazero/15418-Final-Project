---
layout: page
title: Evaluation Function
permalink: /eval_function/
---

# Empty tiles matter, not the number of connected tiles

While an obvious aproach is to count the numbers of 4 in a row, 3 in a row, 2, in a row, etc...
This does not work for various reasons, for example the following board has 4 in a row, but it doesn't matter because the tiles are no longer live.
![4 in a row doesn't matter](/images/4_in_a_row_doesnt_matter.png){:class="img-responsive"}
Another issue is that this may lead to double counting
![4 in a row double counted](/images/4_in_a_row_double_count.png){:class="img-responsive"}

While there are 2 distinct 4 in a rows, in effect there is only one, as completing one will complete the other.

Thus we instead focus primarily on the value of empty tiles.

# Evaluating empty tiles

From each empty tile there are 8 potential directions to evaluate in (4 cardinal directions, and 4 diagonal directions).
![8 directions](/images/potential_eval_directions.png){:class="img-responsive"}

However, this reduces to far fewer cases with respect to evaluation.
The board below seems complicated, but the only tiles that actually matter are marked in red.
All the gray tiles are irrelevant, as there is a bigger line that will be completed if the center tile is placed.
![reduction in cases](/images/reduction_in_cases.png){:class="img-responsive"}

This means we can evaluate each empty treat an empty tile as:
The max number of connected x's and the max number of connected o's.
So our previous picture's center tile is (2,2).
And the picture below's center tile is a (4,2).

![(4,2) board](/images/4_2_board.png){:class="img-responsive"}

We then create a custom function
$$ e(x,o) = ???$$
which evaluates an empty tile given the max number of connected x's and o's.

We then sum over the values of all the empty tiles to get the final evaluation function.

# Special cases

However, we do include some special cases, to account for clearly winning/losing positions.

## Double 4's
Two double fours connected to distinct empty tiles are a win as seen below.
No matter what O plays, X will win.
![double 4's](/images/double_4_win.png){:class="img-responsive"}

