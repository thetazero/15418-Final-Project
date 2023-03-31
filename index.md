---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
title: Home
---

# Title: Parallelizing the Game Tree Search with the Minimax Algorithm for Connect 5

Aleksei Seletskiy, William Wang

# Summary
We are going to try to implement a parallelized engine that plays Connect 5. We will attempt to use task-based parallelism to implement both standard minimax as well as minimax with pruning, as well as potentially use data-parallelism to implement our evaluation function.

# Background
Minimax is a classic game-playing algorithm for zero-sum games that attempts to find the best move in any position by calculating several moves, or plies, ahead using a search tree. Fundamentally, we could search a breadth of M moves in each position to a depth of D plies. When we reach our maximum search depth, we use an evaluation function to evaluate the position, and then let the evaluations propagate backwards to find the best move. If we incorporate pruning, we can eliminate certain branches of the search that are guaranteed to be irrelevant, which reduces the overall breadth of the search.
The tree-searching nature of the problem is conducive to parallelism. At the most basic level, we can spawn different threads to handle different moves in our search, so that we can search several moves in parallel. In addition, because the nature of connect 5 requires us to check all rows, columns, and diagonals for consecutive pieces, and a game tree search leads to the need to evaluate an exponentially high number of positions (O(M^D)), incorporating a data-parallel evaluation function that runs on GPU’s could prove highly useful, so that we can evaluate positions in parallel.

# Challenge
One challenge that we will face is when we implement pruning. First of all, pruning is a sequential process, so when we incorporate parallelism, we will not get the maximum benefit from pruning, which is fine as we still should gain more from parallel search. However, a challenge is to find a way perhaps to maximize the amount of pruning we can get by distributing our workload in a way where each thread gets to prune within its own subset of the tree, so the overall search is faster. 
Another challenge we may face is the workload imbalances. Whether it is due to pruning or that we reach a terminal position early in the depth of the tree, some branches of search may be shorter than others, which leads to divergent execution. Our challenge here will be to find the best way to distribute workload, whether it be through dynamic scheduling or some other mechanism.
Lastly, finding an effective mechanism and mapping of our data to CUDA threads and blocks for parallel game board evaluation could be a challenge as well. There are many ways to divide the workload, and figuring out how to effectively assign all of the positions from all of the branches of our search tree to the GPU, and how much load our GPU can take on is an interesting problem. Furthermore, if some branches are shorter than others, we may have to figure out the timing of our parallel evaluation. Do we evaluate each board the moment it becomes ready? Or do we wait until all searches have reached the leaf node, and then evaluate all of those boards at once. The former requires less synchronization, and thus no task will be left waiting too long, but it remains to be seen whether it maximizes the parallelism of the GPU sufficiently, and whether the overhead from moving data to the GPU overpowers the computation. On the other hand, the latter causes tasks to potentially have to wait, but it may leverage the parallelism offered by GPU’s better.

# Resources
We will use computers that support high amounts of task-level parallelism (either via MPI nodes or threads for openMP) and have a GPU, such as the ones we have previously used for labs. 
Our code base will start from a basic implementation of Minimax search. (E.g. [geeks for geeks minimax alg](https://www.geeksforgeeks.org/minimax-algorithm-in-game-theory-set-4-alpha-beta-pruning/))
Potential Evaluation Algorithms:
[New Heuristic Algorithm to improve the Minimax for Gomoku Artificial Intelligence](https://dr.lib.iastate.edu/server/api/core/bitstreams/39a805d5-8f5b-41e6-b07c-19c07229f813/content)

TODO: Find other resources about parallelizing minimax or evaluating connect 5 board

# Goals and Deliverables:
Plan to Achieve: 
 - Parallelize Minimax without pruning using OpenMP: hopefully given N threads we can achieve near N-times speedup
 - Parallelize Minimax with pruning using OpenMP: may not result in N times speedup compared to sequential pruning, but hopefully runs faster than parallel minimax without pruning
 - Create a Baseline Evaluation Function: implement an evaluation algorithm sequentially that has potential for parallelism. The algorithm doesn’t have to be super strong in terms of playing strength for starters, but it should make sense and make the correct moves in obvious positions
 - Parallelize Game Board Evaluation with CUDA: game board evaluation is purely functional, so with such few dependencies, hopefully there is a good speedup of just evaluating a large set of positions in isolation
 - Incorporate parallel game board evaluation with parallel game tree search: Hard to say exactly what the speedup magnitude would be as that depends on GPU specs and implementation details, but this should run faster than parallel game tree search without parallel board evaluation
Hope to Achieve:
 - Try the above using MPI
 - Improve our evaluation function to make our engine stronger
 - Figure out heuristics that can improve our workload distribution for the game tree and position evaluation (e.g. for workload distribution, how can we assign the moves to different tasks so that we maximize pruning?)
 - Make game-specific optimizations to our algorithm (e.g. certain positions have forced moves, so no need to search too much, or finding a heuristic to figure out which moves to search first and how many total moves we need to search)

# Demo Content
We can run our different versions of our engine with the various optimizations and compare it to the sequential versions. We can also present speedup graphs for various breadths and plies. Lastly, we hope that we can maybe have people play against the various forms of the engine, where the engine has a fixed time to make its move. Or we can even have the various forms of our engine play each other in a timed game. That way, we can see if a faster engine that searches more moves outperforms a slower engine.

# Platform Choice

TODO

# Schedule

## 4/1 - 4/7
1. Implement basic framework/interface for connect 5 (e.g. rules of game, getting next move, game over, etc…)
2. Create interface to work with (e.g. GUI for board, how to feed the board into engine, etc…)
3. Implement basic sequential versions of minimax with/without pruning
4. Implement basic sequential evaluation function
5. Test out our implementations and make basic improvements to sequential version
6 .Gather performance metrics on sequential version
7. Plan out how we will parallelize the game tree search based on our sequential implementation 

## 4/8 - 4/14
1. Implement basic parallel versions of minimax with/without pruning
2. Test out performance, gather metrics for different search depths/breadths
3. Try different ways of workload assignment for searching the branches
4. Plan out how to approach parallelizing the evaluation function

## 4/15 - 4/19 (midterm report due)
1. Write milestone report
2. Begin to implement parallelization of evaluation function in a vacuum

## 4/19 - 4/29
1. Finish implementing basic parallel evaluation in a vacuum
2. Test performance, gather metrics, try different workload assignments
3. Integrate parallelization of evaluation function with the parallelization of search tree
4. Test how well the integration works and scales to more processors
5. Test how strong the engine plays at high depths

## 4/30 - 5/6
1. Final touch-ups on project
2. Write final paper
3. Prepare presentation materials


