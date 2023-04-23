---
layout: page
title: Midterm report
permalink: /examples/
---


## Summary of Work Done So Far and Preliminary Results (1-2 Paragraphs):
To start off, we have implemented a basic sequential minimax tree search with and without alpha-beta pruning for Connect 5. This includes writing the evaluation function, which we believe to be relatively robust. Our tree search can search up to varying depths and will show the best variation. We have then tried to parallelize both our evaluation function and our search itself. For our evaluation function, we are using ISPC, as it is a data parallel operation, and for our tree search, we are using openMP. To test the correctness of our parallel implementations, we are comparing the results to the results of the sequential implementation, and it looks as if our parallel implementations are consistent.
We also are profiling our performances, keeping track of the total time, time spent on evaluation, as well as the number of evaluations, and pruned branches at different search depths in different positions. Below are some of the preliminary results we have gathered from our performance analysis.

Overall: We can search to small depths in reasonable time, while larger depths take a really long time. In the purely sequential version, most of the time is spent on evaluation, so there would be a lot to gain from a good parallelization of that.
- Open mp:
    - Roughly 2x speedup on a 19x19 board in a “difficult case”
    - Severe (10x) slowdown on a 19x19 board in a “trivial case” (lack of pruning in the root results in lots of unnecessary work).
- Eval:
    - ISPC only speeds up about 10-20%  if we search bounds + 1 on each size 
    - If we increase search bounds to bounds + 3 on each size, we get around a 2x speedup
Above statistics are on limited sample size of test cases
- Performance evaluation:
    - Profiled on ~10 test boards, using valgrind/kcachegrind
    - Evaluation function responsible for most of the time (93%)
        - Lots of potential for speedup due to this
        - Summarizing empty tiles is data parallel (good for SIMD)
        - Check_5_straight is used to see if the game is over, a more efficient implementation is possible 
        - Evaluations of boards can be done in parallel



## Schedule/Tasks Moving Forward (½ week increments):
4/19-4/22: explore how we can improve both parallel ISPC performance and parallel tree search to optimize pruning.
Use performance evaluation tools to determine bottlenecks.

4/23-4/25: integrate the parallel ISPC and parallel search together, test and get performance metrics

4/26-4/29: try using CUDA for evaluation, integrated with tree search. This could be challenging as we need to figure out how to maximally load boards into the GPU during the parallel search so we maximize GPU utilization

4/29-5/4: continue exploring CUDA, and create final report and final poster presentation

## Evaluation of Goals (both plan-to-do’s and nice-to-do’s) + Updated Goals:
**Previous Plan-to-Do’:**

- Parallelize Minimax without pruning using OpenMP: we have a basic implementation of this, but we need to figure out how to optimize where in the tree we begin spawning tasks
- Parallelize Minimax with pruning using OpenMP: same as above, but we need to figure out where to start spawning tasks in order to maximize pruning
- Create a Baseline Evaluation Function: we have a pretty reasonable evaluation function right now
- Parallelize Game Board Evaluation with ISPC:We are parallelizing our evaluation across each row and column using ISPC, but the speedup is limited right now, so we need to see how to improve this
- Incorporate parallel game board evaluation with parallel game tree search: This should be achievable moving forward

**Previous Nice-to-Do’s:**

- Try the above using MPI: probably not a priority
- Use CUDA to parallelize board evaluation: this is something we can probably prioritize moving forward if we get time for it, especially as we see that the evaluation takes the majority of the time, so this would be interesting
- Improve our evaluation function to make our engine stronger: probably not a major priority for now
- Figure out heuristics that can improve our workload distribution for the game tree and position evaluation: this is another thing that we can prioritize, as we want to maximize pruning in parallel searches, so this is an interesting problem that can yield lots of benefit
- Make game-specific optimizations to our algorithm: probably not a major priority


## Poster Session Plan:
We can show them our performance metrics of pure sequential vs. our implementations with our parallelism (ISPC evaluation, OMP search, etc…). We’re not sure how much we can speed up the search right now, but if it’s fast enough, we can show them how well the engine plays given that they can search different depths, as right now, searching at depths above 3-4 takes a while. Possibly include statistics about how well it plays against a typical human player.

List of Issues/Concerns: 
- Not sure how to optimize ISPC further to increase speedup, need to ask instructors
- A good pruning implementation in OpenMP requires a lot of communication between workers (early termination, alpha/beta, board state, etc)
- OpenMP implementation suffers from severe workload imbalance.
- Not sure how to profile SIMD code
