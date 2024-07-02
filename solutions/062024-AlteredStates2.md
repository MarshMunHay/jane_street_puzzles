# Altered States 2

This project was my entry to one of the monthly [puzzles](https://www.janestreet.com/puzzles/altered-states-2-index/) hosted by
[Jane Street](https://www.janestreet.com/puzzles/). 

## Intro

The overall goal of the project is to *smoosh* as many U.S. states into a 5x5 grid and maximize the score. The score is 
determined by the population of the state given by the [2020 Census](https://en.wikipedia.org/wiki/2020_United_States_census#State_rankings).
There are some restrictions and qualification to determine whether a state's score to the overall score. A state can contribute to the overall score 
if it is correctly embedded. A state is correctly embedded if it can be composed by incremental single movements in a diagonal, vertical or horizontal direction. 
Another meaningful constraint is that a state's score is only counted once. A single embedding of California is only counted towards the score once.

## Methods

Given the objective of the puzzle and the combinatorial size of the search space, I recognized that I could approach this problem using
a tool from combinatorial optimization.   

### [Simulated Annealing](https://en.wikipedia.org/wiki/Simulated_annealing)

I used simulated annealing (SA) to produce a viable matrix to submit to the competition. I had used combinatorial optimization 
algorithms before however I never used SA. This puzzle had several aspects that indicated that SA might work well. First, 
the objective function is well-defined. The score can be calculated by determining how many states can be found in a matrix. Minimizing the 
objective is a subtraction of the highest possible score (the sum of the population all states) and the sampled score. Second, 
neighbors of a given matrix are easily produced. The positions of letters in a matrix could be swapped or a new letter could be introduced into the matrix randomly. 
Both of these changes result in a new matrix that is only one edit compared to the prior matrix. 

In addition to the well-formed objective function and moves, I could also add constraints to the objective function by discounting
matrices that contained `california` or did not have a continuous connection of states from coast to coast. 

The solution I submitted ranked `204` but it contained several of the awards given out for solutions that had additional constraints.
My other solution with no constraints would have ranked at `18`.

Finding a good solution was relatively easy. However, improving on a solution was much more difficult. One strategy I implemented
was rerunning the process on an already produced solution. It was difficult to say how much improvement was gained by employing this
strategy but I would 

## Resources

These are some resources that I used to solve this problem.

* Weisstein, Eric W. "Contiguous USA Graph." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/ContiguousUSAGraph.html
* S. Kirkpatrick, C. D. Gelatt, and M. P. Vecchi. Optimization by simulated annealing. Science, 220:
  671–680, 1983. [p13, 14]
* https://en.wikipedia.org/wiki/Simulated_annealing
* https://github.com/perrygeo/simanneal