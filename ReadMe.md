# Puzzles

## Introduction 

I primarily work on puzzles hosted by Jane Street and IBM Research because I 
enjoy the mathematical reasoning and algorithmic thinking. I focus on
any problem using combinatorial optimization, problem modeling, or mathematical reasoning.
When I am not too busy, I do my best to solve every puzzle.

## Puzzle Index

### Jane Street

#### [Altered States 2](https://github.com/MarshMunHay/jane_street_puzzles/blob/main/puzzles/jane-street/202406.py)

##### Framing 

The goal of this puzzle was to maximize the score by embedding the name
of US states into a 5x5 grid. States could be spelled by making 1 space movements.
The name of a state could be altered by 1 letter. One letter could be replaced in the name of state could
be replaced with another.

##### Solution

After reading through the puzzle, I had an immediate hunch that
[Simulated Annealing](https://en.wikipedia.org/wiki/Simulated_annealing) would work extremely well.
A 5x5 grid could become another 5x5 grid by swapping single letter of the matrix could be swapped with another letter. 
I used DFS to find names of states in the 5x5 grid and scored the solution based on the rubric.

#### [Knight Moves 6](https://github.com/MarshMunHay/jane_street_puzzles/blob/main/puzzles/jane-street/202410.py)

##### Framing

A 6x6 grid has 3 grouping of cells (A,B,C). A, B, and C can any set of postive distinct integers.
The goal of the puzzle was find two corner-to-corner trips that reached a score of 2024 points. The score was calculated 
by starting with A points and multiplying if the value of the next step was different from the last. Otherwise, add the next step to the 
score.

The sum of the A,B,C must be less than 50.

##### Solution

I used [Simulated Annealing](https://en.wikipedia.org/wiki/Simulated_annealing) to solve this problem. For the strategy
of assigning A,B,C, I started with random permutations of 1, 2, and 3 until the simulated annealing process yielded a solution.

#### [Somewhat Square Sudoku](https://github.com/MarshMunHay/jane_street_puzzles/blob/main/puzzles/jane-street/202501.py)

##### Framing

This puzzle used sudoku rules to solve a 
sudoku puzzle using nine of the ten digits. The nine 9-digit numbers
formed by the rows of the grid has the highest-possible GCD over any
such grid.

##### Solution

I found this puzzle very frustrating because I *knew* there was an approach
that did not use combinatorial optimization. I solved this puzzle
using [Beam Search](https://en.wikipedia.org/wiki/Beam_search) to model the search
for the maximum gcd. It is way too slow but it worked. 

##### Posted Solution 

* Let S be the sum of the digits in the first row
* Let d be the GCD of the 9-digit numbers formed by the nine rows. 
* The sum of these 9-digit numbers will be 111,111,111×S = 3×3×37×333,667×S, and so d must be a divisor of this value.

* It is 111,111,111 because each number in the row represents a "10s" places
* 3×3×37×333667 is the prime factorization of 111,111,111
* Search for the multiples of (3,3,37,333667) for an eight-digit number.
* Only eight digits because 0 must be included in the row (which is 12,345,679)
* Find a multiple of 12,345,679 where the values can be cycled.

The key here is [cyclic permutation](https://en.wikipedia.org/wiki/Cyclic_permutation).

I share the posted solution here because I really enjoyed why it worked!



#### [Hall of Mirrors 3](https://github.com/MarshMunHay/jane_street_puzzles/blob/main/puzzles/jane-street/202501.py)

##### Framing 

This puzzle focused on the placement of "mirrors" in 10x10 grid. A path could start from any
node follow continue until redirected by a mirror. A mirror had two orientations and could be placed
in any cell. The goal of the puzzle was to connect every numbered node to another node and one node could only
connect to one other node.

##### Solution

I solved this puzzle using Breadth-First search and Depth-First search. I used an objective function to determine
when the puzzle was solved. 

### IBM Ponder This

#### [April 2025 - Challenge](https://github.com/MarshMunHay/jane_street_puzzles/blob/main/puzzles/ibm-ponder-this/202504.py) 

##### Framing

This puzzle was a gated traveling salesperson problem. More nodes were allowed to be visited
once a score was reached. The goal of the puzzle was to visit all nodes that had a positive value.

##### Solution

I solved this puzzle using Beam Search and included a measure for the average distance to all unvisited nodes.

![Search](https://github.com/MarshMunHay/jane_street_puzzles/blob/main/puzzles/ibm-ponder-this/202504.gif)