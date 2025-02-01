# Knights Move 6

## Intro

The goal of this puzzle was to construct two paths through a network that meet the given constraints from different starting nodes to different ending nodes. Each node in the network has an integer value of A, B, or C. The score of a trip is given by 

'''
The “score” for a trip is calculated as follows:

Start with A points.
Every time you make a move:
if your move is between two different integers, multiply your score by the value you are moving to;
otherwise, increment your score by the value you are moving to.
'''

## Methods

I recognized this problem as a TSP problem that could be solved through Simmulated Annealing. The energy calculation for the simmulated annealing was interesting because the process was searching for paths with a score equal to 2024. In my experience with Simulated Annealing, the energy calculation was oriented to searching for a minimum or maximum score.  
