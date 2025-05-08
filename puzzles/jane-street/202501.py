import numpy as np
from copy import deepcopy
import networkx as nx
import heapq
import math
import re
from itertools import permutations, combinations, product
from functools import reduce
import time

options = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

def is_valid_sudoku(grid):
    def has_duplicates(lst):
        return len(lst) != len(set(lst))

    n = len(grid)
    subgrid_size = int(n ** 0.5)
    for i, row in enumerate(grid):
        if has_duplicates(row):
            return False

    for i, col in enumerate(range(n)):
        column = [grid[row][col] for row in range(n)]
        if has_duplicates(column):
            return False

    for i in range(0, n, subgrid_size):
        for j in range(0, n, subgrid_size):
            subgrid = []
            for x in range(i, i + subgrid_size):
                for y in range(j, j + subgrid_size):
                    subgrid.append(grid[x][y])
            if has_duplicates(subgrid):
                return False
    return True

def get_valid_values(node_id, graph):
    copied_graph = graph

def get_contrained_values(id, graph):
    one_hop_neighbor_values = set([graph.nodes[v]['value'] for v in graph.neighbors(id) if not graph.nodes[v]['is_free']])
    return [option for option in options if option not in one_hop_neighbor_values]

def generate_numbers(constraints, length, mandatory, current="", results=set()):
    if len(current) == length:
        if all(str(m) in current for m in mandatory):
            results.add(current)
        return

    position = len(current)
    allowed_digits = constraints[position]

    for digit in allowed_digits:
        if str(digit) not in current:
            generate_numbers(constraints, length, mandatory, current + str(digit), results)

def beam_search(g, width=10, maxsteps=8):
    # state is score, row, path, graph
    start_state = (1, 0, [], g)
    beam = [start_state]

    for row in range(maxsteps):
        next_beam = []
        highest_score = 0
        for state in beam:
            score, _, path, g = state
            lower = row * 9
            upper = row * 9 + 9
            constraints = [get_contrained_values(i, g) for i in range(lower, upper)]
            generated_numbers = set()
            generate_numbers(constraints, 9, [0, 2, 5], '', generated_numbers)
            for neighbor in generated_numbers:
                copied_graph = deepcopy(g)
                tmp_path = path + [int(neighbor)]
                tmp_score = tmp_path[-1] if score > 0 else math.gcd(score, int(neighbor))
                if (tmp_score >= 10e5) and (tmp_score <= 10e8):
                    for i, digit in enumerate(neighbor):
                        copied_graph.nodes[lower + i]['value'] = digit
                        copied_graph.nodes[lower + i]['is_free'] = False

                    if (row == 0):
                        heapq.heappush(next_beam, (-tmp_score, row, tmp_path, copied_graph))
                    elif (tmp_score > highest_score) and (row >= 1):
                        highest_score = max(tmp_score, highest_score)
                        heapq.heappush(next_beam, (-tmp_score, row, tmp_path, copied_graph))

        beam = heapq.nsmallest(170000 // (row + 1), next_beam)

    return beam[0]

if __name__ == '__main__':
    '''
    Consider row by row,
    
    calculate 
    '''
    G = nx.sudoku_graph()
    puzzle = [[
        [0, 0, 0, 0, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 5],
        [0, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 5, 0, 0],
    ]]
    flatten_puzzle_input = np.reshape(np.array(puzzle), (81))
    for i, node in enumerate(G.nodes):
        G.nodes[node]['value'] = flatten_puzzle_input[i]
        G.nodes[node]['is_free'] = flatten_puzzle_input[i] == 0
    results = beam_search(G)
    print(results)
    print(G.nodes)


