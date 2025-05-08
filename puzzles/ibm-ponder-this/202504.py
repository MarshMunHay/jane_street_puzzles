import copy
from simanneal import Annealer
from functools import reduce
import random
import math
import numpy as np
import networkx as nx
import heapq
from copy import deepcopy

directions = [[0, 1],[0, -1],[-1, 0],[1, 0]]

puzzle_grid_s = '''0  0  1  5  1  0  0  0  0  0  1 19  0  7  4  2  7 12  7  1
     2  6  0  0  1  8  0  8  1  2  1  0  8  9  1  7 10 13 10  6
     4 11  6  7  5  5 14  1 12  1  0  2  0  2  2  5  1 10  0 14
     15 12  2  5 18  6 19 16 18 11 14  3  1  2  3  3  8  2  1  9
     5  6  8 18  4 17  7 16 14 13  4 13  8  1  2  2  7  5 11 12
     6  7 13 16  1 14  7 17 18  9 14  6 16 10  0  3  2  0  6  5
     11  5 11  3 14 19 19  4 17 16  3 12 17 17  1  2 12  6  7 11
     18  6  6  3 19 13  7  9  5 13  4  4  2 13  2  0  0  5  4  6
     17 19  7  2  4  3  4  1 16  9 13 17 17 15  6  9  1  5  2  0
     8  8 17 18 10 12 10  0  0 13 13 10  8  0  0  7 18 10  6  3
     13  3 19  3  5  9 17 16 12  2 19  9  1 17  3  0 10 11  4 19
     14  5 11 13 15  6  5 10  6  1  7  3  4 15 10 10 13  4  9  7
     2 12  5  7  7 16  3  2 18 14 11 18 12 15  4  2 12 15 10  6
     12  5  2 15  8  9 18  9  5  1 17 17  1  0  8  9  5  6  8 13
     9 13  5  3  9  8 18 15 10  6 12 18 11 15  2 12  6  8 12 15
     14  4  2  0 13  2 18 12 16  2  4 13  0  3 16 15 15 16  7  7
     6 12  1 14  4 12  8 14 10  0 15 16 13  4  5 12  5  2 16 12
     5  5  3  0  8  0  5 16 11  4 17 13 18 17  0  9  8 16 13  6
     15 13 13  5  6  7  9 15 12 18  2 12 19  4  9  5  6  8  9  3
     12 10 11  2  5  8 11  7 16 12  0 14 10  5  9  0 15  4 11  3'''
puzzle_grid_s = puzzle_grid_s.replace('  ', ' ')
puzzle_grid_s = [row.replace('   ', '').split(' ') for row in puzzle_grid_s.split('\n')]

for r in range(len(puzzle_grid_s)):
    for c in range(len(puzzle_grid_s)):
        puzzle_grid_s[r][c] = eval(puzzle_grid_s[r][c])
puzzle_grid = puzzle_grid_s

def is_in_bounds(path, g):
    row_limit = len(g) - 1
    col_limit = len(g[0]) - 1
    pos = [0,0]
    in_bounds = lambda x: (x[0] >= 0) and (x[0] <= row_limit) and (x[1] >= 0) and (x[1] <= col_limit)
    for direction in path:
        pos = [pos[0] + direction[0], pos[1] + direction[1]]
        if not in_bounds(pos):
            return False
    return True

def create_network(g):
    graph = nx.Graph()
    node_map = dict()
    i = 0
    for row in range(len(g)):
        for col in range(len(g[0])):
            node_map[i] = [row, col]
            graph.add_node(i, weight=g[row][col])
            i += 1

    node_to_id = {tuple(pos): id_ for id_, pos in node_map.items()}

    for row in range(len(g)):
        for col in range(len(g[0])):
            latest_pos = (row, col)
            for d in directions:
                neighbor_node = (latest_pos[0] + d[0], latest_pos[1] + d[1])
                if is_in_bounds([neighbor_node], g):
                    graph.add_edge(node_to_id[latest_pos], node_to_id[neighbor_node])

    return node_to_id, node_map, graph


def beam_search(g, width=10, maxsteps=500):
    total_weights = sum([g.nodes[n]['weight'] for n in g.nodes])
    total_nodes_to_visit = sum([1 for n in g.nodes if g.nodes[n]['weight'] > 0])
    # state is score, mean_distance, -gp, gp, step, latest_node, path, graph
    start_state = (-1, 4000, -1, 1, 0, 0, [0], g)
    beam = [start_state]

    visited_states = set()

    for i in range(1, maxsteps):
        next_beam = []
        for state in beam:
            _, _, _, gp, step, l_node, path, gra = state
            score_so_far = math.isqrt(gp)
            for neighbor in gra.neighbors(l_node):
                copied_graph = deepcopy(gra)
                threshold = copied_graph.nodes[neighbor]['weight']
                if threshold <= score_so_far:
                    updated_score = gp + threshold
                    copied_graph.nodes[neighbor]['weight'] = 0
                    updated_path = path + [neighbor]
                    signature = tuple([copied_graph.nodes[n]['weight'] for n in copied_graph.nodes])
                    key = tuple((neighbor, updated_score, signature))
                    if key in visited_states:
                        continue
                    visited_states.add(key)

                    remaining_value_nodes = [n for n in copied_graph.nodes if copied_graph.nodes[n]['weight'] > 0]
                    mean_distance_to_remaining_value_nodes = np.mean([nx.shortest_path_length(copied_graph, l_node, n) for n in remaining_value_nodes])
                    count_remaining_value_nodes = sum([1 for n in copied_graph.nodes if copied_graph.nodes[n]['weight'] > 0])

                    eaten = count_remaining_value_nodes
                    score = eaten
                    heapq.heappush(next_beam, (score, mean_distance_to_remaining_value_nodes, -updated_score, updated_score, i, neighbor, updated_path, copied_graph.copy()))

                    if count_remaining_value_nodes == 0:
                        return updated_path


        beam = heapq.nsmallest(width, next_beam)
        if not beam:
            break

    return beam[0]


if __name__ == '__main__':
    input_graph = copy.deepcopy(puzzle_grid)
    node_t_id, node_t_pos, g = create_network(input_graph)

    res = [0]
    found_path = beam_search(g)
    print(found_path)


