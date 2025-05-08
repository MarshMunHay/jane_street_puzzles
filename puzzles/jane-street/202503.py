import pandas as pd
import numpy as np
import networkx as nx
import copy
from typing import List, Tuple
from functools import reduce
import simanneal
import math
from collections import deque

# Let the outer dot and the center of the square on the perimeter be one unit away
# There will be padding cells to fill in the area to have.
# The top left cell will be considered [0,0] and contain the first row of hints
CONFIG_LEN = 10
example_board_width = CONFIG_LEN + 2
nonpadded_board_width = example_board_width - 2
end_nodes = [
            [[0, i] for i in range(1, example_board_width-1)],
            [[i, 0] for i in range(1, example_board_width-1)],
            [[example_board_width - 1, i] for i in range(1, example_board_width-1)],
            [[i, example_board_width - 1] for i in range(1, example_board_width-1)]
]
example_hints = {
    0: {
        3: 112,
        5: 48,
        6: 3087,
        7: 9,
        10: 1
    },
    2: {
        11: 4
    },
    3: {
        11: 27
    },
    4: {
        0: 27
    },
    7: {
        11: 16
    },
    8: {
        0: 12
    },
    9: {
        0: 225
    },
    11: {
        1: 2025,
        4: 12,
        5: 64,
        6: 5,
        8: 405
    }
}
VALID_END_NODES = reduce(lambda x, y: x + y, end_nodes)
ALL_NODES = [[i, j] for j in range(1, example_board_width-1) for i in range(1, example_board_width-1)]

given_hints = [[col, row] for col, item in example_hints.items() for row, val in item.items()]


def is_out_of_bounds(node: Tuple[int,int], b):
    row = len(b)
    col = len(b[0])
    return (node[0] < 0) or (node[0] > row-1) or (node[1] < 0) or (node[1] > col-1)
def determine_new_direction(p:int, dir:Tuple[int,int]):
    if p == 1:
        new_dir = [dir[1], dir[0]]
    elif p == -1:
        new_dir = [-dir[1], -dir[0]]
    else:
        new_dir = dir

    # print(p, dir, new_dir)
    return new_dir
def determine_initial_direction(source: Tuple[int,int]):
    if source[0] == 0:
        direction = [1, 0]
    if source[0] == example_board_width-1:
        direction = [-1, 0]
    if source[1] == 0:
        direction = [0, 1]
    if source[1] == example_board_width-1:
        direction = [0, -1]
    return direction

def walk(b, source: Tuple[int,int], direction: Tuple[int, int]):
    path = [source]
    cur_node = source
    cur_direction = direction
    count_bends: Tuple[int, int] = []

    # During each direction change store the length of the line segment
    linesegment_length: List[int] = []
    # During each direction change store the node that had the pivot
    prev_pivot = cur_node

    valid_nodes = [node for node in VALID_END_NODES if node not in [source]]
    while cur_node not in valid_nodes:
        cur_node = [c + cur_direction[i] for i, c in enumerate(cur_node)]
        # if cur_node is out_of_bounds break
        distance = abs(cur_node[0] - prev_pivot[0]) + abs(cur_node[1] - prev_pivot[1])
        if is_out_of_bounds(cur_node, b):
            linesegment_length.append(distance)
            break

        cur_direction = determine_new_direction(p=b[cur_node[0]][cur_node[1]], dir=cur_direction)
        cell_value = b[cur_node[0]][cur_node[1]]
        is_pivot = (cell_value == 1) or (cell_value == -1)
        if is_pivot:
            # print(prev_pivot)
            # print(cur_node)
            prev_pivot = cur_node
            linesegment_length.append(distance)
            count_bends.append(cur_node)

        path.append(cur_node)

    linesegment_length.append(distance)
    return (path, linesegment_length, count_bends)

def print_board(b):
    pad = 4
    prnt_out = ''
    for r in b:
        prnt_out += '\n'
        prnt_out += ','.join([str(c).rjust(4, ' ') for c in r])
    #print(prnt_out)

example_board = [[0 for i in range(example_board_width)] for i in range(example_board_width)]
# Row Col notation
# Populate Board
for r, cols in example_hints.items():
    for col, v in cols.items():
        example_board[r][col] = v

def update(b, node_list):
    '''
    An update to a non given hint should occur when the source is a given hint and end is a non given hint

    A NONVALID RESULT OCCURS WHEN
    1. the start is a given hint and the end is a given hint and the start hint and end hint do not match

    Returns a copy of board that has been updated

    Update Hints
    :param b:
    :return:
    '''
    # IF A HINT IS GIVEN, IT CANNOT BE CHANGED
    copied_board = copy.deepcopy(b)
    for source in node_list:
        direction = determine_initial_direction(source=source)
        path, line_segments, pivots_encountered = walk(b=b, source=source, direction=direction)
        # TODO: Hints are numbers and no number has only one multiple
        product = reduce(lambda x, y: x * y, line_segments)

        if len(line_segments) != 1:
            product = reduce(lambda x, y: x * y, line_segments)
        else:
            product = ''
        path_start = path[0]
        path_end = path[-1]
        is_start_given_hint = path_start in given_hints
        is_end_given_hint = path_end in given_hints
        if (is_start_given_hint) & (not is_end_given_hint):
            if product == example_hints[path_start[0]][path_start[1]]:
                copied_board[path_end[0]][path_end[1]] = example_hints[path_start[0]][path_start[1]]
        elif (is_end_given_hint) & (not is_start_given_hint):
            if product == example_hints[path_end[0]][path_end[1]]:
                copied_board[path_start[0]][path_start[1]] = example_hints[path_end[0]][path_end[1]]
        else:
            pass
    return copied_board

def factors(n):
    """Returns all the factors of a positive integer"""
    factors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            factors.append(i)
            if n // i != i:
                factors.append(n // i)
    factors.sort()
    return factors
def filter_factors(factors, n):
    return [fact for fact in factors if fact <= n]
def is_invalid_adjacent_move(distance, node, board_width):
    # distance of 1 and not on the perimeter
    return (distance == 1) and (0 < node[0] < board_width-1) and (0 < node[1] < board_width-1)
def determine_pivot(dir1, dir2):
    # swap
    if dir1 == dir2:
        return ''
    elif (dir2[0] == dir1[1]) & (dir2[1] == dir1[0]):
        return '\\'
    else:
        return '/'
def inbound(node):
    return (0 <= node[0] < len(example_board) - 2) and (0 <= node[1] < len(example_board) - 2)
def test_has_adjacent_mirrors_row_col(p: List[int]):
    row_major = np.array(p).reshape(int(np.sqrt(len(p))), int(np.sqrt(len(p))))
    test_col = []
    for i in range(len(row_major)):
        start = 0
        for j in range(1, len(row_major)):
            cur_val = row_major[i, start]
            next_val = row_major[i, j]
            if ((cur_val == next_val) or (cur_val == -next_val)) and (cur_val != 0):
                test_col.append(True)
                break
            start += 1

    col_major = row_major.T
    test_row = []
    for i in range(len(col_major)):
        start = 0
        for j in range(1, len(col_major)):
            cur_val = col_major[i, start]
            next_val = col_major[i, j]
            if ((cur_val == next_val) or (cur_val == -next_val)) and (cur_val != 0):
                test_row.append(True)
                break
            start += 1
    return (sum(test_col) + sum(test_row)) > 0


def has_adjacent_mirrors_row_col(p: List[int]):
    board_width = len(example_board) - 2
    for i in range(len(p)):
        col = i % board_width
        row = i // board_width
        if inbound([col+1, row]):
            comp_index = row * (board_width) + (col+1)
            if ((p[i] != 0) and (p[comp_index] != 0)):
                return True

        elif inbound([col-1, row]):
            comp_index = row * (board_width) + (col-1)
            if ((p[i] != 0) and (p[comp_index] != 0)):
                return True

        elif inbound([col, row+1]):
            comp_index = (row+1) * (board_width) + (col)
            if ((p[i] != 0) and (p[comp_index] != 0)):
                return True

        elif inbound([col, row-1]):
            comp_index = (row-1) * (board_width) + (col)
            if ((p[i] != 0) and (p[comp_index] != 0)):
                return True
    return False

def bfs(board: List[List[int|str]], source:Tuple[int,int], hints):
    # TODO: CANNOT MAKE ILLEGAL PIVOTS
    # IF THERE IS A CHANGE IN DIRECTION, NEED TO CALCULATE THE LENGTH OF THE LINE SEGMENT
    board_width = len(board[0])
    hint_val = hints[source[0]][source[1]]
    facts = filter_factors(factors(hint_val), board_width-2)
    queue = deque()
    results = []
    cur_direction = determine_initial_direction(source=source)
    directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]

    # PREPOPULATE QUEUE FOR FIRST STEPS
    for f in facts:
        magnitude_dir = [d * f for d in cur_direction]
        updated_hint_val = hint_val / f
        new_pos = [source[i] + n for i, n in enumerate(magnitude_dir)]
        entry_path = [(hint_val, source, cur_direction), (updated_hint_val, new_pos, cur_direction)]
        queue.append(entry_path)

    while queue:
        path: List[Tuple[int, Tuple[int, int]]] = queue.popleft()
        remaining_factors, cur_node, cur_direction = path[-1]
        _, prev_node, _ = path[-2]
        prev_factor = sum([cur_node[i] - prev_node[i] for i,n in enumerate(prev_node)])
        # EVALUATE WHETHER CUR NODE IS ON PERIMETER AND REMAINING FACTOR == 1
        # if ((cur_node[0] == 11) or (cur_node[1] == 11)) and (remaining_factors == 1):
        if ((cur_node[0] == board_width - 1) or (cur_node[1] == board_width - 1)) or ((cur_node[0] == 0) or (cur_node[1] == 0)):
            if remaining_factors == 1:
                results.append(path)
            else:
                continue
        elif not is_out_of_bounds(cur_node, board):
            facts: List[int] = filter_factors(factors(remaining_factors), board_width-2)
            opposite_dir = [-d for d in cur_direction]
            possible_directions = [d for d in directions if d != opposite_dir]
            possible_directions = [d for d in possible_directions if d != cur_direction]
            for fact in facts:
                for direction in possible_directions:
                    # TODO DETERMINE WHETHER MOVE IS LEGAL
                    new_path = copy.deepcopy(path)
                    magnitude_dir = [d * fact for d in direction]
                    new_pos = [cur_node[i] + n for i, n in enumerate(magnitude_dir)]
                    updated_hint_val = remaining_factors / fact
                    dist = abs(sum([new_pos[i] - cur_node[i] for i, p in enumerate(cur_node)]))
                    if is_invalid_adjacent_move(dist, new_pos, board_width):
                        continue
                    else:
                        new_path.append((updated_hint_val, new_pos, direction))
                        queue.append(new_path)
        # IF remaining_factors ==1 and not on the parameter
        # any remaining_factors and out_of_bounds
        else:
            continue
    return results
def convert_path_to_vector(pathd):
    b = copy.deepcopy(example_board)
    nodes = [p[1] for p in pathd]
    dirs = [p[2] for p in pathd]

    start = 0
    end = len(nodes)-1
    for i in range(start, end):
        pvt = determine_pivot(dirs[i], dirs[i+1])
        if pvt != '':
            b[int(nodes[i][0])][int(nodes[i][1])] = pvt
    return [b[i][j] for i in range(1, len(b) - 1) for j in range(1, len(b) - 1)]
def map_str_to_num(s):
    m = {
        0:0,
        '/':-1,
        '\\':1
    }
    return m[s]
def determine_valid_array_pair(arr_a, arr_b):
    for a, b in zip(arr_a, arr_b):
        if (a == -1) and (b == 1):
            return False
        if (a == 1) and (b == -1):
            return False
    return True
def combine_array(arr_a, arr_b):
    exit_arr = []
    for a, b in zip(arr_a, arr_b):
        if a == b:
            exit_arr.append(a)
        elif (a == 0) and ((b == 1) or (b == -1)):
            exit_arr.append(b)
        else:
            exit_arr.append(a)
    return exit_arr
def get_combinations_dfs(arr, k, b):
    result = []

    def dfs(start: int, current_combination: List[int]):
        if start == k:
            result.append(current_combination.copy())
            return

        node_start = arr[start][1]
        for path in arr[start][0]:
            is_array_comb_valid = determine_valid_array_pair(current_combination, path)
            if is_array_comb_valid:
                composition_paths = combine_array(current_combination, path)
                prev_combination = copy.deepcopy(current_combination)
                current_combination = composition_paths
                if not test_has_adjacent_mirrors_row_col(current_combination):
                    dfs(start + 1, current_combination)
                    if len(result) == 1:
                        pass
                        # print(result)
                current_combination = prev_combination

    dfs(0, [b[i][j] for i in range(1, len(b) - 1) for j in range(1, len(b) - 1)])
    return result
def objective_f(b):
    '''
    The Objective Function Decreases
    When There is a match between a valid end note and is_hint

    The objective function should increase when there are pivots that does nothing
    or a cause a given end node to
    :param b:
    :return:
    '''
    score = len(given_hints)

    for source in given_hints:
        direction = determine_initial_direction(source=source)
        path, line_segments, pivots_encountered = walk(b=b, source=source, direction=direction)
        start = path[0]
        end = path[-1]
        is_start_given_hint = start in given_hints
        start_val = b[source[0]][source[1]] if b[source[0]][source[1]] != 0 else 0
        end_val = b[end[0]][end[1]] if b[end[0]][end[1]] != 0 else 0
        product = reduce(lambda x, y: x*y, line_segments)
        if (product == start_val) or (product == end_val):
            score -= 1

    return score

def backward_pass(b):
    '''
    Perform a walk from every node

    if a non given hint node with a hint ends on a non given hint node
    reset the hints to ''
    '''
    copied_board = copy.deepcopy(b)
    for source in VALID_END_NODES:
        direction = determine_initial_direction(source=source)
        path, line_segments, pivots_encountered = walk(b=copied_board, source=source, direction=direction)
        start = path[0]
        end = path[-1]
        is_start_hint = start in given_hints
        is_end_hint = end in given_hints

        update_start = '' if not is_start_hint else b[start[0]][start[1]]
        update_end = '' if not is_end_hint else b[end[0]][end[1]]
        if (is_start_hint) & (copied_board[start[0]][start[1]] == copied_board[end[0]][end[1]]):
            # print('Start Hint', start, end)
            copied_board[start[0]][start[1]] = update_start
            copied_board[end[0]][end[1]] = update_start
        elif (is_end_hint) & (copied_board[start[0]][start[1]] == copied_board[end[0]][end[1]]):
            # print('End Hint', end, start)
            copied_board[start[0]][start[1]] = update_end
            copied_board[end[0]][end[1]] = update_end
        elif (not is_start_hint) & (not is_end_hint):
            copied_board[start[0]][start[1]] = update_start
            copied_board[end[0]][end[1]] = update_end


    return copied_board

def fill(b, node_list):
    '''
    Very similar to update but fill the remaining not hint nodes
    :param b:
    :return:
    '''
    copied_board = copy.deepcopy(b)
    for source in node_list:
        direction = determine_initial_direction(source=source)
        path, line_segments, pivots_encountered = walk(b=b, source=source, direction=direction)
        # TODO: Hints are numbers and no number has only one multiple
        product = reduce(lambda x, y: x * y, line_segments)

        if len(line_segments) != 1:
            product = reduce(lambda x, y: x * y, line_segments)
        else:
            product = ''
        path_start = path[0]
        path_end = path[-1]
        is_start_given_hint = path_start in given_hints
        is_end_given_hint = path_end in given_hints

        if (not is_start_given_hint) and (not is_end_given_hint):
            copied_board[path_start[0]][path_start[1]] = product
            copied_board[path_end[0]][path_end[1]] = product

    return copied_board
def overwrite_board(b, config):
    copied_b = copy.deepcopy(b)
    for j in range(len(config)):
        col = j % (len(example_board) - 2)
        row = j // (len(example_board) - 2)
        copied_b[row + 1][col + 1] = config[j]
    return copied_b

if __name__ == '__main__':
    paths_by_example_hist = dict()
    for placement in given_hints:
        if placement[0] not in paths_by_example_hist:
            paths_by_example_hist[placement[0]] = dict()
        if placement[1] not in paths_by_example_hist[placement[0]]:
            paths_by_example_hist[placement[0]][placement[1]] = bfs(example_board, placement, hints=example_hints)

    # FILTER OUT ANY PATH THAT ENDS WITH A GIVEN HINT
    for row, cols in paths_by_example_hist.items():
        for col in cols:
            prefilter = len(paths_by_example_hist[row][col])
            accepted_paths = []
            for path in paths_by_example_hist[row][col]:
                start_hint = example_hints[path[0][1][0]][path[0][1][1]]
                end_hint = example_hints[path[-1][1][0]][path[-1][1][1]] if path[-1][1] in given_hints else 0
                if start_hint == end_hint:
                    accepted_paths.append(path)
                if path[-1][1] not in given_hints:
                    accepted_paths.append(path)
            paths_by_example_hist[row][col] = accepted_paths


            postfilter = len(paths_by_example_hist[row][col])

    flat_paths_by_example_hist = [[] for i in range(len(given_hints))]
    count = 0
    for row, cols in paths_by_example_hist.items():
        for col in cols:
            paths = paths_by_example_hist[row][col]
            for path in paths:
                flat_path = list(map(lambda x: map_str_to_num(x), convert_path_to_vector(path)))
                flat_paths_by_example_hist[count].append(flat_path)
            count += 1

    for i in flat_paths_by_example_hist:
        if len(i) == 1:
            for config_index in range(len(i[0])):
                col = config_index % (len(example_board) - 2)
                row = config_index // (len(example_board) - 2)
                if example_board[row + 1][col + 1] == 0:
                    example_board[row + 1][col + 1] = i[0][config_index]

    filtered_hints = [(f,g) for f,g in zip(flat_paths_by_example_hist, given_hints) if len(f) > 1]
    filtered_hints.sort(key=lambda x: [len(x[0]), x[1]])

    res = get_combinations_dfs(filtered_hints, len(filtered_hints), example_board)

    for r in res:
        test = update(overwrite_board(example_board, r), given_hints)
        if objective_f(test) == 0:
            state = test

    # print_board(state)
    filled = fill(state, VALID_END_NODES)
    # print_board(filled)
    submission = [list(filter(lambda x: x not in given_hints, row)) for row in end_nodes]
    submission = [sum(map(lambda x: filled[x[0]][x[1]], row)) for row in submission]
    final_submission = reduce(lambda x, y: x * y, submission)
    print(final_submission)

