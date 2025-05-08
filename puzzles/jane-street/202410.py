import numpy as np
import networkx as nx
from simanneal import Annealer
import random
from collections import Counter

# a + b + c <= 50
# MIN: 1 + 2 + 3 = 6
# MAX: 17+15+18 = 50
from collections import defaultdict


l = [1, 2, 3, 4, 5, 6]
rows = {i: 7 - i for i in l[::-1]}
cols = {6-i: k for i, k in enumerate(['a','b','c','d','e','f'][::-1])}
f = lambda x: cols[x[0]] + str(rows[x[1]])

knight_moves = [
    (2, 1), (2, -1), (-2, 1), (-2, -1),
    (1, 2), (1, -2), (-1, 2), (-1, -2)
]

group_to_value = {
    'a': 1,
    'b': 3,
    'c': 2
}

group_to_node = {
    'a': ['a6','a5','a4','a3','a2','a1',
          'b4','b3','b2','b1',
          'c2','c1'
          ],
    'b': ['b6','b5',
          'c6','c5','c4','c3',
          'd4','d3','d2','d1',
          'e2','e1'
          ],
    'c': ['d6','d5',
          'e6','e5','e4','e3',
          'f6','f5','f4','f3','f2','f1'
          ]
}

node_to_group = dict()

for k, v in group_to_node.items():
    for node in v:
        node_to_group[node] = k


def get_knight_moves(x, y):
    moves = []
    for dx, dy in knight_moves:
        new_x = x + dx
        new_y = y + dy
        if 1 <= new_x <= 6 and 1 <= new_y <= 6:
            moves.append((new_x, new_y))
    return moves


board = {}
for row in range(1, 7):
    for col in range(1, 7):
        board[(row, col)] = get_knight_moves(row, col)

remapped_board = dict()
for position, moves in board.items():
    remapped_position = f(position)
    remapped_moves = list(map(f, moves))
    remapped_board[remapped_position] = remapped_moves

class KnightsMoveProblem(Annealer):
    def __init__(self, state, rewards):
        super(KnightsMoveProblem, self).__init__(state)
        self.G = nx.Graph(remapped_board)
        self.rewards = rewards

    def add_node(self):
        '''
        It is not possible to add just one node to a path.

        It has to be done in pairs due to the knights move condition
        '''
        G = self.G
        path = self.state
        initial_energy = self.energy()
        # Select two nodes that are 1 index apart
        n = len(path)
        choice = np.random.randint(0, n-1)
        start = path[choice]
        end = path[choice+1]
        G.remove_edge(start, end)
        sub_paths = nx.all_shortest_paths(G, start, end)
        G.add_edge(start, end)
        beginning = path[:choice]
        ending = path[choice+2:]
        for sp in sub_paths:
            res = all(map(lambda x: (x not in beginning) & (x not in ending), sp))
            new_path = beginning + sp + ending
            assert (sum(Counter(path).values()) == len(Counter(path).keys()))
            if (res) and (len(new_path) >= n):
                self.state = new_path
                return self.energy() - initial_energy


        return int(self.energy() - initial_energy)


    def remove_node(self):
        G = self.G
        path = self.state
        initial_energy = self.energy()
        n = len(path)
        if n == 5:
            self.state = path
            return int(self.energy() - initial_energy)

        # Select two nodes that are 3 indicies apart
        choices = [i for i in range(0, n-3)]
        for i in range(100):
            choice = np.random.choice(choices, 1)[0]
            start = path[choice]
            end = path[choice+3]
            sub_paths = nx.all_shortest_paths(G, start, end)
            # Find the shortest path
            for sp in sub_paths:
                res = all(map(lambda x: (x not in path[:choice]) & (x not in path[choice+3+1:]), sp))
                assert (sum(Counter(path).values()) == len(Counter(path).keys()))
                new_path = path[:choice] + sp + path[choice+3+1:]
                if (res) & (len(new_path) <= len(path)):
                    self.state = new_path
                    return int(self.energy() - initial_energy)
            choices.remove(choice)

        # self.state = new_path
        # print(self.energy() - initial_energy)
        return int(self.energy() - initial_energy)

    def move(self):
        """
        Make the state changes

        1. Add Nodes
        2. Remove Nodes
        """
        initial_energy = self.energy()
        direction = random.random()
        if direction > 0.5:
            e = self.add_node()
            return e
        else:
            e = self.remove_node()
            return e

    def eval_path(self, path):
        score = 0
        previous_node = None
        for node in path:
            if previous_node:
                previous_group = node_to_group[previous_node]
                current_group = node_to_group[node]
                prev_score = score
                score = score * self.rewards[node_to_group[node]] if current_group != previous_group else score + self.rewards[node_to_group[node]]
                previous_node = node
            else:
                prev_score = score
                score += self.rewards[node_to_group[node]]
                previous_node = node
        return score

    def energy(self):
        score = self.eval_path(self.state)
        energy = np.abs(score - 2024)
        return int(energy)

def print_result(km, ta, tb):
    km_ = f'{key_map["a"]},{key_map["b"]},{key_map["c"]}'
    trip_a_ = ','.join(trip_a)
    trip_b_ = ','.join(trip_b)
    return f"{km_},{trip_a_},{trip_b_}"

if __name__ == '__main__':
    key_map = {'a': 1, 'b': 3, 'c': 2}
    g = nx.Graph(remapped_board)
    min_trip_len = 100
    trip_a = []
    trip_b = []
    e = 10000000

    # TRIP A
    for i in range(10):
        seed_path = nx.all_simple_paths(g, 'a1', 'f6')
        tsp = KnightsMoveProblem(next(seed_path), key_map)
        t_a, t_e = tsp.anneal()
        if (t_e == 0) and (len(t_a) < min_trip_len):
            trip_a = t_a
            e = t_e
            min_trip_len = len(t_a)
            break

    for i in range(10):
        seed_path = nx.all_simple_paths(g, 'a6', 'f1')
        tsp = KnightsMoveProblem(next(seed_path), key_map)
        t_b, t_e = tsp.anneal()
        if (t_e == 0) and (len(t_b) < min_trip_len):
            trip_b = t_b
            e = t_e
            min_trip_len = len(t_b)
            break

    print(len(trip_a) + len(trip_b))
    print(trip_a + trip_b)
    print_result(key_map, trip_a, trip_b)