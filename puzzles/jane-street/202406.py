from simanneal import Annealer
from itertools import compress
import random
import numpy as np
import pandas as pd
import networkx as nx

usa_graph = {
    'washington': ['idaho', 'oregon'],
    'oregon': ['washington', 'idaho', 'california', 'nevada'],
    'idaho': ['washington', 'oregon', 'montana', 'wyoming', 'utah', 'nevada'],
    'montana': ['idaho', 'northdakota', 'southdakota', 'wyoming'],
    'northdakota': ['montana', 'minnesota', 'southdakota'],
    'minnesota': ['northdakota', 'wisconsin', 'iowa', 'southdakota'],
    'wisconsin': ['minnesota', 'iowa', 'illinois', 'michigan'],
    'michigan': ['wisconsin', 'illinois', 'indiana', 'ohio'],
    'illinois': ['wisconsin', 'indiana', 'kentucky', 'missouri', 'iowa'],
    'indiana': ['illinois', 'michigan', 'ohio', 'kentucky'],
    'ohio': ['michigan', 'indiana', 'kentucky', 'westvirginia', 'pennsylvania'],
    'pennsylvania': ['ohio', 'westvirginia', 'maryland', 'delaware', 'newjersey', 'newyork'],
    'newyork': ['pennsylvania', 'newjersey', 'vermont', 'massachusetts', 'connecticut'],
    'vermont': ['newyork', 'newhampshire'],
    'newhampshire': ['vermont', 'maine'],
    'maine': ['newhampshire'],
    'massachusetts': ['newyork', 'connecticut', 'rhodeisland', 'newhampshire', 'vermont'],
    'connecticut': ['newyork', 'massachusetts', 'rhodeisland'],
    'rhodeisland': ['massachusetts', 'connecticut'],
    'newjersey': ['pennsylvania', 'delaware', 'newyork'],
    'delaware': ['pennsylvania', 'maryland', 'newjersey'],
    'maryland': ['delaware', 'pennsylvania', 'westvirginia', 'virginia'],
    'westvirginia': ['pennsylvania', 'ohio', 'kentucky', 'virginia'],
    'virginia': ['maryland', 'westvirginia', 'kentucky', 'tennessee', 'northcarolina'],
    'kentucky': ['ohio', 'indiana', 'illinois', 'missouri', 'tennessee', 'virginia', 'westvirginia'],
    'tennessee': ['kentucky', 'virginia', 'northcarolina', 'georgia', 'alabama', 'mississippi', 'arkansas', 'missouri'],
    'northcarolina': ['virginia', 'tennessee', 'georgia', 'southcarolina'],
    'southcarolina': ['northcarolina', 'georgia'],
    'georgia': ['northcarolina', 'southcarolina', 'florida', 'alabama', 'tennessee'],
    'florida': ['georgia', 'alabama'],
    'alabama': ['mississippi', 'tennessee', 'georgia', 'florida'],
    'mississippi': ['arkansas', 'louisiana', 'tennessee', 'alabama'],
    'louisiana': ['texas', 'arkansas', 'mississippi'],
    'arkansas': ['missouri', 'tennessee', 'mississippi', 'louisiana', 'texas', 'oklahoma'],
    'missouri': ['iowa', 'illinois', 'kentucky', 'tennessee', 'arkansas', 'oklahoma', 'kansas', 'nebraska'],
    'iowa': ['minnesota', 'wisconsin', 'illinois', 'missouri', 'nebraska', 'southdakota'],
    'nebraska': ['southdakota', 'iowa', 'missouri', 'kansas', 'colorado', 'wyoming'],
    'kansas': ['nebraska', 'missouri', 'oklahoma', 'colorado'],
    'oklahoma': ['kansas', 'missouri', 'arkansas', 'texas', 'newmexico', 'colorado'],
    'texas': ['oklahoma', 'arkansas', 'louisiana', 'newmexico'],
    'newmexico': ['colorado', 'oklahoma', 'texas', 'arizona', 'utah'],
    'colorado': ['wyoming', 'nebraska', 'kansas', 'oklahoma', 'newmexico', 'utah'],
    'utah': ['idaho', 'wyoming', 'colorado', 'arizona', 'nevada'],
    'nevada': ['idaho', 'utah', 'arizona', 'california'],
    'arizona': ['utah', 'newmexico', 'texas', 'california', 'nevada'],
    'california': ['oregon', 'nevada', 'arizona']
}

def read_us_population_table():
    with open("us_population_by_state.html") as f:
        eval_df = pd.read_html(f)[0]
    eval_df = eval_df[["State", "Population (2020)[86]"]].rename(
        columns={"State": "state", "Population (2020)[86]": "score"}
    )
    eval_df["state"] = eval_df["state"].str.lower().str.replace(" ", "")

    eval_df = eval_df.loc[
        (eval_df["state"] != "districtofcolumbia")
        & (eval_df["state"] != "unitedstates")
        ]
    return eval_df

def generate_inbound_moves(g, r, c):
    """
    Creates a set of moves that are one king's move away from (r,c)
    """
    R = len(g)
    C = len(g[0])
    k_moves = [
        [r - 1, c - 1],
        [r - 1, c],
        [r - 1, c + 1],
        [r, c - 1],
        [r, c + 1],
        [r + 1, c - 1],
        [r + 1, c],
        [r + 1, c + 1],
    ]
    f_k_moves = list(
        map(lambda x: ((x[0] < R) & (x[0] >= 0)) & ((x[1] < C) & (x[1] >= 0)), k_moves)
    )
    mv = list(compress(k_moves, f_k_moves))
    return mv

def find_state(g, positions, word, edits, found_word, idx):
    """
    Performs a depth first search to find whether a state has been embedded into a graph within 1 edit
    """
    if (len(word) == len(found_word)) and (edits <= 1):
        return True

    if idx >= len(word):
        return False

    if edits >= 2:
        return False

    i = positions[0]
    j = positions[1]
    if found_word == "":
        if g[i][j] != word[0]:
            edits += 1
        found_word = g[i][j]
        idx = 1

    mvs = generate_inbound_moves(g, i, j)
    for position in mvs:
        r = position[0]
        c = position[1]
        if (r >= 0) and (c >= 0) and (r < len(g)) and (c < len(g[0])):
            tmp_edits = edits
            if word[idx] != g[position[0]][position[1]]:
                tmp_edits = edits + 1
            tmp_word = found_word + g[position[0]][position[1]]
            tmp_idx = idx + 1
            res = find_state(g, position, word, tmp_edits, tmp_word, tmp_idx)
            if res:
                return res
            tmp_word = found_word
            tmp_idx = idx
            tmp_edits = edits
    return False

def dfs(g, word):
    """
    Iterates through every starting position and performs a depth first search
    """
    for r in range(len(g)):
        for c in range(len(g[0])):
            if find_state(g, [r, c], word, 0, "", 0):
                return True
    else:
        return False

def determine_states(df, s):
    """
    Iterates through every state to which state has been embedded
    """
    found = []
    states = df["state"].tolist()
    for state in states:
        if (dfs(s, state)):
            found.append(state)
    return found


def eval_states(df, states):
    """
    Evaluates the score of a subset of states
    """
    return df.loc[df["state"].isin(states)]["score"].sum()

def determine_coast_to_coast(s):
    '''
    Determines if a set of states contains a path from the Pacific Coast to the Atlantic Coast
    '''
    filtered_dict = {k:v for k,v in usa_graph.items() if k in s}

    for k,v in filtered_dict.items():
        f = []
        for s in v:
            if s in s:
                f.append(s)
        filtered_dict[k] = f

    G = nx.Graph(filtered_dict)
    pacific_states = [
        'washington',
        'oregon',
        'california'
    ]

    atlantic_coast_states = [
        'maine', 'newhampshire', 'massachusetts', 'rhodeisland', 'connecticut',
        'newyork', 'newjersey', 'pennsylvania', 'delaware', 'maryland',
        'virginia', 'northcarolina', 'southcarolina', 'georgia', 'florida'
    ]

    for ps in pacific_states:
        for at_s in atlantic_coast_states:
            try:
                return True if nx.shortest_path(G, ps, at_s) else False
            except:
                pass
    return False

class AlteredStatesProblem(Annealer):
    def __init__(self, state, data):
        super(AlteredStatesProblem, self).__init__(state)  # important!
        self.data = data
        self.TOTAL = self.data['score'].sum()

    def move_add_letter(self):
        """
        Generates a random letter and places it into a random position
        """
        initial_energy = self.energy()
        random_r = np.random.randint(0, len(self.state))
        random_c = np.random.randint(0, len(self.state[0]))
        letter = chr(np.random.randint(ord("a"), ord("a") + 26 - 1))
        self.state[random_r][random_c] = letter
        return self.energy() - initial_energy

    def move_swap_letter(self):
        """
        Swaps two letters within the matrix
        """
        initial_energy = self.energy()
        random_r_a = np.random.randint(0, len(self.state))
        random_c_a = np.random.randint(0, len(self.state[0]))

        random_r_b = np.random.randint(0, len(self.state))
        random_c_b = np.random.randint(0, len(self.state[0]))

        tmp = self.state[random_r_a][random_c_a]
        self.state[random_r_a][random_c_a] = self.state[random_r_b][random_c_b]
        self.state[random_r_b][random_c_b] = tmp
        return self.energy() - initial_energy

    def move(self):
        """
        Make the state changes

        1. Add New State
        2. Find new orientation of latest placed state
        """
        initial_energy = self.energy()
        states_to_choose = []
        direction = random.random()
        if direction > 0.5:
            return self.move_add_letter()
        else:
            return self.move_swap_letter()

    def energy(self):
        MAX_E = self.TOTAL
        if self.state:
            f = determine_states(self.data, self.state)
            T = eval_states(self.data, f)
        return MAX_E - T

class ConstrainedAlteredStatesProblem(AlteredStatesProblem):
    def __init__(self, state, data):
        super(ConstrainedAlteredStatesProblem, self).__init__(state, data)

    def energy(self):
        MAX_E = self.TOTAL
        if self.state:
            f = determine_states(self.data, self.state)

            if 'california' in f:
                T = 0

            else:
                T = eval_states(self.data, f)
                if (determine_coast_to_coast(f)):
                    T = T * 1.10
                else:
                    T = T * 0.60
        return (MAX_E - T)

if __name__ == '__main__':
    n = 5
    eval_df = read_us_population_table()
    test = [["" for _ in range(n)] for _ in range(n)]
    tsp = ConstrainedAlteredStatesProblem(test, eval_df)
    tsp.copy_strategy = "deepcopy"
    state, e = tsp.anneal()
    print(state)