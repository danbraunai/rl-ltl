"""
A textual grid-world environment adapted from the OpenAI gym FrozenLake environment
https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py.
"""
import sys
from contextlib import closing
from io import StringIO

import numpy as np

from gym import utils, Env, spaces
from gym.utils import seeding


def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution.
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()

ACTIONS = ["RIGHT", "DOWNRIGHT", "DOWN"]

MAPS = {
    "5x5": [
        "SPPPP",
        "FFFFP",
        "FFFFP",
        "FFFFP",
        "HHHHG",
    ],
}


class FrozenLake(Env):
    """
    This environment was adapted from the OpenAI gym FrozenLake environment
    https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py. It follows a
    similar structure to the environments in https://github.com/RodrigoToroIcarte/reward_machines.

    Winter is here. You and your friends were tossing around a frisbee at the
    park when you made a wild throw that left the frisbee out in the middle of
    the lake. The water is mostly frozen, but there are a few holes where the
    ice has melted. If you step into one of those holes, you'll fall into the
    freezing water. At this time, there's an international frisbee shortage, so
    it's absolutely imperative that you navigate across the lake and retrieve
    the disc. However, the ice is slippery, so you won't always move in the
    direction you intend.
    The surface is described using a grid like the following

        SPPPP
        FFFFP
        FFFFP
        FFFFP
        HHHHG

    S : starting point, safe, not slippery
    P : path, safe, not slippery
    F : frozen surface, safe, slippery
    H : hole, fall to your doom
    G : goal, the other side of the lake

    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, and zero otherwise.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, map_name="5x5", slip=0.2):
        """
        Setup environment, including saving all possible state-transitions and their
        probability.

        Args:
            map_name: String corresponding to a key in MAPS
            slip: Agent takes a random action with prob slip.
        """
        self.desc = np.asarray(MAPS[map_name], dtype='c')
        self.nrow, self.ncol = self.desc.shape
        self.slip = slip

        # For rendering
        self.lastaction = None
        self.nA = len(ACTIONS)
        self.nS = self.nrow * self.ncol

        # Assign uniform distribution over initial states (those encoded by "S" in desc)
        self.isd = np.array(self.desc == b'S').astype('float64').ravel()
        self.isd /= self.isd.sum()

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self.seed()
        # Initial state
        self.s = categorical_sample(self.isd, self.np_random)
        self.P = self.get_model()

    def _get_new_position(self, row, col, a):
        if ACTIONS[a] == "DOWN":
            row = min(row + 1, self.nrow - 1)
        elif ACTIONS[a] == "RIGHT":
            col = min(col + 1, self.ncol - 1)
        elif ACTIONS[a] == "DOWNRIGHT":
            # Don't move if we are in the last row or column
            if row != self.nrow -1 and col != self.ncol - 1:
                row = min(row + 1, self.nrow - 1)
                col = min(col + 1, self.ncol - 1)
        return (row, col)

    def _to_s(self, row, col):
        """Get an integer representation of the state"""
        return row * self.ncol + col

    def _update_probability_matrix(self, row, col, action):
        newrow, newcol = self._get_new_position(row, col, action)
        newstate = self._to_s(newrow, newcol)
        newletter = self.desc[newrow, newcol]
        done = bytes(newletter) in b'GH'
        reward = float(newletter == b'G')
        return newstate, reward, done

    def get_model(self):
        """
        Stores the possible transitions after taking an action in a state.
        Returns:
            P: Dict of the form P[state][action] == [(probability, nextstate, reward, done), ...].
        """
        P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        for row in range(self.nrow):
            for col in range(self.ncol):
                s = self._to_s(row, col)
                for a in range(self.nA):
                    letter = self.desc[row, col]
                    if letter in b'GH':
                        P[s][a].append((1.0, s, 0, True))
                    else:
                        # If on a slippery frozen square, take a random action with prob slip
                        if letter == b'F':
                            # Store the given action with prob 1-slip + slip/self.nA
                            P[s][a].append((
                                1. - self.slip + self.slip / self.nA,
                                *self._update_probability_matrix(row, col, a)
                            ))
                            # Store all other actions with prob slip / self.nA
                            for other_action in [i for i in range(self.nA) if i != a]:
                                P[s][a].append((
                                    self.slip / self.nA,
                                    *self._update_probability_matrix(row, col, other_action)
                                ))
                        else:
                            P[s][a].append((
                                1., *self._update_probability_matrix(row, col, a)
                            ))
        return P

    def get_events(self):
        raise NotImplementedError

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction = None
        return int(self.s)

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]
        self.s = s
        self.lastaction = a
        return (int(s), r, d, {"prob": p})

    def render(self, mode='human'):
        """Render the game grid."""
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(ACTIONS[self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
