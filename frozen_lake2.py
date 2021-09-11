import sys
from contextlib import closing

import numpy as np
from io import StringIO

from gym import utils
from gym.envs.toy_text import discrete


ACTIONS = ["DOWN", "RIGHT", "DOWNRIGHT"]

MAPS = {
    "5x5": [
        "SPPPP",
        "FFFFP",
        "FFFFP",
        "FFFFP",
        "HHHHG",
    ],
}


class FrozenLake2Env(discrete.DiscreteEnv):
    """
    Winter is here. You and your friends were tossing around a frisbee at the
    park when you made a wild throw that left the frisbee out in the middle of
    the lake. The water is mostly frozen, but there are a few holes where the
    ice has melted. If you step into one of those holes, you'll fall into the
    freezing water. At this time, there's an international frisbee shortage, so
    it's absolutely imperative that you navigate across the lake and retrieve
    the disc. However, the ice is slippery, so you won't always move in the
    direction you intend.
    The surface is described using a grid like the following

        SFFF
        FHFH
        FFFH
        HFFG

    S : starting point, safe, not slippery
    P : path, safe, not slippery
    F : frozen surface, safe, slippery
    H : hole, fall to your doom
    G : goal, where the frisbee is located

    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, and zero otherwise.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="5x5", slip_factor=0.2):
        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)

        nA = 3
        nS = nrow * ncol

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            """Get an integer representation of the state"""
            return row*ncol + col

        def inc(row, col, a):
            if ACTIONS[a] == "DOWN":
                row = min(row + 1, nrow - 1)
            elif ACTIONS[a] == "RIGHT":
                col = min(col + 1, ncol - 1)
            elif ACTIONS[a] == "DOWNRIGHT":
                # Don't move if we are in the last row or column
                if row != nrow -1 and col != ncol - 1:
                    row = min(row + 1, nrow - 1)
                    col = min(col + 1, ncol - 1)
            return (row, col)

        def update_probability_matrix(row, col, action):
            newrow, newcol = inc(row, col, action)
            newstate = to_s(newrow, newcol)
            newletter = desc[newrow, newcol]
            done = bytes(newletter) in b'GH'
            reward = float(newletter == b'G')
            return newstate, reward, done

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(nA):
                    letter = desc[row, col]
                    if letter in b'GH':
                        P[s][a].append((1.0, s, 0, True))
                    else:
                        # If on a slippery frozen square, take a random action with prob slip_factor
                        if letter == b'F':
                            # Take the given action with prob 1-slip_factor
                            P[s][a].append((
                                1. - slip_factor , *update_probability_matrix(row, col, a)
                            ))
                            # Take a random action with prob slip_factor
                            for other_action in [i for i in range(nA) if i != a]:
                                P[s][a].append((
                                    slip_factor / (nA - 1),
                                    *update_probability_matrix(row, col, other_action)
                                ))
                        else:
                            P[s][a].append((
                                1., *update_probability_matrix(row, col, a)
                            ))

        super(FrozenLake2Env, self).__init__(nS, nA, P, isd)

    def render(self, mode='human'):
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
