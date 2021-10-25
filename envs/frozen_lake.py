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

from reward_machines.rm_environment import RewardMachineEnv


def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution.
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()

ACTIONS_RIGHT = ["RIGHT", "DOWNRIGHT", "DOWN"]
ACTIONS = ["RIGHT", "DOWNRIGHT", "DOWN", "DOWNLEFT", "LEFT", "UPLEFT", "UP", "UPRIGHT"]

# Predefined maps and objects
MAPS = {
    "5x5": {
        "env": [
            "SPPPP",
            "IIIIP",
            "IIIIP",
            "IIIIP",
            "HHHHP",
        ],
        "objects_t1": {
            (4, 4): "f",
        },
        "objects_t2": {
            (2, 3): "r",
            (4, 4): "f",
        },
        "objects_t3": {
            (3, 3): "r",
            (2, 2): "s",
            (0, 1): "t",
            (4, 4): "f",
        },
    },
}



class FrozenLake(Env):
    """
    This environment was adapted from the OpenAI gym FrozenLake environment
    https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py. It follows a
    similar structure (and some of the same methods) as the environments in
    https://github.com/RodrigoToroIcarte/reward_machines.

    Winter is here. You and your friends were tossing around a frisbee on a (mostly) frozen lake
    when your friend fell through the ice trying to take an epic catch. Luckily, someone left
    some rope on the lake for this exact purpose. You must grab the rope, and go over to your
    friend to pull them out of the water with it. There are a few holes in the lake where the
    ice is melted. If you step into one of those holes, you will fall in and be stranded like
    your friend. Also, the ice is slippery, so you won't always move in the direction you intend.
    The surface itself is described using a grid like the following

        SPPPP
        IIIIP
        IIIIP
        IIIIP
        HHHHP

    S : starting point, safe, not slippery
    P : path, safe, not slippery
    I : ice, safe, slippery
    H : hole, fall to your doom

    The rope and your friend are objects referenced by zero-indexed grid locations.
    E.g. an object entry of (1,1): "r" means that a rope is on the second row and column,
    (2,3): "f" means that a friend is at the thrid row and forth column.

    The episode ends when you pull your friend out of the hole with the rope or fall in a hole.
    You receive a reward of 1 if you succeed, and zero otherwise.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, map_name="5x5", obj_name="objects_v0", slip=0.2, seed=None, all_acts=True):
        """
        Setup environment, including saving all possible state-transitions and their
        probability.

        Args:
            map_name: String corresponding to a key in MAPS to obtain the surface.
            obj_name: String corresponding to a key in MAPS[map_name] to obtain the objects.
            slip: Agent takes a random action with prob slip.
            seed: Seed the RNG for reproducability.
            all_acts: Use ACTIONS_ALL if True, otherwise use ACTIONS.
        """
        self.desc = np.asarray(MAPS[map_name]["env"], dtype='c')
        self.nrow, self.ncol = self.desc.shape
        # TODO: allow random generation of objects
        self.objects = MAPS[map_name][obj_name]
        self.slip = slip
        self.actions = ACTIONS if all_acts else ACTIONS_RIGHT

        # For rendering
        self.lastaction = None
        self.nA = len(self.actions)
        self.nS = self.nrow * self.ncol

        # Current positions of agent
        self.s = tuple(np.argwhere(self.desc == b"S")[0])
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Box(
            low=0, high=max([self.nrow, self.ncol]), shape=(2,), dtype=np.uint8
        )

        self.seed(seed)
        self.P = self.get_transitions()

    def _get_new_position(self, row, col, a):
        """
        Move the agent. If the new direction is outside of the grid, leave the agent where she is.
        """
        if self.actions[a] == "RIGHT":
            col = min(col + 1, self.ncol - 1)
        elif self.actions[a] == "DOWNRIGHT":
            # Don't move if we are in the last row or column
            if row != self.nrow -1 and col != self.ncol - 1:
                row += 1
                col += 1
        elif self.actions[a] == "DOWN":
            row = min(row + 1, self.nrow - 1)
        elif self.actions[a] == "DOWNLEFT":
            # Don't move if we are in the last row or first column
            if row != self.nrow -1 and col != 0:
                row += 1
                col -= 1
        elif self.actions[a] == "LEFT":
            col = max(col - 1, 0)
        elif self.actions[a] == "UPLEFT":
            # Don't move if we are in the first row or first column
            if row != 0 and col != 0:
                row -= 1
                col -= 1
        elif self.actions[a] == "UP":
            row = max(row - 1, 0)
        elif self.actions[a] == "UPRIGHT":
            # Don't move if we are in the first row or last column
            if row != 0 and col != self.ncol - 1:
                row -= 1
                col += 1
        return (row, col)

    def _to_s(self, row, col):
        """Get an integer representation of the state"""
        return row * self.ncol + col

    def _update_probability_matrix(self, row, col, action):
        newrow, newcol = self._get_new_position(row, col, action)
        newletter = self.desc[newrow, newcol]
        # Env only "done" when fall down hole, other done statuses are handled by reward machine
        done = (bytes(newletter) in b'H') or (newrow, newcol) == (self.nrow - 1, self.ncol - 1)
        # All rewards come from reward machine
        reward = 0
        return (newrow, newcol), reward, done

    def get_label(self, state):
        try:
            label = self.objects[state]
        except KeyError:
            label = ""
        return label

    def get_transitions(self):
        """
        Stores the possible transitions after taking an action in a state.
        A move on a slippery surface will move to the desired state self.slip percent of the time
        and move in a uniformly random direction otherwise.
        Returns:
            P: Dict. P[state][action] == [(probability, nextstate, reward, done, label), ...],
            where state and nextstate are of the form (row, col).
        """

        P = {}
        for row in range(self.nrow):
            for col in range(self.ncol):
                P[(row, col)] = {a: [] for a in range(self.nA)}
                for a in range(self.nA):
                    letter = self.desc[row, col]
                    if (letter in b'H' or 
                            (self.actions == ACTIONS_RIGHT and 
                            (row, col) == (self.nrow - 1, self.ncol - 1))):
                        # If fallen in hole or at bottom right corner of env with only ACTIONS_RIGHT
                        # available (and thus can't move),
                        # loop transition to the same state and tag that we are done
                        P[(row, col)][a].append((1.0, (row, col), 0, True))
                    elif letter == b'I':
                        # If on a slippery frozen square, take a random action with prob slip
                        # Store the given action with prob 1-slip + slip/self.nA
                        P[(row, col)][a].append((
                            1. - self.slip + self.slip / self.nA,
                            *self._update_probability_matrix(row, col, a)
                        ))
                        # Store all other actions with prob slip / self.nA
                        for other_action in [i for i in range(self.nA) if i != a]:
                            P[(row, col)][a].append((
                                self.slip / self.nA,
                                *self._update_probability_matrix(row, col, other_action)
                            ))
                    else:
                        P[(row, col)][a].append((1., *self._update_probability_matrix(row, col, a)))

        return P

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        return [seed]

    def reset(self):
        self.s = tuple(np.argwhere(self.desc == b"S")[0])
        self.lastaction = None
        return self.s

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]
        self.s = s
        self.lastaction = a
        return (s, r, d, {"prob": p})

    def get_events(self):
        """
        Returns the string with the propositions that are True in this state
        """
        try:
            props = self.objects[self.s]
        except KeyError:
            # No object in this position
            props = ""
        return props

    def render(self, mode='human'):
        """Render the game grid."""
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        # row, col = self.s // self.ncol, self.s % self.ncol
        row, col = self.s
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(self.actions[self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()


class FrozenLakeRMEnv(RewardMachineEnv):
    def __init__(self, rm_files, **kwargs):
        env = FrozenLake(**kwargs)
        super().__init__(env, rm_files)
