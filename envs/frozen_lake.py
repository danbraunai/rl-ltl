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

ACTIONS = ["RIGHT", "DOWNRIGHT", "DOWN"]

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
        "objects_v0": {
            (0,4): ["r"],
            (4,4): ["f"],
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
    E.g. an object entry of (1,1): ["r"] means that a rope is on the second row and column,
    and (2,3): ["f"] means that a friend is at the thrid row and forth column.

    The episode ends when you pull your friend out of the hole with the rope or fall in a hole.
    You receive a reward of 1 if you succeed, and zero otherwise.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, map_name="5x5", obj_name="objects_v0", slip=0.2, seed=None):
        """
        Setup environment, including saving all possible state-transitions and their
        probability.

        Args:
            map_name: String corresponding to a key in MAPS to obtain the surface.
            obj_name: String corresponding to a key in MAPS[map_name] to obtain the objects.
            slip: Agent takes a random action with prob slip.
        """
        self.desc = np.asarray(MAPS[map_name]["env"], dtype='c')
        self.nrow, self.ncol = self.desc.shape
        # TODO: allow random generation of objects
        # self.objects = {self._to_s(*k): v for k, v in MAPS[map_name][obj_name].items()}
        self.objects = MAPS[map_name][obj_name]
        self.slip = slip

        # For rendering
        self.lastaction = None
        self.nA = len(ACTIONS)
        self.nS = self.nrow * self.ncol

        # Assign uniform distribution over initial states (those encoded by "S" in desc)
        # self.isd = np.array(self.desc == b'S').astype('float64').ravel()
        # self.isd /= self.isd.sum()
        # Current positions of agent
        self.s = tuple(np.argwhere(self.desc == b'S')[0])
        self.action_space = spaces.Discrete(self.nA)
        # self.observation_space = spaces.Discrete(self.nS)
        self.observation_space = spaces.Box(
            low=0, high=max([self.nrow, self.ncol]), shape=(2,), dtype=np.uint8
        )

        self.seed(seed)
        # The integer index representing the state of the agent
        # self.s = categorical_sample(self.isd, self.np_random)
        self.P = self.get_transitions()

        self.s = (1,1)
        self.step(1)

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
        # newstate = self._to_s(newrow, newcol)
        newletter = self.desc[newrow, newcol]
        # Env only "done" when fall down hole, other done statuses are handled by reward machine
        done = bytes(newletter) in b'H'
        # reward = float(newletter == b'G')
        # All rewards come from reward machine
        reward = 0
        # return newstate, reward, done
        return (newrow, newcol), reward, done

    def get_transitions(self):
        """
        Stores the possible transitions after taking an action in a state.
        Returns:
            P: Dict of the form P[state][action] == [(probability, nextstate, reward, done), ...],
            where state and nextstate are of the form (row,col).
        """

        P = {}
        for row in range(self.nrow):
            for col in range(self.ncol):
                P[(row,col)] = {a: [] for a in range(self.nA)}
                # s = self._to_s(row, col)
                for a in range(self.nA):
                    letter = self.desc[row, col]
                    # If fallen in hole, loop transition to the same state
                    # if letter in b'GH':
                    if letter in b'H':
                        P[(row,col)][a].append((1.0, (row,col), 0, True))
                    else:
                        # If on a slippery frozen square, take a random action with prob slip
                        if letter == b'I':
                            # Store the given action with prob 1-slip + slip/self.nA
                            P[(row,col)][a].append((
                                1. - self.slip + self.slip / self.nA,
                                *self._update_probability_matrix(row, col, a)
                            ))
                            # Store all other actions with prob slip / self.nA
                            for other_action in [i for i in range(self.nA) if i != a]:
                                P[(row,col)][a].append((
                                    self.slip / self.nA,
                                    *self._update_probability_matrix(row, col, other_action)
                                ))
                        else:
                            P[(row,col)][a].append((
                                1., *self._update_probability_matrix(row, col, a)
                            ))
        return P

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        return [seed]

    def reset(self):
        # self.s = categorical_sample(self.isd, self.np_random)
        self.s = tuple(np.argwhere(self.desc == b'S')[0])
        self.lastaction = None
        # return int(self.s)
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
        ret = ""
        if self.s in self.objects:
            ret += "&".join(self.objects[self.s])
        return ret

    def get_features(self):
        """
        Returns the features of the current state (i.e., the location of the agent)
        """
        x,y = self.agent
        return np.array([x,y])

    def render(self, mode='human'):
        """Render the game grid."""
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        # row, col = self.s // self.ncol, self.s % self.ncol
        row, col = self.s
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


class FrozenLakeRMEnv(RewardMachineEnv):
    def __init__(self, **kwargs):
        env = FrozenLake(**kwargs)
        rm_files = ["./envs/rm1_frozen_lake.txt"]
        super().__init__(env, rm_files)

    # def render(self, mode='human'):
    #     if mode == 'human':
    #         # commands
    #         str_to_action = {"w":0,"d":1,"s":2,"a":3}

    #         # play the game!
    #         done = True
    #         while True:
    #             if done:
    #                 print("New episode --------------------------------")
    #                 obs = self.reset()
    #                 print("Current task:", self.rm_files[self.current_rm_id])
    #                 self.env.show()
    #                 print("Features:", obs)
    #                 print("RM state:", self.current_u_id)
    #                 print("Events:", self.env.get_events())

    #             print("\nAction? (WASD keys or q to quite) ", end="")
    #             a = input()
    #             print()
    #             if a == 'q':
    #                 break
    #             # Executing action
    #             if a in str_to_action:
    #                 obs, rew, done, _ = self.step(str_to_action[a])
    #                 self.env.show()
    #                 print("Features:", obs)
    #                 print("Reward:", rew)
    #                 print("RM state:", self.current_u_id)
    #                 print("Events:", self.env.get_events())
    #             else:
    #                 print("Forbidden action")
    #     else:
    #         raise NotImplementedError

    def test_optimal_policies(self, num_episodes, epsilon, gamma):
        """
        This code computes optimal policies for each reward machine and evaluates them using epsilon-greedy exploration

        PARAMS
        ----------
        num_episodes(int): Number of evaluation episodes
        epsilon(float):    Epsilon constant for exploring the environment
        gamma(float):      Discount factor

        RETURNS
        ----------
        List with the optimal average-reward-per-step per reward machine
        """
        S,A,L,T = self.env.get_model()
        print("\nComputing optimal policies... ", end='', flush=True)
        optimal_policies = [value_iteration(S,A,L,T,rm,gamma) for rm in self.reward_machines]
        print("Done!")
        optimal_ARPS = [[] for _ in range(len(optimal_policies))]
        print("\nEvaluating optimal policies.")
        for ep in range(num_episodes):
            if ep % 1000 == 0 and ep > 0:
                print("%d/%d"%(ep,num_episodes))
            self.reset()
            s = tuple(self.obs)
            u = self.current_u_id
            rm_id = self.current_rm_id
            rewards = []
            done = False
            while not done:
                a = random.choice(A) if random.random() < epsilon else optimal_policies[rm_id][(s,u)]
                _, r, done, _ = self.step(a)
                rewards.append(r)
                s = tuple(self.obs)
                u = self.current_u_id
            optimal_ARPS[rm_id].append(sum(rewards)/len(rewards))
        print("Done!\n")

        return [sum(arps)/len(arps) for arps in optimal_ARPS]