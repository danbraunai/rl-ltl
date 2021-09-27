"""
QT-Learning implementation as in Harada1997 https://www.aaai.org/Papers/AAAI/1997/AAAI97-090.pdf.
"""
import random
import numpy as np
from utils import set_random_seed

class QTLearning:
    """Initialise QTLearning model, setting all learning params"""
    def __init__(self,
                 env,
                 seed=None,
                 lr=0.5,
                 gamma=1,
                 epsilon=0.1,
                 n_episodes=1000,
                 horizon=10,
                 use_crm=True,
                 use_rs=True,
                 print_freq=10000,
                 **_):
        # Set global seed for reproducibility
        if seed is not None:
            set_random_seed(seed)

        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_episodes = n_episodes
        self.horizon = horizon
        self.use_crm = use_crm
        self.use_rs = use_rs
        self.print_freq = print_freq
        # q dict will have the form (time_rem, state): q-val
        self.q = {}

    def learn(self):
        """
        Run qlearning, updating self.q after each step. Adapted from
        https://github.com/RodrigoToroIcarte/reward_machines.
        """
        reward_total = 0
        step = 0
        num_episodes = 0
        for _ in range(self.n_episodes):
            s = tuple(self.env.reset())
            # Iterate through steps remaining
            for t in range(self.horizon, 0, -1):
                if (t, s) not in self.q:
                    self.q[(t, s)] = [0] * self.env.action_space.n
                    # self.q[(t, s)] = np.zeros(self.env.action_space.n)
                # Get epsilon-greedy action
                a, _ = self.predict((t, s), deterministic=False)
                # Take step in envrionment
                sn, r, done, info = self.env.step(a)
                sn = tuple(sn)

                # Updating the q-values
                experiences = []
                if self.use_crm:
                    # Adding counterfactual experience for all combinations of reward machine
                    # state and timestep remaining (this will already include shaped rewards
                    # if use_rs=True)
                    for _s, _a, _r, _sn, _done in info["crm-experience"]:
                        for t_rem in range(1, self.horizon + 1):
                            experiences.append((t_rem, tuple(_s), _a, _r, tuple(_sn), _done))
                elif self.use_rs:
                    # Include only the current experince for current t but shape the reward
                    experiences = [(t, s, a, info["rs-reward"], sn, done)]
                else:
                    # Include only the current experience for the current timestep (standard
                    # q-learning on extended SxT space)
                    experiences = [(t, s, a, r, sn, done)]

                for _t, _s, _a, _r, _sn, _done in experiences:
                    # if _s not in Q: Q[_s] = dict([(b,q_init) for b in actions])
                    if (_t, _s) not in self.q:
                        self.q[(_t, _s)] = [0] * self.env.action_space.n
                    if (_t, _sn) not in self.q:
                        self.q[(_t, _sn)] = [0] * self.env.action_space.n
                    # Don't bootstrap when done or if only one timestep remaining
                    if _done or _t == 1:
                        _delta = _r - self.q[(_t, _s)][_a]
                    else:
                        # _delta = _r + self.gamma * get_qmax(Q, _sn,actions,q_init) - Q[_s][_a]
                        _delta = (
                            _r + self.gamma * np.max(self.q[(_t - 1, _sn)]) - self.q[(_t, _s)][_a]
                        )
                    self.q[(_t, _s)][_a] += self.lr * _delta

                # moving to the next state
                reward_total += r
                step += 1
                if step % self.print_freq == 0:
                    print("steps", step)
                    print("episodes", num_episodes)
                    print("total reward", reward_total)
                    reward_total = 0
                if done:
                    num_episodes += 1
                    break
                s = sn

    def predict(self, info, deterministic=False):
        """
        Select an action, using epsilon-greedy when deterministic=False, and selecting an action
        with the max q-value for state otherwise.

        Args:
            info: Tuple of the form (timestep, state).
        """
        t, s = info
        if not deterministic and (random.random() < self.epsilon):
            a = self.env.action_space.sample()
        else:
            # Randomly select among best actions based on q-values
            a = random.choice(np.flatnonzero(self.q[(t, s)] == np.max(self.q[(t, s)])))

        # stable-baselines3 returns an action and a model state for recurrent models.
        return a, None
