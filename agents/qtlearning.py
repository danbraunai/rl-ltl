"""
QT-Learning implementation as in Harada1997 https://www.aaai.org/Papers/AAAI/1997/AAAI97-090.pdf.
"""
import random
from copy import deepcopy
import numpy as np
from utils import set_random_seed

class QTLearning:
    """Initialise QTLearning model, setting all learning params"""
    def __init__(self,
                 env,
                 seed=None,
                 lr=0.1,
                 lr_decay=0.85,
                 lr_decay_freq=200000,
                 gamma=1,
                 epsilon=0.1,
                 total_steps=1000,
                 horizon=20,
                 eps_per_reset=3,
                 use_t=True,
                 use_crm=True,
                 eval_freq=100,
                 num_eval_eps=30,
                 q_init=1.0001,
                 **_):
        # Set global seed for reproducibility
        if seed is not None:
            set_random_seed(seed)

        self.env = env
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_decay_freq = lr_decay_freq
        self.gamma = gamma
        self.epsilon = epsilon
        self.total_steps = total_steps
        self.horizon = horizon
        self.eps_per_reset = eps_per_reset
        self.use_t = use_t
        self.use_crm = use_crm
        self.eval_freq = eval_freq
        self.num_eval_eps = num_eval_eps
        # For evaluation
        self.env_copy = deepcopy(env)
        self.q_init = q_init
        # q dict will have the form (time_rem, state): q-val
        self.q = {}

    def learn(self):
        """
        Run qlearning. Adapted from
        https://github.com/RodrigoToroIcarte/reward_machines.
        """
        policy_info = {"samples": [], "updates": [], "rewards": []}

        eps_until_reset = 0
        step = 0
        updates = 0
        while step < self.total_steps:
            # Reset the state (includes env and RM states) every self.eps_per_reset eps.
            if eps_until_reset == 0:
                s = tuple(self.env.reset())
                eps_until_reset = self.eps_per_reset

            # Iterate through steps remaining in horizon
            for t in range(self.horizon, 0, -1):
                if (t, s) not in self.q:
                    self.q[(t, s)] = [self.q_init] * self.env.action_space.n
                # Get epsilon-greedy action
                a, _ = self.predict((t, s), deterministic=False)
                # Take step in envrionment
                sn, r, done, info = self.env.step(a)
                sn = tuple(sn)
                step += 1

                # Reduce the learning rate
                if step % self.lr_decay_freq == 0:
                    self.lr *= self.lr_decay

                # Updating the q-values
                experiences = []
                if self.use_crm:
                    # Adding counterfactual experience for all reward machine states
                    for _s, _a, _r, _sn, _done in info["crm-experience"]:
                        if self.use_t:
                            # Also add counterfactual experiences for each timestep
                            for t_rem in range(self.horizon, 0, -1):
                                experiences.append((t_rem, tuple(_s), _a, _r, tuple(_sn), _done))
                        else:
                            experiences.append((t, tuple(_s), _a, _r, tuple(_sn), _done))
                else:
                    if self.use_t:
                        # Add counterfactual experiences for each timestep
                        for t_rem in range(self.horizon, 0, -1):
                            experiences.append((t_rem, s, a, r, sn, done))
                    else:
                        # Include only the current experience for the current timestep (standard
                        # q-learning on extended SxT space)
                        experiences.append((t, s, a, r, sn, done))

                for _t, _s, _a, _r, _sn, _done in experiences:
                    # if _s not in Q: Q[_s] = dict([(b,q_init) for b in actions])
                    if (_t, _s) not in self.q:
                        self.q[(_t, _s)] = [self.q_init] * self.env.action_space.n
                    if (_t - 1, _sn) not in self.q and _t > 1:
                        self.q[(_t - 1, _sn)] = [self.q_init] * self.env.action_space.n
                    # Don't bootstrap when done or if only one timestep remaining
                    if _done or _t == 1:
                        _delta = _r - self.q[(_t, _s)][_a]
                    else:
                        _delta = (
                            _r + self.gamma * np.max(self.q[(_t - 1, _sn)]) - self.q[(_t, _s)][_a]
                        )
                    self.q[(_t, _s)][_a] += self.lr * _delta
                    updates += 1

                if step % self.eval_freq == 0:
                    # Evaluate the current policy
                    policy_info["samples"].append(step)
                    policy_info["updates"].append(updates)
                    # policy_info["rewards"].append(reward_total / self.eval_freq)
                    # policy_info["rewards"].append(reward_total / eval_eps)
                    policy_info["rewards"].append(self.eval_policy())

                if done or step == self.total_steps:
                    # Reset to initial state on next episode if we are done.
                    eps_until_reset = 0
                    break
                s = sn
            eps_until_reset = max(0, eps_until_reset - 1)
        return policy_info

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

    def eval_policy(self):
        """Calculate the mean return of the policy over self.num_eval_eps episodes."""
        total_rewards = 0.
        for _ in range(self.num_eval_eps):
            s = tuple(self.env_copy.reset())
            # Iterate through steps remaining in horizon
            for t in range(self.horizon, 0, -1):
                if (t, s) not in self.q:
                    self.q[(t, s)] = [self.q_init] * self.env.action_space.n
                # Get epsilon-greedy action
                a, _ = self.predict((t, s), deterministic=True)
                # Take step in env
                sn, r, done, _ = self.env_copy.step(a)
                total_rewards += (self.gamma ** (self.horizon - t)) * r
                if done:
                    break
                s = tuple(sn)
        return total_rewards / self.num_eval_eps