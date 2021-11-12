"""
Q-Learning implementation.
"""
import random
from copy import deepcopy
import numpy as np
from utils import set_random_seed

class QLearning:
    """Initialise QLearning model, setting all learning params"""
    def __init__(self,
                 env,
                 seed=None,
                 lr=0.1,
                 lr_decay=0.85,
                 lr_decay_freq=200000,
                 gamma=1,
                 epsilon=0.1,
                 total_steps=2000000,
                 n_rollout_steps=1000,
                 use_crm=True,
                 use_rs=False,
                 print_freq=50000,
                 eval_freq=5000,
                 num_eval_eps=20,
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
        self.n_rollout_steps = n_rollout_steps
        self.use_crm = use_crm
        self.use_rs = use_rs
        self.print_freq = print_freq
        self.eval_freq = eval_freq
        self.num_eval_eps = num_eval_eps
        # For evaluation
        self.env_copy = deepcopy(env)
        self.q_init = q_init
        self.q = {}

    def learn(self):
        """
        Run qlearning. Adapted from
        https://github.com/RodrigoToroIcarte/reward_machines.
        """
        policy_info = {"samples": [], "updates": [], "rewards": []}

        step = 0
        updates = 0
        while step < self.total_steps:
            # Reset to initial (env and RM) state
            s = tuple(self.env.reset())
            if s not in self.q:
                self.q[s] = [self.q_init] * self.env.action_space.n
            for _ in range(self.n_rollout_steps):
                # Get epsilon-greedy action
                a, _ = self.predict(s, deterministic=False)
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
                        experiences.append((tuple(_s), _a, _r, tuple(_sn), _done))
                else:
                    # Include only the current experience (standard q-learning)
                    experiences.append((s, a, r, sn, done))

                for _s, _a, _r, _sn, _done in experiences:
                    if _s not in self.q:
                        self.q[_s] = [self.q_init] * self.env.action_space.n
                    if _sn not in self.q:
                        self.q[_sn] = [self.q_init] * self.env.action_space.n
                    if _done:
                        _delta = _r - self.q[_s][_a]
                    else:
                        _delta = _r + self.gamma * np.max(self.q[_sn]) - self.q[_s][_a]
                    self.q[_s][_a] += self.lr * _delta
                    updates += 1

                if step % self.eval_freq == 0:
                    # Evaluate the current policy
                    policy_info["samples"].append(step)
                    policy_info["updates"].append(updates)
                    # policy_info["rewards"].append(reward_total / self.eval_freq)
                    # policy_info["rewards"].append(reward_total / eval_eps)
                    policy_info["rewards"].append(self.eval_policy())
                if done or step == self.total_steps:
                    break
                s = sn
        return policy_info

    def predict(self, s, deterministic=False):
        """
        Select an action, using epsilon-greedy when deterministic=False, and selecting an action
        with the max q-value for state otherwise.
        """
        if not deterministic and (random.random() < self.epsilon):
            a = self.env.action_space.sample()
        else:
            # Randomly select among best actions based on q-values
            a = random.choice(np.flatnonzero(self.q[s] == np.max(self.q[s])))

        # stable-baselines3 returns an action and a model state for recurrent models.
        return a, None

    def eval_policy(self):
        """Calculate the mean return of the policy over self.num_eval_eps episodes."""
        total_rewards = 0.
        for _ in range(self.num_eval_eps):
            s = tuple(self.env_copy.reset())
            # Use more rollout steps in evaluation as the policies may have loops and require
            # some stochasticity.
            for step in range(self.n_rollout_steps * 3):
                # If no q-values for this state, initialise them
                if s not in self.q:
                    self.q[s] = [self.q_init] * self.env.action_space.n
                # Must allow for random action to avoid trying to always move off the grid (and
                # thus staying in the same place)
                a, _ = self.predict(s, deterministic=True)
                # Take step in env
                sn, r, done, _ = self.env_copy.step(a)
                total_rewards += (self.gamma ** step) * r
                if done:
                    break
                s = tuple(sn)
        return total_rewards / self.num_eval_eps