"""
QT-Learning implementation as in Harada1997 https://www.aaai.org/Papers/AAAI/1997/AAAI97-090.pdf.
"""
import random
import numpy as np
from stable_baselines3.common.utils import set_random_seed

class QTLearning:
    """Initialise QTLearning model, setting all learning params"""
    def __init__(self,
                 env,
                 seed=None,
                 lr=0.5,
                 gamma=1,
                 epsilon=0.1,
                 n_episodes=1000,
                 horizon=10):
        # Set global seed for reproducibility
        if seed is not None:
            set_random_seed(seed)

        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_episodes = n_episodes
        self.horizon = horizon

        # Initialise q-values to zero
        self.q = np.zeros(
            (self.env.observation_space.n, self.env.action_space.n, self.horizon)
        )

    def learn(self):
        """Run qlearning, updating self.q for each timestep."""
        for _ in range(self.n_episodes):
            s = self.env.reset()
            for step in range(self.horizon):
                # Store the max timesteps remaining in episode, where t = 0 represents one timestep
                # remaining due to zero-indexing
                t = self.horizon - 1 - step
                # Get epsilon-greedy action
                a, _ = self.predict((s, t), deterministic=False)
                # Take step in envrionment
                new_s, rew, done, _ = self.env.step(a)
                # Update q-vals
                # If done, don't bother calculating the q-value of the new state (they will be 0)
                if done or t == 0:
                    td_err = rew - self.q[s, a, t]
                else:
                    td_err = rew + self.gamma * np.max(self.q[new_s, :, t-1]) - self.q[s, a, t]

                self.q[s, a, t] += self.lr * td_err

                s = new_s
                if done:
                    break

    def predict(self, info, deterministic=False):
        """
        Select an action, using epsilon-greedy when deterministic=False, and selecting an action
        with the max q-value for state otherwise.

        Args:
            info: Tuple of the form (state, timestep).
        """
        s, t = info
        if not deterministic and (random.random() < self.epsilon):
            a = self.env.action_space.sample()
        else:
            # Randomly select among best actions based on q-values
            a = random.choice(np.flatnonzero(self.q[s, :, t] == np.max(self.q[s, :, t])))

        # stable-baselines3 returns an action and a model state for recurrent models.
        return a, None
