"""
Q-Learning implementation.
"""
import random
import numpy as np
from stable_baselines3.common.utils import set_random_seed

class QLearning:
    """Initialise QLearning model, setting all learning params"""
    def __init__(self,
                 env,
                 seed=None,
                 lr=0.5,
                 gamma=1,
                 epsilon=0.1,
                 n_episodes=1000,
                 n_rollout_steps=100):
        # Set global seed for reproducibility
        if seed is not None:
            set_random_seed(seed)

        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_episodes = n_episodes
        self.n_rollout_steps = n_rollout_steps
        # Initialise q-values to zero
        self.q = np.zeros([self.env.observation_space.n, self.env.action_space.n])

    def learn(self):
        """Run qlearning, updating self.q after each step."""
        for _ in range(self.n_episodes):
            state = self.env.reset()
            for _ in range(self.n_rollout_steps):
                # Get epsilon-greedy action
                action, _ = self.predict(state, deterministic=False)

                if random.random() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    # Randomly select among best actions
                    action = random.choice(np.flatnonzero(self.q[state] == np.max(self.q[state])))

                new_state, reward, done, _ = self.env.step(action)

                # Update q-vals
                # If done, don't bother calculating the q-value of the new state (they will be 0)
                if done:
                    td_err = reward - self.q[state][action]
                else:
                    td_err = reward + self.gamma * np.max(self.q[new_state]) - self.q[state][action]

                self.q[state][action] += self.lr * td_err

                state = new_state
                if done:
                    break

    def predict(self, state, deterministic=False):
        """
        Select an action, using epsilon-greedy when deterministic=False, and selecting an action
        with the max q-value for state otherwise.
        """
        if not deterministic and (random.random() < self.epsilon):
            action = self.env.action_space.sample()
        else:
            # Randomly select among best actions based on q-values
            action = random.choice(np.flatnonzero(self.q[state] == np.max(self.q[state])))

        # stable-baselines3 returns an action and a model state for recurrent models.
        return action, None
