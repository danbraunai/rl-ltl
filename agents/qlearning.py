"""
Q-Learning implementation. Format consistent with OpenAI stable-baselines3.
"""
import numpy as np
import random
from stable_baselines3.common.utils import set_random_seed


def learn(env,
          seed=None,
          lr=0.5,
          gamma=1,
          epsilon=0.1,
          n_episodes=1000,
          n_rollout_steps=100):

    set_random_seed(seed)

    # Initialise q-values
    q = np.zeros([env.observation_space.n, env.action_space.n])

    for ep in range(n_episodes):
        state = env.reset()
        for step in range(n_rollout_steps):
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                # Randomly select among best actions
                action = random.choice(np.flatnonzero(q[state] == np.max(q[state])))

            new_state, reward, done, info = env.step(action)

            # Update q-vals
            q[state][action] += lr * (reward + gamma * np.max(q[new_state]) - q[state][action])

            state = new_state
            if done:
                break
    return q