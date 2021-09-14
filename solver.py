"""
This module contains functions for calculating optimal policies and state-values in an openai
gym environment.
"""

import numpy as np

def value_iter(env, gamma, threshold=1e-4):
    """
    Runs value iteration until convergence (within specified threshold)

    Args:
        env: Openai gym environment that contains a transition matrix P which is a dictionary of
            lists, where P[s][a] == [(probability, nextstate, reward, done), ...].
        gamma: Reward discount factor.
        threshold: Parameter controlling when to stop value iteration (i.e. when the diff between
            state-values at each iteration are all less than threshold).
    """
    # Initialise v, the estimated value at each state.
    v = np.zeros(env.observation_space.n)
    # Track the number of passes through all states
    n_iter = 0
    # Delta tracks the max difference between v estimates at each iteration
    delta = threshold

    while delta >= threshold:
        delta = 0
        for s in range(env.nS):
            q_vals = []
            for a in range(env.nA):
                # Weighted sum of expected rewards by taking action a, bootstrapping from
                # values at the new state
                q = np.sum([i[0] * (i[2] + gamma * v[i[1]]) for i in env.P[s][a]])
                q_vals.append(q)
            v_new = np.max(q_vals)
            delta = np.max([delta, np.abs(v_new - v[s])])
            v[s] = v_new
        n_iter += 1
    return v, n_iter

def optimal_policy(env, gamma, threshold=1e-4):
    v, _ = value_iter(env, gamma, threshold)
    pol = {}
    for s in range(env.nS):
        q_vals = []
        for a in range(env.nA):
            # Weighted sum of expected rewards by taking action a, bootstrapping from
            # values at the new state
            q = np.sum([i[0] * (i[2] + gamma * v[i[1]]) for i in env.P[s][a]])
            q_vals.append(q)
        pol[s] = list(np.flatnonzero(q_vals == np.max(q_vals)))
    return pol


def value_iter_finite(env, gamma, horizon=20):
    """
    Runs value iteration for at most horizon steps. The values for each iteration t are stored, and
    represent the expected value of that state given t timesteps remaining in the episode.

    Args:
        env: Openai gym environment that contains a transition matrix P which is a dictionary of
            lists, where P[s][a] == [(probability, nextstate, reward, done), ...].
        gamma: Reward discount factor.
        horizon: The maximum number of iterations to run. Note that value iteration will also
            stop if the values for each state don't change between iterations.
    """
    # A numpy array representing the expected values for each state and each timestep remaining
    v = np.zeros((env.observation_space.n, horizon))

    for t in range(horizon):
        for s in range(env.nS):
            q_vals = []
            for a in range(env.nA):
                # At first iteration, don't bootstrap from value of next state at previous timestep
                if t == 0:
                    # Weighted sum of rewards over actions
                    q = np.sum([i[0] * i[2] for i in env.P[s][a]])
                else:
                    # Weighted sum of expected rewards by taking action a, bootstrapping from
                    # values at the new state from the previous iteration
                    q = np.sum([i[0] * (i[2] + gamma * v[i[1],t-1]) for i in env.P[s][a]])
                q_vals.append(q)
            v[s,t] = np.max(q_vals)
        # If the values haven't changed from the previous timestep, we are done
        if t > 0 and np.array_equal(v[:,t], v[:,t-1]):
            break
    # All state-values after t are irrelevant (they would have been identical if we didn't
    # break early)
    return v[:,:t]

def optimal_policy_finite(env, gamma, horizon=20):
    v = value_iter_finite(env, gamma, horizon)
    pol = {}
    for t in range(v.shape[1]):
        pol[t] = {}
        for s in range(env.nS):
            q_vals = []
            for a in range(env.nA):
                # If one timestep remaining, don't bootstrap from value of next state at
                # previous timestep
                if t == 0:
                    # Weighted sum of rewards over actions
                    q = np.sum([i[0] * i[2] for i in env.P[s][a]])
                else:
                    # Weighted sum of expected rewards by taking action a, bootstrapping from
                    # values at the new state from the previous iteration
                    q = np.sum([i[0] * (i[2] + gamma * v[i[1],t-1]) for i in env.P[s][a]])
                q_vals.append(q)
            # Get all actions with max q_vals
            pol[t][s] = list(np.flatnonzero(q_vals == np.max(q_vals)))
    return pol
