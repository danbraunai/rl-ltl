"""
This module contains functions for calculating optimal policies and state-values in an openai
gym environment.
"""

import numpy as np

def value_iteration(env, rm, gamma, threshold=0.00001, get_policy=True, step_penalty=0):
    """
    Runs value iteration until convergence (within specified threshold). Mostly the same as
    https://github.com/RodrigoToroIcarte/reward_machines/blob/master/reward_machines/envs/grids/value_iteration.py
    but handles stoachstic environments.
    Note that the value at given at an env state may not be valid if that env state cannot be
    visited unless in a terminal rm state. E.g. consider a simple reachability task in a grid
    world, with a goal state of cell G. Note that there is one non-terminal rm state U and one
    terminal state. Value iteration will output V(G,U)=1, even though it is impossible to be in
    state G whilst in a non-terminal rm state U.

    Args:
        env: Openai gym environment that contains a transition matrix P which is a dictionary of
            lists, where P[s][a] == [(probability, nextstate, reward, done, label), ...].
        rm: The reward machine instance
        gamma: Reward discount factor.
        threshold: Parameter controlling when to stop value iteration (i.e. when the diff between
            state-values at each iteration are all less than threshold).
        get_policy: Bool indicating whether to calculate the optimal policy given the optimal vals.
        step_penalty: Positive float indicating the reward penalty to give for each step.
    """
    assert len(env.reward_machines) == 1, "Value iteration only supports one reward machine"
    env_states = list(env.P.keys())
    rm_states = rm.get_states()
    # Initialise v, the estimated value at each state.
    v = {(s, u): 0 for s in env_states for u in rm_states}
    # Track the number of passes through each (s, u) pair
    n_iter = 0
    # Delta tracks the max difference between v estimates at each iteration
    delta = threshold

    while delta >= threshold:
        delta = 0
        for s in env_states:
            for u in rm_states:
                q_vals = []
                for a in range(env.nA):
                    q_val = 0
                    # Iterate through possible next states
                    for prob, new_s, _, _ in env.P[s][a]:
                        label = env.get_label(new_s)
                        new_u, r, done = rm.step(u, label, None)
                        r -= step_penalty
                        # print(step_penalty)
                        # Weighted sum of expected rewards
                        if done:
                            q_val += prob * r
                        else:
                            q_val += prob * (r + gamma * v[(new_s, new_u)])
                    q_vals.append(q_val)
                v_new = np.max(q_vals)
                delta = np.max([delta, np.abs(v_new - v[(s, u)])])
                v[(s, u)] = v_new
                n_iter += 1
        if n_iter / len(env_states) * len(rm_states) == 10000:
            print("No convergence")
            break

    policy = {}
    if get_policy:
        for s in env_states:
            for u in rm_states:
                q_vals = []
                for a in range(env.nA):
                    q_val = 0
                    # Iterate through possible next states
                    for prob, new_s, _, _ in env.P[s][a]:
                        label = env.get_label(new_s)
                        new_u, r, done = rm.step(u, label, None)
                        r -= step_penalty
                        # Weighted sum of expected rewards
                        if done:
                            q_val += prob * r
                        else:
                            q_val += prob * (r + gamma * v[(new_s, new_u)])
                    q_vals.append(q_val)
                # Store list of actions with max q_vals
                policy[(s, u)] = list(np.flatnonzero(q_vals == np.max(q_vals)))

    return v, n_iter, policy

def value_iteration_finite(env, rm, gamma, horizon=20, get_policy=True):
    """
    Runs value iteration for horizon steps, or until the values don't change between steps.
    The values for each step t are stored, and represent the expected value of that state
    given t timesteps remaining in the episode.

    Args:
        env: Openai gym environment that contains a transition matrix P which is a dictionary of
            lists, where P[s][a] == [(probability, nextstate, reward, done, label), ...].
        rm: The reward machine instance
        gamma: Reward discount factor.
        horizon: The maximum number of iterations to run. Note that value iteration will also
            stop if the values for each state don't change between iterations.
        get_policy: Bool indicating whether to calculate the optimal policy given the optimal vals.
    """

    assert len(env.reward_machines) == 1, "Value iteration only supports one reward machine"
    env_states = list(env.P.keys())
    rm_states = rm.get_states()
    # Initialise v, the estimated value at each state.
    v = {(s, u, t): 0 for s in env_states for u in rm_states for t in range(1, horizon + 1)}
    # Track the number of passes through all states
    n_iter = 0

    for t in range(1, horizon + 1):
        for s in env_states:
            for u in rm_states:
                q_vals = []
                for a in range(env.nA):
                    q_val = 0
                    # Iterate through possible next states
                    for prob, new_s, _, _ in env.P[s][a]:
                        label = env.get_label(new_s)
                        new_u, r, done = rm.step(u, label, None)
                        # Weighted sum of expected rewards. Don't bootstrap if only 1 step left
                        if done or t == 1:
                            q_val += prob * r
                        else:
                            # Bootstrap using values from previous t as one less step remaining
                            q_val += prob * (r + gamma * v[(new_s, new_u, t - 1)])
                    q_vals.append(q_val)

                v[(s, u, t)] = np.max(q_vals)
                n_iter += 1

    policy = {}
    if get_policy:
        for t in range(1, horizon + 1):
            for s in env_states:
                for u in rm_states:
                    q_vals = []
                    for a in range(env.nA):
                        q_val = 0
                        # Iterate through possible next states
                        for prob, new_s, _, _ in env.P[s][a]:
                            label = env.get_label(new_s)
                            new_u, r, done = rm.step(u, label, None)
                            # Weighted sum of expected rewards
                            if done or t == 1:
                                q_val += prob * r
                            else:
                                # Bootstrap using values from previous t as one less step remaining
                                q_val += prob * (r + gamma * v[(new_s, new_u, t - 1)])
                        q_vals.append(q_val)
                    # Store list of actions with max q_vals
                    policy[(s, u, t)] = list(np.flatnonzero(q_vals == np.max(q_vals)))

    return v, n_iter, policy
