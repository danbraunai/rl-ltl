"""
Various utility functions used in rl-ltl experiments.
"""
import random
import numpy as np


def display_grid_optimals(v_dict, pol_dict, state_shape, n_rm_states, horizon=None):
    """Display optimal values and policy for grid environments."""
    v = display_grid_optimal_vals(v_dict, state_shape, n_rm_states, horizon)
    pol = display_grid_optimal_policy(pol_dict, state_shape, n_rm_states, horizon)
    return v, pol

def display_grid_optimal_vals(v_dict, state_shape, n_rm_states, horizon=None):
    """
    Convert the dictionary v_dict of values produced by value_iteration on a grid env of the form
    ((row, col), rm_s): value, to a 3D numpy array of the form (rm_state, row, col).
    """
    if horizon:
        # Note that our return array is zero-indexed, so the first
        # index of the time_rem corresponds to a time remaining of 1 step
        v = np.empty((horizon, n_rm_states, *state_shape))
        for ((row, col), rm_s, time_rem), val in v_dict.items():
            v[time_rem - 1, rm_s, row, col] = round(val, 3)
    else:
        v = np.empty((n_rm_states, *state_shape))
        for ((row, col), rm_s), val in v_dict.items():
            v[rm_s, row, col] = round(val, 3)
    return v

def display_grid_optimal_policy(pol_dict, state_shape, n_rm_states, horizon=None):
    """
    Convert the dictionary pol_dict of optimal actions produced by value_iteration on a grid env
    of the form ((row, col), rm_s): action_list, to a 3D numpy array of the form
    (rm_state, row, col) with list entries.
    """
    if horizon:
        # Note that our return array is zero-indexed, so the first
        # index of the time_rem corresponds to a time remaining of 1 step
        pol = np.empty((horizon, n_rm_states, *state_shape), dtype=object)
        for ((row, col), rm_s, time_rem), actions in pol_dict.items():
            pol[time_rem - 1, rm_s, row, col] = tuple(actions)
    else:
        pol = np.empty((n_rm_states, *state_shape), dtype=object)
        for ((row, col), rm_s), actions in pol_dict.items():
            pol[rm_s, row, col] = tuple(actions)
    return pol

def set_random_seed(seed):
    """Set global random seeds for reproducability."""
    np.random.seed(seed)
    random.seed(seed)
