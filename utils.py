import numpy as np
import random

def display_optimal_vals(v_dict, state_shape, n_rm_states):
    """
    Convert the dictionary v_dict of values produced by value_iteration of the form
    ((row, col), rm_s): value, to a 3D numpy array of the form (rm_state, row, col).
    """
    v = np.empty((n_rm_states,*state_shape))
    for ((row, col), rm_s), val in v_dict.items():
        v[rm_s, row, col] = round(val, 3)
    print(v)

def display_optimal_policy(pol_dict, state_shape, n_rm_states):
    """
    Convert the dictionary pol_dict of optimal actions produced by value_iteration of the form
    ((row, col), rm_s): action_list, to a 3D numpy array of the form (rm_state, row, col) with
    list entries.
    """
    pol = np.empty((n_rm_states,*state_shape), dtype=object)
    for ((row, col), rm_s), actions in pol_dict.items():
        pol[rm_s, row, col] = tuple(actions)
    print(pol)

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)