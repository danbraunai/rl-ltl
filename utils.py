"""
Various utility functions used in rl-ltl experiments.
"""
import random
import numpy as np
import re


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

def scalarize_rewards(rm_files, weights):
    """
    Use linear scalarization of the constant rewards in rm_files. As per
    https://www.jmlr.org/papers/volume15/vanmoffaert14a/vanmoffaert14a.pdf Sec 2.2.1. Assumes all
    files have the same structure.
    """
    # Get constant rewards from each file
    rewards = []
    for rm_file in rm_files:
        with open(rm_file) as f:
            data = f.read()
            f_rewards = re.findall(r"(?<=ConstantRewardFunction\()(.*?)(?=\))", data)
            rewards.append([float(r) for r in f_rewards])

    scaled_rewards = [weights[0] * r1 + weights[1] * r2 for r1, r2 in zip(rewards[0], rewards[1])]

    # Combine filenames. e.g. [./envs/rm1_frozen_lake.txt, ./envs/rm1_frozen_lake.txt]
    # becomes ./envs/rm1_rm2_frozen_lake.txt
    rm_ids = [f.split("/")[-1].split("_")[0] for f in rm_files]
    env_name = rm_files[0].split("/")[-1].split("_", 1)[1]
    parent_dir = "/".join(rm_files[0].split("/")[:-1]) + "/"
    comb_filename = parent_dir + "_".join(rm_ids) + "_" + env_name

    # Write new file replacing rewards with scaled rewards
    with open(comb_filename, "w") as f:
        # Use most recently opened file (stored in data)
        for line in data.split("\n"):
            if "ConstantRewardFunction" not in line:
                f.write(line + "\n")
            else:
                f.write(
                    re.sub(r"(?<=Function\()(.*?)(?=\))", str(scaled_rewards.pop(0)), line) + "\n"
                )
    return comb_filename

def combine_results(data):
    """
    Convert a list of tuples of the form
    [(alg_name, {samples: [...], updates: [...], rewards: [...]}),...] to a dictionary of the form
    {alg_name: {samples: [...], updates: [...], rewards: [[...],[...],...]}}
    """
    res_dict = {}
    for alg, d, qs in data:
        qs_str = {str(k): v for k, v in qs.items()}
        try:
            res_dict[alg]["rewards"].append(d["rewards"])
        except KeyError:
            # alg has not beed added to res_dict
            res_dict[alg] = {
                "samples": d["samples"], "updates": d["updates"], "qs": qs_str, "rewards": [d["rewards"]]
            }
    return res_dict