import time
import numpy as np
import gym
from envs.frozen_lake import FrozenLakeRMEnv
# from stable_baselines3 import A2C

from agents.qlearning import QLearning
from agents.qtlearning import QTLearning
import solver
from reward_machines.rm_environment import RewardMachineEnv, RewardMachineWrapper
import utils


def run_always_down():
    env = FrozenLake(map_name="5x5", slip=0.5)
    go_down = 0
    n_rollout_steps = 20
    for step in range(n_rollout_steps):
        obs, reward, done, _ = env.step(go_down)
        print(f"Step: {step}. Obs: {obs}. Reward: {reward}. Done: {done}")
        env.render()
        if done:
            print("Rollout finished.")
            break

# def run_a2c():
#     env = FrozenLake(map_name="5x5", slip=0.5)
#     # With gamma<1, agent learns to only go DownRight in s3
#     model = A2C('MlpPolicy', env, gamma=0.99, verbose=1)
#     model.learn(total_timesteps=10000)
#     rollout_model(env, model)

def rollout_model(env, model, num_eps=1, horizon=20):
    for ep in range(num_eps):
        s = tuple(env.reset())
        env.render()
        for t in range(horizon, 0, -1):
            if horizon:
                # If no q-values for this state, add them
                if (t, s) not in model.q:
                    model.q[(t, s)] = [0] * model.env.action_space.n
                a, _ = model.predict((t, s), deterministic=True)
            else:
                # If no q-values for this state, add them
                if s not in model.q:
                    model.q[s] = [0] * model.env.action_space.n
                a, _ = model.predict(s, deterministic=True)
            new_s, _, done, _ = env.step(a)
            env.render()
            s = tuple(new_s)

            if done:
                print("Finished ep")
                break

def run_value_iteration(finite=False):
    seed = 41
    rm_files = ["./envs/rm_t1_frozen_lake.txt"]
    multi_objective_weights = None
    map_name = "5x5"
    obj_name = "objects_t2"
    options = {
        "seed": seed,
        "lr": 0.5,
        "gamma": 1,
        "epsilon": 0.1,
        "n_episodes": 1000,
        "n_rollout_steps": 100,
        "use_crm": True,
        "use_rs": False,
        "horizon": 15,
        "all_acts": True,
    }

    if multi_objective_weights:
        # Create a new rm_file combining rm_files with multi_objective_weights
        rm_files = [utils.scalarize_rewards(rm_files, multi_objective_weights)]

    rm_env = FrozenLakeRMEnv(
        rm_files, map_name=map_name, obj_name=obj_name, slip=0.5, seed=options["seed"],
        all_acts=options["all_acts"]
    )
    rm_env = RewardMachineWrapper(rm_env, options["use_crm"], options["use_rs"], options["gamma"], 1)
    # Only one reward machine for these experiments
    rm = rm_env.reward_machines[0]

    if finite:
        optim_vals, n_iter, optim_pol = solver.value_iteration_finite(
            rm_env, rm, options["gamma"], horizon=options["horizon"]
        )
        v, pol = utils.display_grid_optimals(
            optim_vals, optim_pol, rm_env.env.desc.shape, len(rm.get_states()), options["horizon"]
        )
    else:
        optim_vals, n_iter, optim_pol = solver.value_iteration(rm_env, rm, options["gamma"])
        v, pol = utils.display_grid_optimals(
            optim_vals, optim_pol, rm_env.env.desc.shape, len(rm.get_states()), horizon=None
        )
    print(v)
    for i in range(pol.shape[0]):
        print(i, pol[i], "\n")
    # print(pol)
    print(n_iter)

def run_qlearning(finite=False):
    seed = 33
    rm_files = ["./envs/rm1_frozen_lake.txt"]
    map_name = "5x5"
    obj_name = "objects_t1"
    options = {
        "seed": seed,
        "lr": 0.5,
        "gamma": 1,
        "epsilon": 0.4,
        "n_episodes": 10000,
        # Only gets used in QLearning
        "n_rollout_steps": 100,
        # Only gets used in QTlearning
        "horizon": 10,
        "use_crm": True,
        "use_rs": False,
        "print_freq": 10000,
        "all_acts": False,
    }

    rm_env = FrozenLakeRMEnv(
        rm_files, map_name=map_name, obj_name=obj_name, slip=0.5, seed=options["seed"],
        all_acts=options["all_acts"]
    )
    rm_env = RewardMachineWrapper(rm_env, options["use_crm"], options["use_rs"], options["gamma"], 1)

    if finite:
        ql = QTLearning(rm_env, **options)
    else:
        ql = QLearning(rm_env, **options)
    ql.learn()

    qs = {}
    for k in ql.q:
        qs[f"{k}"] = ql.q[k]
    import json
    with open("qs.json", "w") as f:
        json.dump(qs, f, indent=4)

    rollout_model(rm_env, ql, num_eps=1, horizon=options["horizon"])


if __name__ == "__main__":
    start = time.time()
    run_value_iteration(finite=False)
    # run_qlearning(finite=False)
    # run_always_down()
    # run_a2c()
    print("Time taken:", time.time() - start)
