import time
import numpy as np
import gym
from envs.frozen_lake import FrozenLakeRMEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import A2C

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

def run_a2c():
    env = FrozenLake(map_name="5x5", slip=0.5)
    # With gamma<1, agent learns to only go DownRight in s3
    model = A2C('MlpPolicy', env, gamma=0.99, verbose=1)
    model.learn(total_timesteps=10000)
    rollout_model(env, model)

def rollout_model(env, model, num_eps=1, horizon=1e7):
    for ep in range(num_eps):
        s = tuple(env.reset())
        env.render()
        t = 0
        for _ in range(20):
            # If no q-values for this state, add them
            if s not in model.q:
                print("No q-vals for:", s)
                model.q[s] = np.zeros(model.env.action_space.n)
            if horizon:
                a, _ = model.predict((s, t), deterministic=True)
            else:
                a, _ = model.predict(s, deterministic=True)
            new_s, reward, done, info = env.step(a)
            env.render()
            s = tuple(new_s)

            t += 1
            if done or t == horizon:
                print("Finished ep")
                break

def run_value_iteration(finite=False):
    seed = 41
    rm_file = "./envs/rm1_frozen_lake.txt"
    map_name = "5x5"
    obj_name = "objects_v1"
    options = {
        "seed": seed,
        "lr": 0.5,
        "gamma": 1,
        "epsilon": 0.1,
        "n_episodes": 1000,
        "n_rollout_steps": 100,
        "use_crm": True,
        "use_rs": False,
        "horizon": 10,
    }

    rm_env = FrozenLakeRMEnv(
        [rm_file], map_name=map_name, obj_name=obj_name, slip=0.5, seed=options["seed"]
    )
    # rm_env = RewardMachineWrapper(rm_env, args.use_crm, args.use_rs, args.gamma, args.rs_gamma)
    rm_env = RewardMachineWrapper(rm_env, options["use_crm"], options["use_rs"], options["gamma"], 1)
    # Only one reward machine for these experiments
    rm = rm_env.reward_machines[0]

    if finite:
        optim_vals, n_iter, optim_pol = solver.value_iteration_finite(
            rm_env, rm, options["gamma"], horizon=options["horizon"]
        )
        utils.display_grid_optimals(
            optim_vals, optim_pol, rm_env.env.desc.shape, len(rm.get_states()), options["horizon"]
        )
    else:
        optim_vals, n_iter, optim_pol = solver.value_iteration(rm_env, rm, options["gamma"])
        utils.display_grid_optimals(
            optim_vals, optim_pol, rm_env.env.desc.shape, len(rm.get_states()), horizon=None
        )

def run_qlearning():
    seed = 33
    rm_file = "./envs/rm1_frozen_lake.txt"
    map_name = "5x5"
    obj_name = "objects_v1"
    options = {
        "seed": seed,
        "lr": 0.5,
        "gamma": 1,
        "epsilon": 0.4,
        "n_episodes": 10000,
        "n_rollout_steps": 100,
        "use_crm": True,
        "use_rs": False,
    }

    rm_env = FrozenLakeRMEnv(
        [rm_file], map_name=map_name, obj_name=obj_name, slip=0.5, seed=options["seed"]
    )
    # rm_env = RewardMachineWrapper(rm_env, args.use_crm, args.use_rs, args.gamma, args.rs_gamma)
    rm_env = RewardMachineWrapper(rm_env, options["use_crm"], options["use_rs"], options["gamma"], 1)
    ql = QLearning(rm_env, **options, print_freq=20000)
    ql.learn()

    qs = {}
    for k in ql.q:
        qs["".join(str(i) for i in k)] = ql.q[k].tolist()
    import json
    with open("qs.json", "w") as f:
        json.dump(qs, f, indent=4)

    rollout_model(rm_env, ql, num_eps=1, horizon=None)


if __name__ == "__main__":
    start = time.time()
    run_value_iteration(finite=True)
    # run_qlearning()
    # run_qtlearning()
    # run_always_down()
    # run_a2c()
    print("Time taken:", time.time() - start)
