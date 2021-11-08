import time
import datetime
import json
import numpy as np
import gym
from envs.frozen_lake import FrozenLakeRMEnv
# from stable_baselines3 import A2C

from agents.qlearning import QLearning
from agents.qtlearning import QTLearning
import solver
from reward_machines.rm_environment import RewardMachineEnv, RewardMachineWrapper
import utils
import plotting

TODAY = datetime.datetime.now().strftime("%d-%m-%Y")

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

def run_value_iteration(task_num, step_penalty=0, horizon=None, gamma=1):
    rm_files = [f"./envs/rm_t{task_num}_frozen_lake.txt"]
    multi_objective_weights = None
    map_name = "5x5"
    obj_name = f"objects_t{task_num}"
    options = {
        "gamma": gamma,
        "horizon": horizon,
    }

    if multi_objective_weights:
        # Create a new rm_file combining rm_files with multi_objective_weights
        rm_files = [utils.scalarize_rewards(rm_files, multi_objective_weights)]

    rm_env = FrozenLakeRMEnv(
        rm_files, map_name=map_name, obj_name=obj_name, slip=0.5, seed=None,
        all_acts=True
    )
    rm_env = RewardMachineWrapper(rm_env, False, False, options["gamma"], 1)
    # Only one reward machine for these experiments
    rm = rm_env.reward_machines[0]

    if horizon:
        optim_vals, n_iter, optim_pol = solver.value_iteration_finite(
            rm_env, rm, options["gamma"], horizon=options["horizon"]
        )
        v, pol = utils.display_grid_optimals(
            optim_vals, optim_pol, rm_env.env.desc.shape, len(rm.get_states()), options["horizon"]
        )
    else:
        optim_vals, n_iter, optim_pol = solver.value_iteration(
            rm_env, rm, options["gamma"], step_penalty=step_penalty
        )
        v, pol = utils.display_grid_optimals(
            optim_vals, optim_pol, rm_env.env.desc.shape, len(rm.get_states()), horizon=None
        )
    print(v)
    for i in range(pol.shape[0]):
        print(i, pol[i], "\n")
    # print(pol)
    print(n_iter)


def run_q_algo(rm_files, map_name, obj_name, options, out_dir, finite=False):
    rm_env = FrozenLakeRMEnv(
        rm_files, map_name=map_name, obj_name=obj_name, slip=0.5, seed=options["seed"],
        all_acts=True
    )
    rm_env = RewardMachineWrapper(rm_env, options["use_crm"], options["use_rs"], options["gamma"], 1)

    if finite:
        ql = QTLearning(rm_env, **options)
    else:
        ql = QLearning(rm_env, **options)
    return ql.learn()

def finite_horizon_experiments(task_num, horizon, n_rollout_steps, seed=None):
    rm_files = [f"./envs/rm_t{task_num}_frozen_lake.txt"]
    map_name = "5x5"
    obj_name = f"objects_t{task_num}"
    out_dir = "results"
    num_runs = 2

    policy_info = {}
    for use_t in [False, True]:
        for use_crm in [False, True]:
            alg_prefix = "Q"
            if use_t:
                alg_prefix += "T"
            if use_crm:
                alg_prefix += "RM"
            alg_name = alg_prefix + "-learning"
            policy_info[alg_name] = {}
            for i in range(num_runs):
                print(f"Run: {i} with {alg_name}")
                seed += 1
                options = {
                    "seed": seed,
                    "lr": 0.1,
                    "gamma": 1,
                    "epsilon": 0.1,
                    "total_steps": 200000,
                    "n_rollout_steps": n_rollout_steps,
                    # Only gets used in QTlearning
                    "horizon": horizon,
                    "use_t": use_t,
                    "use_crm": use_crm,
                    "use_rs": False,
                    "print_freq": 50000,
                    "eval_freq": 5000,
                    "num_eval_eps": 30,
                }

    #             policy_info[alg_name][str(i)] = run_q_algo(
    #                 rm_files, map_name, obj_name, options, out_dir, finite=True
    #             )
    # with open(f"{out_dir}/fin_t{task_num}_{TODAY}.json", "w") as f:
    #     json.dump(policy_info, f, indent=4)

    with open(f"{out_dir}/fin_t{task_num}_{TODAY}.json") as f:
        policy_info = json.load(f)
    

    plotting.plot_rewards(policy_info, out_dir, f"Task{task_num}", "finite")
    # print(policy_info)

def infinite_horizon_experiments(task_num=2, seed=None):
    rm_files = [f"./envs/rm_t{task_num}_frozen_lake.txt"]
    map_name = "5x5"
    obj_name = f"objects_t{task_num}"
    out_dir = "results"
    num_runs = 20

    policy_info = {}
    for use_crm in [False, True]:
        alg_name = "CRM" if use_crm else "Q-learning"
        policy_info[alg_name] = {}
        for i in range(num_runs):
            print(f"Run: {i} with {alg_name}")
            seed += 1
            options = {
                "seed": seed,
                "lr": 0.1,
                "gamma": 0.99,
                "epsilon": 0.1,
                "total_steps": 200000,
                "n_rollout_steps": 50,
                # Only gets used in QTlearning
                "horizon": 10,
                "use_crm": use_crm,
                "use_rs": False,
                "print_freq": 50000,
                "eval_freq": 5000,
                "num_eval_eps": 30,
            }

    #         policy_info[alg_name][str(i)] = run_q_algo(
    #             rm_files, map_name, obj_name, options, out_dir, finite=False
    #         )
    # with open(f"{out_dir}/inf_t{task_num}_{TODAY}.json", "w") as f:
    #     json.dump(policy_info, f)

    with open(f"{out_dir}/inf_t{task_num}_{TODAY}.json") as f:
        policy_info = json.load(f)

    plotting.plot_rewards(policy_info, out_dir, f"Task{task_num}", "infinite")
    # return
    # qs = {}
    # for k in ql.q:
    #     qs[f"{k}"] = ql.q[k]
    # import json
    # with open(f"{out_dir}/qs.json", "w") as f:
    #     json.dump(qs, f, indent=4)

    # rollout_model(rm_env, ql, num_eps=1, horizon=options["horizon"])

def heuristic_experiments():
    # Reducing reward discount factor
    # for gamma in [0.6, 0.75, 0.9, 1]:
    #     run_value_iteration(task_name="t1", horizon=None, gamma=gamma)

    # Step-based penalties
    run_value_iteration(task_num=1, step_penalty=0.1, horizon=None, gamma=0.999)


if __name__ == "__main__":
    start = time.time()
    seed = 36

    # run_value_iteration(task_num=1, step_penalty=None, horizon=None, gamma=0.8)
    # run_value_iteration(task_num=1, step_penalty=None, horizon=10, gamma=1)

    # run_q_algo(finite=False)
    # for i in [2, 3]:
    #     infinite_horizon_experiments(task_num=i, seed=seed)
    finite_horizon_experiments(task_num=2, horizon=6, n_rollout_steps=15, seed=seed)
    finite_horizon_experiments(task_num=3, horizon=15, n_rollout_steps=40, seed=seed)

    # heuristic_experiments()

    # run_always_down()
    # run_a2c()
    print("Time taken:", time.time() - start)
