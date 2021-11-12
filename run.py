import time
import datetime
import multiprocessing
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

TODAY = datetime.datetime.now().strftime("%d-%m-%y")

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
    # for i in range(pol.shape[0]):
        # print(i, pol[i], "\n")
    # print(pol)
    print(n_iter)
    return v


def run_q_algo(options, rm_files, map_name, obj_name, out_dir, finite=False):
    rm_env = FrozenLakeRMEnv(
        rm_files, map_name=map_name, obj_name=obj_name, slip=0.5, seed=options["seed"],
        all_acts=True
    )
    rm_env = RewardMachineWrapper(rm_env, options["use_crm"], options["use_rs"], options["gamma"], 1)
    if finite:
        ql = QTLearning(rm_env, **options)
    else:
        ql = QLearning(rm_env, **options)
    # Since starmap returns in any order, we must identify the algorithm
    return options["alg_name"], ql.learn(), ql.q

def finite_horizon_experiments(task_num, horizon, eps_per_reset, total_steps, lr=0.1, lr_decay=0.9,
                               lr_decay_freq=100000, seed=None, file_date=None):
    rm_files = [f"./envs/rm_t{task_num}_frozen_lake.txt"]
    map_name = "5x5"
    obj_name = f"objects_t{task_num}"
    out_dir = "results"
    num_runs = 30
    optimal_vals = run_value_iteration(task_num, step_penalty=0, horizon=horizon, gamma=1)
    # Get the optimal value corresponding to the first env and RM state at the horizon.
    # This corresponds to the probability of completing the task and the optimal expected reward
    # (assuming the only initial state is at (0,0))
    optimal_reward = optimal_vals[-1][0][0, 0]

    date = file_date or TODAY
    if not file_date:
        run_options = []
        for use_t in [True, False]:
            for use_crm in [True, False]:
                alg_prefix = "Q"
                if use_t:
                    alg_prefix += "T"
                if use_crm:
                    alg_prefix += "RM"
                alg_name = alg_prefix + "-learning"
                for i in range(num_runs):
                    seed += 1
                    options = {
                        "alg_name": alg_name,
                        "seed": seed,
                        "lr": lr,
                        "lr_decay": lr_decay,
                        "lr_decay_freq": lr_decay_freq,
                        "gamma": 1,
                        "epsilon": 0.1,
                        "total_steps": total_steps,
                        "eps_per_reset": eps_per_reset,
                        # Only gets used in QTlearning
                        "horizon": horizon,
                        "use_t": use_t,
                        "use_crm": use_crm,
                        "use_rs": False,
                        "print_freq": 5000000,
                        "eval_freq": 10000,
                        "num_eval_eps": 50,
                        "q_init": 1.0001,
                    }
                    run_options.append(options)

        print(f"Running task {task_num} in finite horizon setting")
        # run_q_algo(options, rm_files, map_name, obj_name, out_dir, True)
        q_args = [[opt] + [rm_files, map_name, obj_name, out_dir, True] for opt in run_options] 
        with multiprocessing.Pool(multiprocessing.cpu_count()-2) as pool:
            results = pool.starmap(run_q_algo, q_args)

        results_dict = utils.combine_results(results)

        with open(f"{out_dir}/fin_t{task_num}_{date}.json", "w") as f:
            json.dump(results_dict, f, indent=4)

    with open(f"{out_dir}/fin_t{task_num}_{date}.json") as f:
        results_dict = json.load(f)
    

    plotting.plot_rewards(results_dict, out_dir, f"Task{task_num}", "finite", optimal_reward, date)

def infinite_horizon_experiments(task_num, total_steps, gamma=0.99, lr=0.01, lr_decay=0.9,
                                 lr_decay_freq=200000, seed=None, file_date=None):
    rm_files = [f"./envs/rm_t{task_num}_frozen_lake.txt"]
    map_name = "5x5"
    obj_name = f"objects_t{task_num}"
    out_dir = "results"
    num_runs = 30
    optimal_vals = run_value_iteration(task_num, step_penalty=0, horizon=None, gamma=gamma)
    # Get the optimal value corresponding to the first env and RM state. This corresponds to the
    # probability of completing the task and the optimal expected reward
    # (assuming the only initial state is at (0,0))
    optimal_reward = optimal_vals[0][0, 0]

    date = file_date or TODAY
    if not file_date:
        run_options = []
        for use_crm in [True]:
            alg_name = "CRM" if use_crm else "Q-learning"
            for i in range(num_runs):
                seed += 1
                options = {
                    "alg_name": alg_name,
                    "seed": seed,
                    "lr": lr,
                    "lr_decay": lr_decay,
                    "lr_decay_freq": lr_decay_freq,
                    "gamma": gamma,
                    "epsilon": 0.1,
                    "total_steps": total_steps,
                    "n_rollout_steps": 50,
                    # Only gets used in QTlearning
                    "horizon": None,
                    "use_crm": use_crm,
                    "use_rs": False,
                    "print_freq": 5000000,
                    "eval_freq": 10000,
                    "num_eval_eps": 50,
                    "q_init": 1.0001,
                }
                run_options.append(options)

        print(f"Running task {task_num} in infinite horizon setting")
        # run_q_algo(options, rm_files, map_name, obj_name, out_dir, False)
        q_args = [[opt] + [rm_files, map_name, obj_name, out_dir] for opt in run_options] 
        with multiprocessing.Pool(multiprocessing.cpu_count()-2) as pool:
            results = pool.starmap(run_q_algo, q_args)

        results_dict = utils.combine_results(results)

        with open(f"{out_dir}/inf_t{task_num}_{date}.json", "w") as f:
            json.dump(results_dict, f, indent=4)

    with open(f"{out_dir}/inf_t{task_num}_{date}.json") as f:
        results_dict = json.load(f)

    plotting.plot_rewards(results_dict, out_dir, f"Task{task_num}", "infinite", optimal_reward, date)

def heuristic_experiments():
    # Reducing reward discount factor
    # for gamma in [0.6, 0.75, 0.9, 1]:
    #     run_value_iteration(task_name="t1", horizon=None, gamma=gamma)

    # Step-based penalties
    run_value_iteration(task_num=1, step_penalty=0.1, horizon=None, gamma=0.999)


if __name__ == "__main__":
    start = time.time()
    seed = 42

    # run_value_iteration(task_num=2, step_penalty=0, horizon=None, gamma=0.99)
    # run_value_iteration(task_num=3, step_penalty=0, horizon=None, gamma=0.1)


    infinite_horizon_experiments(
        task_num=2, total_steps=1000000, gamma=0.99, lr=0.01, lr_decay=0.9, lr_decay_freq=200000,
        seed=seed, file_date="12-11-21"
    )
    infinite_horizon_experiments(
        task_num=3, total_steps=4000000, gamma=0.99, lr=0.01, lr_decay=0.9, lr_decay_freq=200000,
        seed=seed, file_date="12-11-21"
    )
    # finite_horizon_experiments(
    #     task_num=2, horizon=6, eps_per_reset=3, total_steps=1000000, lr=0.1, lr_decay=0.9, lr_decay_freq=100000,
    #     seed=seed, file_date="11-11-21"
    # )
    # finite_horizon_experiments(
    #     task_num=3, horizon=15, eps_per_reset=3, total_steps=4000000, lr=0.1, lr_decay=0.9, lr_decay_freq=100000,
    #     seed=seed, file_date="11-11-21"
    # )

    # heuristic_experiments()

    # run_always_down()
    # run_a2c()
    print("Time taken:", time.time() - start)
