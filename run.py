import os
import argparse
import yaml
import time
import datetime
import multiprocessing
import json
import numpy as np
import gym
from envs.frozen_lake import FrozenLakeRMEnv, FrozenLake
# Uncomment if you wish to test with stable_baselines3. Not used by default due to package size
# from stable_baselines3 import A2C

from agents.q_inf_learning import QInfLearning
from agents.q_fin_learning import QFinLearning
import solver
from reward_machines.rm_environment import RewardMachineEnv, RewardMachineWrapper
import utils
import plotting

TODAY = datetime.datetime.now().strftime("%d-%m-%y")


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--params", type=str, default="thesis", help="name of first level key in hyperparams.yml"
    )
    parser.add_argument(
        "--experiments", type=str, nargs="+", default="all",
        help=""" Names of experiments to run, as labelled by hyperparams.yml second level keys.
                 Leave empty to run all experiments for the first level key in --params """
    )
    parser.add_argument(
        "--num_processes", type=int, default=1,
        help="Number of processes used for experiments. Uses multiprocessing if more than one"
    )
    parser.add_argument(
        "--seed", type=int, nargs="?", help="Seed for reproducibility"
    )
    parser.add_argument(
        "--out_dir", type=str, default="results", help="Directory in which to save results"
    )
    parser.add_argument(
        "--preload_date", type=str, default=None, help="Date in which to load presaved experiments"
    )  
    args = parser.parse_args()
    return args

def run_always_down():
    env = FrozenLake(map_name="5x5", slip=0.5)
    go_down_act = 2
    n_rollout_steps = 20
    for step in range(n_rollout_steps):
        obs, reward, done, _ = env.step(go_down_act)
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

def rollout_model(env, model, num_eps=1, n_steps=20):
    for ep in range(num_eps):
        s = tuple(env.reset())
        env.render()
        for _ in range(n_steps):
            a, _ = model.predict(s, deterministic=True)
            new_s, _, done, _ = env.step(a)
            env.render()
            s = tuple(new_s)
            if done:
                print("Finished ep")
                break

def value_iteration_experiment(args, params):
    task_num = params["task_num"]
    horizon = params["horizon"]
    rm_files = [f"./envs/rm_t{task_num}_frozen_lake.txt"]
    multi_objective_weights = None
    map_name = "5x5"
    obj_name = f"objects_t{task_num}"

    if multi_objective_weights:
        # Create a new rm_file combining rm_files with multi_objective_weights
        rm_files = [utils.scalarize_rewards(rm_files, multi_objective_weights)]

    rm_env = FrozenLakeRMEnv(
        rm_files, map_name=map_name, obj_name=obj_name, slip=params["slip"], seed=None,
        all_acts=True
    )
    rm_env = RewardMachineWrapper(rm_env, False, False, 1, 1)
    # Only one reward machine for these experiments
    rm = rm_env.reward_machines[0]

    # Handle experiments with single or multiple gamma values
    for gamma in params["gammas"]:
        if params["horizon"]:
            optim_vals, n_iter, optim_pol = solver.value_iteration_finite(
                rm_env, rm, gamma, horizon=params["horizon"]
            )
            v, pol = utils.display_grid_optimals(
                optim_vals, optim_pol, rm_env.env.desc.shape, len(rm.get_states()), params["horizon"]
            )
        else:
            optim_vals, n_iter, optim_pol = solver.value_iteration(
                rm_env, rm, gamma, step_penalty=params["step_penalty"]
            )
            v, pol = utils.display_grid_optimals(
                optim_vals, optim_pol, rm_env.env.desc.shape, len(rm.get_states()), horizon=None
            )
        print(f"Optimal values Task {task_num}, horizon {params['horizon']}, gamma={gamma}:\n", v)
    # Only use return value when single gamma passed for q_learning experiments
    return v


def run_q_algo(params, rm_files, map_name, obj_name, out_dir, finite=False):
    rm_env = FrozenLakeRMEnv(
        rm_files, map_name=map_name, obj_name=obj_name, slip=params["slip"], seed=params["seed"],
        all_acts=True
    )
    rm_env = RewardMachineWrapper(rm_env, params["use_crm"], False, params["gamma"], 1)
    if finite:
        ql = QFinLearning(rm_env, **params)
    else:
        ql = QInfLearning(rm_env, **params)
    # Since starmap returns results in any order, we must identify the algorithm
    return params["alg_name"], ql.learn(), ql.q

def q_experiment(args, params):
    seed = args.seed
    task_num = params["task_num"]
    rm_files = [f"./envs/rm_t{task_num}_frozen_lake.txt"]
    map_name = "5x5"
    obj_name = f"objects_t{task_num}"
    out_dir = args.out_dir
    horizon_type = "finite" if params["horizon"] is not None else "infinite"

    optimal_vals = value_iteration_experiment(args, {
        "task_num": task_num, "step_penalty": 0, "horizon": params["horizon"],
        "gammas": [params["gamma"]], "slip": params["slip"]
    })
    if horizon_type == "finite":
        # Optimal reward for the horizon step, the initial env state (0,0) and RM state (0).
        # This corresponds to the probability of completing the task and the optimal expected
        # reward over an episode.
        optimal_reward = optimal_vals[params["horizon"]-1][0][0, 0]
    else:
        # Without horizon, get the optimal reward from the initial env state (0,0) and RM state (0)
        optimal_reward = optimal_vals[0][0, 0]

    date = args.preload_date or TODAY

    # See if we need to run the experiments from scratch
    if not args.preload_date:
        all_params = []
        if horizon_type == "finite":
            # Finite horizon experiment, running Q-, QT-, QRM-, QTRM- learning
            for use_t in [True, False]:
                for use_crm in [True, False]:
                    alg_prefix = "Q"
                    if use_t:
                        alg_prefix += "T"
                    if use_crm:
                        alg_prefix += "RM"
                    alg_name = alg_prefix + "-learning"
                    for i in range(params["num_runs"]):
                        # Iterate on the seed if the user provided one
                        seed = seed + 1 if seed is not None else None
                        # Add the run specific parameters
                        options = {
                            "seed": seed, "alg_name": alg_name, "use_crm": use_crm, "use_t": use_t,
                            **params
                        }
                        all_params.append(options)
        else:
            # Infinite horizon experiment, running CRM and Q-learning
            for use_crm in [True, False]:
                alg_name = "CRM" if use_crm else "Q-learning"
                for i in range(params["num_runs"]):
                    # Iterate on the seed if the user provided one
                    seed = seed + 1 if seed is not None else None
                    # Add the run specific parameters
                    options = {
                        "seed": seed, "alg_name": alg_name, "use_crm": use_crm, "use_t": False,
                        **params
                    }
                    all_params.append(options)

        print(f"Running task {task_num} with horizon {params['horizon']}")
        q_args = [[run_params, rm_files, map_name, obj_name, out_dir] for run_params in all_params]
        if args.num_processes > 1:
            with multiprocessing.Pool(multiprocessing.cpu_count()-1) as pool:
                results = pool.starmap(run_q_algo, q_args)
        else:
            results = [run_q_algo(*args) for args in q_args]

        results_dict = utils.combine_results(results)
        
        # Save results
        with open(f"{out_dir}/{horizon_type[:3]}_t{task_num}_{date}.json", "w") as f:
            json.dump(results_dict, f, indent=4)

    with open(f"{out_dir}/{horizon_type[:3]}_t{task_num}_{date}.json") as f:
        results_dict = json.load(f)
    

    plotting.plot_rewards(results_dict, out_dir, f"Task{task_num}", horizon_type, optimal_reward, date)

def get_experiments(args, hyperparams):
    """
    Get all experiments and their corresponding args, by converting the --experiments names given
    by the user to their corresponding functions in this module. Example conversions:
    q_fin_task3 -> (fin_horizon_experiment, <corresponding params>)
    q_inf_task2 -> (inf_horizon_experiment, <correpsonding params>)
    vidiscount_inf_task3 -> (value_iteration_experiment, <correpsonding params>)
    vi_fin_task3 -> (value_iteration_experiment, <correpsonding params>)
    """
    # Check that the user entered a valid --params
    assert args.params in hyperparams, f"{args.params} is not a key in hyperparams.yml"

    # Get all experiment names
    if args.experiments == "all":
        exp_names = list(hyperparams[args.params].keys())
    else:
        # Check that user entered a valid --experiments
        for name in args.experiments:
            assert name in hyperparams[args.params], (
                f"{name} is not a key in hyperparams.yml under {args.params}"
            )
        exp_names = args.experiments

    global_vars = globals()
    experiments = []
    for exp_name in exp_names:
        alg_type, horizon_type, task  = exp_name.split("_")
        task_num = int(task.replace("task", ""))
        # Get the function corresponding to the alg_type and horizon_type
        func_str = ""
        if alg_type[0] == "q":
            # func_str = f"{horizon_type}_horizon_experiment"
            func_str = f"q_experiment"
        elif alg_type[:2] == "vi":
            func_str = "value_iteration_experiment"
        if func_str not in global_vars:
            raise KeyError(f"Experiment key {exp_name} does not correspond to a known function")

        func = global_vars[func_str]
        # Add the task number to the hyperparams
        params = {"task_num": task_num, **hyperparams[args.params][exp_name]}
        experiments.append((func, params))
    return experiments



if __name__ == "__main__":
    start = time.time()

    # Uncomment to test environment with stable_baselines algorithm or naive agent
    # run_a2c()
    # run_always_down()

    args = read_args()
    with open("hyperparams.yml") as f:
        hyperparams = yaml.safe_load(f)
    
    # Collect experiments
    experiments = get_experiments(args, hyperparams)

    # Ensure out_dir exists
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # Run experiments
    for func, hyperparams in experiments:
        func(args, hyperparams)

    print("Time taken:", time.time() - start)
