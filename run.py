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
        print(s)
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

def run_value_iter(finite=False):
    seed = 41

    options = {
        "seed": seed,
        "lr": 0.5,
        "gamma": 1,
        "epsilon": 0.1,
        "n_episodes": 1000,
        "n_rollout_steps": 100,
    }

    env = FrozenLake(map_name="5x5", slip=0.5)
    # Set for reproducibility
    env.seed(seed)
    env.action_space.seed(seed)

    if finite:
        optimal_vals = solver.value_iter_finite(env, options["gamma"])
        # Easier to view
        print(np.swapaxes(optimal_vals, 0, 1).reshape(-1, env.nrow, env.ncol))
        pol = solver.optimal_policy_finite(env, options["gamma"])
        for t in pol:
            print(f"{t+1} steps remaining\n{pol[t]}.")
    else:
        optimal_vals, n_iter = solver.value_iter(env, options["gamma"])
        print(optimal_vals.reshape((env.nrow, env.ncol)))
        pol = solver.optimal_policy(env, options["gamma"])
        print(pol)
        print("Number of iterations:", n_iter)

def run_qlearning(qt=False):
    seed = 42

    options = {
        "seed": seed,
        "lr": 0.5,
        "gamma": 1,
        "epsilon": 0.1,
        "n_episodes": 2000,
        "n_rollout_steps": 100,
    }

    env = FrozenLake(map_name="5x5", slip=0.5, seed=42)
    # env = gym.make("FrozenLake-v1", slip=0.5, seed=seed)

    ql = QLearning(env, **options)
    ql.learn()
    print(ql.q)
    rollout_model(env, ql, num_eps=1, horizon=None)

def run_qtlearning():
    seed = 42

    options = {
        "seed": seed,
        "lr": 0.5,
        "gamma": 1,
        "epsilon": 0.1,
        "n_episodes": 20000,
        "horizon": 10,
    }

    env = FrozenLake(map_name="5x5", slip=0.5, seed=seed)

    qtl = QTLearning(env, **options)
    qtl.learn(plain=False)
    # Show q-values for one timestep remaining
    print(qtl.q[:, :, 0])
    rollout_model(env, qtl, num_eps=1, horizon=7)

def run_rm():
    # TODO: Fix seed bug - giving same results for each non-None seed
    seed = 42

    options = {
        "seed": None,
        "lr": 0.5,
        "gamma": 0.9,
        "epsilon": 0.1,
        "n_episodes": 5000,
        "n_rollout_steps": 100,
        "use_crm": True,
        "use_rs": False,
    }

    rm_env = FrozenLakeRMEnv(map_name="5x5", slip=0.5)
    # rm_env = RewardMachineWrapper(rm_env, args.use_crm, args.use_rs, args.gamma, args.rs_gamma)
    rm_env = RewardMachineWrapper(rm_env, options["use_crm"], options["use_rs"], options["gamma"], 1)
    # rm_env = RewardMachineWrapper(rm_env, options["use_crm"], options["use_rs"], 0.9, 0.9)
    rm_env.seed(options["seed"])

    ql = QLearning(rm_env, **options)
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
    # run_value_iter(finite=False)
    # run_qlearning()
    run_rm()
    # run_qtlearning()
    # run_always_down()
    # run_a2c()
    print("Time taken:", time.time() - start)
