import time
import numpy as np
import gym
from frozen_lake2 import FrozenLake2
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import A2C

from agents.qlearning import QLearning
from agents.qtlearning import QTLearning
import solver


def run_always_down():
    env = FrozenLake2(map_name="5x5", slip_factor=0.5)
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
    env = FrozenLake2(map_name="5x5", slip_factor=0.5)
    # With gamma<1, agent learns to only go DownRight in s3
    model = A2C('MlpPolicy', env, gamma=0.99, verbose=1)
    model.learn(total_timesteps=10000)
    rollout_model(env, model)

def rollout_model(env, model, num_eps=1, horizon=1e7):
    for ep in range(num_eps):
        s = env.reset()
        env.render()
        t = 0
        while True:
            if horizon:
                a, _ = model.predict((s, t), deterministic=True)
            else:
                a, _ = model.predict(s, deterministic=True)
            new_s, reward, done, info = env.step(a)
            env.render()
            s = new_s

            t += 1
            if done or t == horizon:
                print("Finished ep")
                break

def run_value_iter(finite=False):
    seed = 42

    options = {
        "seed": seed,
        "lr": 0.5,
        "gamma": 1,
        "epsilon": 0.1,
        "n_episodes": 1000,
        "n_rollout_steps": 100,
    }

    env = FrozenLake2(map_name="5x5", slip_factor=0.5)
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
        "n_episodes": 50000,
        "n_rollout_steps": 100,
    }

    env = FrozenLake2(map_name="5x5", slip_factor=0.5)
    # Set for reproducibility
    env.seed(seed)
    env.action_space.seed(seed)

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
        "n_episodes": 50000,
        "horizon": 6,
    }

    env = FrozenLake2(map_name="5x5", slip_factor=0.5)
    # Set for reproducibility
    env.seed(seed)
    env.action_space.seed(seed)

    qtl = QTLearning(env, **options)
    qtl.learn()
    # Show q-values for one timestep remaining
    print(qtl.q[:, :, 0])
    rollout_model(env, qtl, num_eps=1, horizon=4)



if __name__ == "__main__":
    start = time.time()
    # run_value_iter(finite=False)
    run_qlearning()
    # run_qtlearning()
    # run_always_down()
    # run_a2c()
    print("Time taken:", time.time() - start)
