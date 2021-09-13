import time
import gym
from frozen_lake2 import FrozenLake2
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import A2C

from agents.qlearning import QLearning


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
    # With gamma<1, agent learns to go DownRight in s3
    model = A2C('MlpPolicy', env, gamma=0.99, verbose=1)
    model.learn(total_timesteps=10000)

    rollout_model(env, model)

def rollout_model(env, model, num_eps=1):
    for ep in range(num_eps):
        state = env.reset()
        env.render()
        while True:
            action, _ = model.predict(state, deterministic=True)
            new_state, reward, done, info = env.step(action)
            env.render()
            state = new_state
            if done:
                print("Finished ep")
                break

def run():
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

    ql = QLearning(env, **options)
    ql.learn()
    print(ql.q)

    rollout_model(env, ql, num_eps=1)

if __name__ == "__main__":
    start = time.time()
    run()
    # run_always_down()
    # run_a2c()
    print("Time taken:", time.time() - start)
