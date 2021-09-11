import gym
from frozen_lake2 import FrozenLake2Env
from stable_baselines3.common.env_util import make_vec_env

from agents import qlearning


def run_always_down():
    env = FrozenLake2Env(map_name="5x5")
    GO_DOWN = 0
    n_rollout_steps = 20
    for step in range(n_rollout_steps):
      obs, reward, done, info = env.step(GO_DOWN)
      print(f"Step: {step}. Obs: {obs}. Reward: {reward}. Done: {done}")
      env.render()
      if done:
        print("Rollout finished.")
        break

def run():
    env = FrozenLake2Env(map_name="5x5", slip_factor=0.5)
    # TODO wrap the environment to get the global seed working
    # env = make_vec_env(lambda: env, n_envs=1)
    options = {
        "seed": 42,
        "lr": 0.5,
        "gamma": 1,
        "n_episodes": 1000,
        "n_rollout_steps": 100,
    }

    qvals = qlearning.learn(env, **options)
    print(qvals)

if __name__ == "__main__":
    run()
    # run_always_down()