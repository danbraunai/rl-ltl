# rl-ltl
This repository contains the code used in my Honours Dissertation "Getting to school on time: Completing linear temporal logic objectives before a fixed deadline with reinforcement learning". This work introduces QTRM-learning (Q-learning with Time and Reward Machines), an algorithm extending the [work](https://arxiv.org/abs/2010.03950) [on](http://proceedings.mlr.press/v80/icarte18a.html) [reward machines](https://www.ijcai.org/Proceedings/2019/840) to learn optimal policies in finite horizon problems with few samples.

The modules in *reward_machines/* were taken directly from [https://github.com/RodrigoToroIcarte/reward_machines/tree/master/reward_machines/reward_machines](https://github.com/RodrigoToroIcarte/reward_machines/tree/master/reward_machines/reward_machines), and the agents in *q_inf_learning.py* and *q_fin_learning.py* are based on those in [https://github.com/RodrigoToroIcarte/reward_machines/tree/master/reward_machines/rl_agents/qlearning](https://github.com/RodrigoToroIcarte/reward_machines/tree/master/reward_machines/rl_agents/qlearning)


## Environment
The repository contains a custom FrozenLake environment (based on [FrozenLake-v0](https://gym.openai.com/envs/FrozenLake-v0/)) that allows specification of tasks using reward machine templates. The environment is described as follows:

> You and your friends were tossing around a frisbee on a (mostly) frozen lake when your friend fell through the ice trying to take an epic catch. Luckily, somebody left rope on the lake for this exact scenario. You must grab a rope, and go over and pull your friend out of the water. The ice is slippery, so you will not always move in the direction you intend. There are also some areas where the ice is thinner. If you step into one of those traps, you will fall in and drown.

The parameters of the environment are described as follows:
- Actions: The agent can request to move one step in any of the 8 directions (Right, Down-Right, Down, ...).
- Light Blue Squares: Slippery ice, with 50% chance (this is adjustable) of an action leading to the desired state, and 50% chance of transitioning to a state uniformly from all 8 directions.
- Green Squares: Safe path, with all actions deterministic and leading to the desired state.
- Dark Blue Squares: Trap, where the agent would fall in and get stuck perpetually.
- Edge Conditions: If the direction of movement is off the grid, the agent will remain in place.

The repository contains three tasks for this environment, with reward machine templates and sample images shown in _envs/_. To create a new task, simply place a new reward machine template in this folder.

## Agents
Reinforcement learning agents are placed in _agents/_, with the repository currently supporting CRM and the Q-learning baseline from [https://github.com/RodrigoToroIcarte/reward_machines](https://github.com/RodrigoToroIcarte/reward_machines) for the infinite horizon problem (*agents/q_inf_learning.py*), and the Q-learning baseline, QT-learning, QRM-learning, and QTRM-learning for the finite horizon problem (*agents/q_fin_learning.py*).

The FrozenLake environment can be ran with any agent that interfaces with [OpenAI Gym](https://github.com/openai/gym), such as those found in [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3). For example, A2C from Stable Baselines 3 can be run on the environment by uncommenting run_a2c() in *run.py* (and install the stable baselines package). However, the reward machine specific algorithms in *agents/* are needed to make use of rewards from the reward machine in this environment.

## Requirements
The required Python packages can be installed by running the following from the root directory:
```
pip3 install -r requirements.txt
```

## Running Experiments
To reproduce the results for all experiments shown in the paper, including the corresponding figures, run
```
python3 run.py --seed 42 --num_processes [NUM_PROCESSES]
```
It is recommended to use a machine with many CPUs, as the whole suite of experiments may take up to 2 hours on a machine with 64 virtual CPUs.

All hyperparameters are specified in *hyperparams.yml*. To run a single experiment using the same hyperparameters as the dissertation, run for example:
```
python3 run.py --seed 42 --num_processes [NUM_PROCESSES] --params thesis --experiments q_fin_task3
```
The --params argument specifies the top level key in *hyperparams.yml*, and the --experiments argument specifies the second level key and corresponds to the name of the experiment(s). For example, *--params play --experiments q_fin_task2 vi_fin_task1* will run the finite horizon experiments for Task 2, as well as value iteration for Task 1, using the corresponding hyperparameters as specified under the *play* key in *hyperparams.yml*.

The results of the experiments will be output to *results/*, although this can be changed with the cmd argument *--out_dir*.

## Repository Structure
```bash
.
├── agents
│   ├── q_fin_learning.py
│   └── q_inf_learning.py
├── envs
│   ├── FrozenLake1.png
│   ├── FrozenLake2.png
│   ├── FrozenLake3.png
│   ├── frozen_lake.py
│   ├── __init__.py
│   ├── rm_t1_frozen_lake.txt
│   ├── rm_t2_frozen_lake.txt
│   └── rm_t3_frozen_lake.txt
├── hyperparams.yml
├── plotting.py
├── README.md
├── requirements.txt
├── results
│   ├── finite_Task2_rew_vs_samples_11-11-21.png
│   ├── finite_Task2_rew_vs_samples_11-11-21_zoomed.png
│   ├── finite_Task2_rew_vs_updates_11-11-21.png
│   ├── finite_Task3_rew_vs_samples_11-11-21.png
│   ├── finite_Task3_rew_vs_samples_11-11-21_zoomed.png
│   ├── finite_Task3_rew_vs_updates_11-11-21.png
│   ├── fin_t2_11-11-21.json
│   ├── fin_t3_11-11-21.json
│   ├── infinite_Task2_rew_vs_samples_12-11-21.png
│   ├── infinite_Task2_rew_vs_updates_12-11-21.png
│   ├── infinite_Task3_rew_vs_samples_12-11-21.png
│   ├── infinite_Task3_rew_vs_updates_12-11-21.png
│   ├── inf_t2_12-11-21.json
│   └── inf_t3_12-11-21.json
├── reward_machines
│   ├── __init__.py
│   ├── reward_functions.py
│   ├── reward_machine.py
│   ├── reward_machine_utils.py
│   └── rm_environment.py
├── run.py
├── solver.py
├── utils.py
```