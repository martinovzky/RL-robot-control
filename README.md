# RL Robot Arm Control

RL project to control a simulated robotic arm using Stable-Baselines3 and a custom Gymnasium environment.

## Overview

This project demonstrates how to train a robotic arm to perform goal-directed tasks in a simulated environment using **Proximal Policy Optimization (PPO)**. The robotic arm is simulated using PyBullet,
the custom environment (`envs/robot_arm_env.py`) is built to simulate the UR5 robotic arm, that has for goal to leanr how to move its end effector to a target position. 


The goal of this project was to build a basic robotic simulation environment, learn how to implement RL pipelines using broadly used RL algorithms and get familiar with the training mechanism of an RL model. 

## Features

- **Custom Environment**: Implements a Gymnasium-compatible environment for the UR5 robotic arm.
- **Reinforcement Learning**: Uses PPO from Stable-Baselines3 to train the agent.
- **Simulation**: Powered by PyBullet for realistic physics-based simulation.
- **Visualization**: Training metrics are logged to TensorBoard for real-time monitoring.

## Requirements

- Python 3.8+
- `gymnasium`
- `numpy`
- `torch`
- `stable-baselines3`
- `pybullet`
- PIL

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Train

To train the agent (in headless mode):

```bash
python main_RL_headless.py
```

This will:
1. Initialize the custom robotic arm environment.
2. Train the PPO agent for **2,500,000 timesteps**. (Much more efficient with accelerated compute)
3. Save the trained model as `ppo_robot_arm.zip`.

You can monitor the training process using TensorBoard:

```bash
tensorboard --logdir=ppo_robot_tensorboard/
```

## Custom Environment

The custom environment (`robot_arm_env.py`) simulates the UR5 robotic arm for the Gymnasium interface. It includes:
- **`reset()`**: Resets the environment to its initial state.
- **`step(action)`**: Applies an action, advances the simulation, and returns the new state, reward, and done flag.
- **`observation_space`**: Defines the state space (joint positions and velocities).
- **`action_space`**: Defines the action space (joint target positions).

### Reward Function
The reward is based on the distance between the end effector and the target position. The closer the end effector is to the target, the higher the reward.

### Done Condition
The episode ends when the end effector is within **5 cm** of the target position.

## Notes

- The PPO policy used is `"MlpPolicy"` (multi-layer perceptron).
- You can adjust `TOTAL_TIMESTEPS` in [`main_RL.py`](main_RL.py) to train the agent for a longer duration.
- The environment uses PyBullet for physics simulation, and inertial data has been added to the URDF file for stability.
- `KMP_DUPLICATE_LIB_OK` is set to avoid library loading issues on macOS systems.
- A PyBullet-GUI version (`main_RL.py`) is included in the run_scripts folder but isn't fully developped.

## Future Improvements

- Add support for more complex tasks like obstacle avoidance or trajectory following.
- Implement curriculum learning to gradually increase task difficulty.
- Finish the GU version.

## License

**MIT License**.
