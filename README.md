# RL Robot Arm Control

A reinforcement learning project to control a simulated robotic arm using **Stable-Baselines3** and a custom **Gymnasium** environment.

## Overview

This project demonstrates how to train a robotic arm to perform goal-directed tasks in a simulated environment using **Proximal Policy Optimization (PPO)**. The robotic arm is simulated using **PyBullet**, and the training process is visualized with **TensorBoard**.

The custom environment (`envs/robot_arm_env.py`) is built to simulate the UR5 robotic arm, with a focus on learning to move its end effector to a target position. The project showcases how reinforcement learning can be applied to robotics in a simulated setting.

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

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Train

To train the agent:

```bash
python main_RL.py
```

This will:
1. Initialize the custom robotic arm environment.
2. Train the PPO agent for **10,000 timesteps**.
3. Save the trained model as `ppo_robot_arm.zip`.

You can monitor the training process using TensorBoard:

```bash
tensorboard --logdir=ppo_robot_tensorboard/
```

## Custom Environment

The custom environment (`robot_arm_env.py`) simulates the UR5 robotic arm and adheres to the Gymnasium interface. It includes:
- **`reset()`**: Resets the environment to its initial state.
- **`step(action)`**: Applies an action, advances the simulation, and returns the new state, reward, and done flag.
- **`render()`**: (Optional) Renders the environment (PyBullet GUI is already enabled).
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

## Future Improvements

- Add support for more complex tasks, such as obstacle avoidance or trajectory following.
- Implement curriculum learning to gradually increase task difficulty.
- Extend the environment to support continuous control with torque-based actions.

## License

This project is licensed under the **MIT License**.