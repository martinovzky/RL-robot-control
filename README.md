# RL Robot Arm Control

A reinforcement learning project to control a simulated robotic arm using Stable-Baselines3 and a custom OpenAI Gym environment.

## Overview

This project trains a robotic arm using **Proximal Policy Optimization (PPO)** to perform goal-directed tasks in a simulated environment.

The custom environment is implemented in `envs/robot_arm_env.py`. The agent is trained using Stable-Baselines3 and visualized with TensorBoard.

##  Requirements

- Python 3.8+
- `gym`
- `numpy`
- `torch`
- `stable-baselines3`

Install dependencies:

```bash
pip install -r requirements.txt
```

##  How to Train

To train the agent:

```bash
python main_RL.py
```

This runs PPO for 10,000 timesteps and saves the model as `ppo_robot_arm.zip`.


## Notes

- The PPO policy used is `"MlpPolicy"` (multi-layer perceptron).
- You can increase `total_timesteps` in `main_RL.py` to improve performance.
- `KMP_DUPLICATE_LIB_OK` is set to avoid library loading issues on some macOS systems.

## Custom Environment

Make sure `robot_arm_env.py` correctly implements the OpenAI Gym interface:
- `reset()`
- `step(action)`
- `render()`
- `observation_space`
- `action_space`

## License

MIT License