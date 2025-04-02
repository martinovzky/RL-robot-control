import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from envs.robot_arm_env import RobotArmEnv
from stable_baselines3 import PPO

#instance
env = RobotArmEnv()

#PPO agent 
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    tensorboard_log="./ppo_robot_tensorboard/", #for visualization
)   

#train agent
model.learn(total_timesteps=10_000)

#save
model.save("ppo_robot_arm")

env.close()


