from envs.robot_arm_env import RobotArmEnv
import time

#instance of env 
env = RobotArmEnv()

#initializes sim
obs = env.reset()
done = False

#simulates 1000 steps of random interaction with the robot arm environment
for _ in range(1000): 
    action = env.action_space.sample() #random action 
    obs, reward, done, _ = env.step(action) #simulates one step

    #print reward occiasonally
    if _ % 100 == 0:
        print(f"Step {_}: reward = {reward:.4f}")

    if done:
        print("Target position reached within 5cm, end of episode.")
        break

    time.sleep(1./240.) #sim runs at 240Hz, i.e. each step takes 1/240 s 

env.close()
