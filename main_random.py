from envs.robot_arm_env import RobotArmEnv
import time

#instance of env 
env = RobotArmEnv()

#initializes sim
obs = env.reset()
done = False

#simulates 1000 steps of random interaction with the robot arm environment
for i in range(1000): 
    action = env.action_space.sample() #random action 
    obs, reward, done, info = env.step(action) #simulates one step

    #print reward occiasonally
    if i % 100 == 0:
        print(f"Step {i}: reward = {reward:.4f}")

    if bool(done):
        print("Target position reached within 5cm, end of episode.")
        break #probably won't happen as all the actions are completely random


    time.sleep(1./240.) #sim runs at 240Hz, i.e. each step takes 1/240 s 

env.close()
