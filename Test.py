import pybullet as p 
import pybullet_data 
import time
from envs.robot_arm_env import RobotArmEnv


#starts PyBullet in GUI mode
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

#loads a plane and a robot
plane_id = p.loadURDF("plane.urdf")
robot_id = p.loadURDF("r2d2.urdf", basePosition=[0, 0, 0.5])

for _ in range(5000):  #runs for a longer time (about 20 sec)
    p.stepSimulation()
    time.sleep(1./240.)

#now disconnects after running for some time
p.disconnect()

def test_environment():
    try:
        #initializes the environment
        env = RobotArmEnv()
        print("Environment initialized successfully.")

        #resets the environment
        obs, info = env.reset()
        print("Environment reset successfully.")
        print("Initial observation:", obs)

        #takes a random step
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print("Step executed successfully.")
        print("Observation:", obs)
        print("Reward:", reward)
        print("Done:", done)

    except Exception as e:
        print(f"An error occurred during testing: {e}")

    finally:
        #closes the environment
        env.close()
        print("Environment closed.")

if __name__ == "__main__":
    test_environment()


