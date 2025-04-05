import gymnasium as gym  # Updated to Gymnasium
import pybullet as p
import numpy as np
import pybullet_data
from gymnasium import spaces  # Updated to Gymnasium

class RobotArmEnv(gym.Env):

    def __init__(self):
        super().__init__()
        
        self.physics_client = p.connect(p.GUI) #stores GUI Simulation's ID 
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        #loads robot and plane
        self.plane = p.loadURDF("plane.urdf") #flat plane
        self.robot = p.loadURDF("envs/ur5.urdf", basePosition = [0,0,0.05], useFixedBase=True) #Universal Robots's UR5 robotic arm, use fixed base for it not to fall trough the plane ground

        
        p.setGravity(0,0,-9.81)

        #defines agent's action space, shape = 6 for UR5's 6 joints (DOF), this vector is totally random.
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,),dtype=np.float32)   

        #defines agent's observation space, shape = 12 for each joint's position & velocity 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype= np.float32) 



    def step(self, action): #action = 6d action vector
        
        """Apply action, simulate, and return new state, reward, and done bool."""

        #applies action to each joint 
        for i in range (6):
            p.setJointMotorControl2(self.robot, i, p.POSITION_CONTROL, targetPosition=action[i])
        
        #advances sim by one step
        p.stepSimulation()

        #gets observation of the action
        obs = []
        for i in range(6):
            joint_info = p.getJointState(self.robot, i)
            obs.extend([joint_info[0], joint_info[1]]) #pos, vel

        obs = np.array(obs) #new state
        
        #reward
        target_position = np.array([0.5,0.5,0.5]) 
        end_effector_pos = np.array(p.getLinkState(self.robot, 5)[0]) #current pos of end effector 
        reward = -np.linalg.norm(end_effector_pos - target_position) #closer distance -> less negative reward

        #done bool
        done = -1 * reward < 0.05 #done if end effector is within 5cm of the target

        return obs, reward, done, {} #{} = debug dic

    
    def reset(self, *, seed=None, options=None):  # Updated reset signature for Gymnasium
        """Resets env to initial conditions, needed when agent restarts an episode """

        super().reset(seed=seed)  # Call parent reset for seeding
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.81)
        self.plane = p.loadURDF("plane.urdf") #flat plane
        self.robot = p.loadURDF("envs/ur5.urdf", basePosition = [0,0,0.05], useFixedBase=True)

        obs = []
        for i in range(6):
            joint_info = p.getJointState(self.robot, i)
            obs.extend([joint_info[0], joint_info[1]])  

        return np.array(obs), {}  # Gymnasium requires returning a tuple (obs, info)
    
    def render(self, mode ="human"):
        pass #we already have GUI running, still needed to avoid errors

    def close(self):
        p.disconnect()


