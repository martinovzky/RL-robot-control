import gymnasium as gym  #updates to Gymnasium
import pybullet as p
import numpy as np
import pybullet_data
import time  #adds missing import
from gymnasium import spaces  #updates to Gymnasium

class RobotArmEnv(gym.Env):

    def __init__(self):
        super().__init__()
        
        self.physics_client = p.connect(p.GUI) #stores GUI simulation's ID 
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        #loads robot and plane
        self.plane = p.loadURDF("plane.urdf") #flats plane
        self.robot = p.loadURDF("envs/ur5.urdf", basePosition = [0,0,0.05], useFixedBase=True) #universals Robot's UR5 robotic arm, uses fixed base for it not to fall through the plane ground

        p.setGravity(0,0,-9.81)

        #defines agent's action space, shape = 6 for UR5's 6 joints (DOF), this vector is totally random
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,),dtype=np.float32)   

        #defines agent's observation space, shape = 12 for each joint's position & velocity 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype= np.float32) 

    def step(self, action):
        #converts normalized actions to actual joint positions/velocities
        MAX_JOINT_CHANGE = 0.05
        scaled_action = action * MAX_JOINT_CHANGE
        
        #gets current joint positions
        current_joint_positions = [p.getJointState(self.robot, i)[0] for i in range(6)]
        
        #calculates target positions by adding scaled actions
        target_positions = [current + change for current, change in zip(current_joint_positions, scaled_action)]
        
        #applies the actions using position control
        for i in range(6):
            p.setJointMotorControl2(
                bodyIndex=self.robot,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_positions[i],
                maxVelocity=1.0  #adds velocity limit for stability
            )
        
        #step the simulation with proper timing
        for _ in range(10):  #multiple substeps for stability
            p.stepSimulation()
        
        #uses a more precise sleep method
        start_time = time.time()
        while (time.time() - start_time) < (1./240.):
            pass

        #gets new state after action
        obs = []
        for i in range(6):
            joint_info = p.getJointState(self.robot, i)
            obs.extend([joint_info[0], joint_info[1]]) #pos, vel

        obs = np.array(obs) #new state
        
        #calculates reward based on distance to target
        target_position = np.array([0.5,0.5,0.5]) 
        end_effector_pos = np.array(p.getLinkState(self.robot, 5)[0]) #current pos of end effector 
        reward = -np.linalg.norm(end_effector_pos - target_position) #closer distance -> less negative reward

        #checks if episode is done
        done = -1 * reward < 0.05 #done if end effector is within 5cm of target

        #additional info for debugging
        info = {
            'distance_to_target': np.linalg.norm(end_effector_pos - target_position),
            'joint_positions': current_joint_positions
        }
        
        return obs, reward, done, info

    def reset(self, *, seed=None, options=None):  #updates reset signature for Gymnasium
        """resets env to initial conditions, needs when agent restarts an episode """

        super().reset(seed=seed)  #calls parent reset for seeding
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.81)
        self.plane = p.loadURDF("plane.urdf") #flat plane
        self.robot = p.loadURDF("envs/ur5.urdf", basePosition = [0,0,0.05], useFixedBase=True)

        obs = []
        for i in range(6):
            joint_info = p.getJointState(self.robot, i)
            obs.extend([joint_info[0], joint_info[1]])  

        return np.array(obs), {}  #gymnasium requires returning a tuple (obs, info)
    
    def render(self, mode ="human"):
        pass #we already have GUI running, still needed to avoid errors

    def close(self):
        p.disconnect()


