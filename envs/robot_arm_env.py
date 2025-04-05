import gymnasium as gym  # Updated to Gymnasium
import pybullet as p
import numpy as np
import pybullet_data
import time  # Added missing import
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



    def step(self, action):
        # Convert normalized actions to actual joint positions/velocities
        MAX_JOINT_CHANGE = 0.05
        scaled_action = action * MAX_JOINT_CHANGE
        
        # Get current joint positions
        current_joint_positions = [p.getJointState(self.robot, i)[0] for i in range(6)]
        
        # Calculate target positions by adding scaled actions
        target_positions = [current + change for current, change in zip(current_joint_positions, scaled_action)]
        
        # Apply the actions using position control
        for i in range(6):
            p.setJointMotorControl2(
                bodyIndex=self.robot,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_positions[i],
                maxVelocity=1.0  # Add velocity limit for stability
            )
        
        # Step the simulation with proper timing
        for _ in range(10):  # Multiple substeps for stability
            p.stepSimulation()
        
        # Use a more precise sleep method
        start_time = time.time()
        while (time.time() - start_time) < (1./240.):
            pass

        # Get new state after action
        obs = []
        for i in range(6):
            joint_info = p.getJointState(self.robot, i)
            obs.extend([joint_info[0], joint_info[1]]) #pos, vel

        obs = np.array(obs) #new state
        
        # Calculate reward based on distance to target
        target_position = np.array([0.5,0.5,0.5]) 
        end_effector_pos = np.array(p.getLinkState(self.robot, 5)[0]) #current pos of end effector 
        reward = -np.linalg.norm(end_effector_pos - target_position) #closer distance -> less negative reward

        # Check if episode is done
        done = -1 * reward < 0.05 #done if end effector is within 5cm of the target

        # Additional info for debugging
        info = {
            'distance_to_target': np.linalg.norm(end_effector_pos - target_position),
            'joint_positions': current_joint_positions
        }
        
        return obs, reward, done, info

    
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


