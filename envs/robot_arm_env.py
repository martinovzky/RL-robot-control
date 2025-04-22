import gymnasium as gym  
import pybullet as p
import numpy as np
import pybullet_data
import time  
from gymnasium import spaces  

class RobotArmEnv(gym.Env):

    def __init__(self, headless=False):
        super().__init__()
        
        # Use DIRECT mode for headless operation, GUI for visualization
        connection_mode = p.DIRECT if headless else p.GUI
        self.physics_client = p.connect(connection_mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Use simpler physics settings - key fix for DummyVecEnv issue
        p.setTimeStep(1./60.)
        p.setPhysicsEngineParameter(numSolverIterations=5)

        #loads robot and plane
        self.plane = p.loadURDF("plane.urdf") #flats plane
        self.robot = p.loadURDF("envs/ur5.urdf", basePosition = [0,0,0.05], useFixedBase=True) #universals Robot's UR5 robotic arm, uses fixed base for it not to fall through the plane ground

        p.setGravity(0,0,-9.81)

        #defines agent's action space, shape = 6 for UR5's 6 joints (DOF), this vector is totally random
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,),dtype=np.float32)   

        #defines agent's observation space, shape = 12 for each joint's position & velocity 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype= np.float32)
        
        #headless mode parameter
        self.headless = headless 
        
        #few simulation steps to stabilize
        for _ in range(3):
            p.stepSimulation()

    def step(self, action):
        #converts normalized actions to actual joint positions/velocities
        MAX_JOINT_CHANGE = 0.15
        scaled_action = action * MAX_JOINT_CHANGE #for stability, limits the change in joint positions to a maximum of 0.15 radians per step
        
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
                maxVelocity=0.5  #adds velocity limit for stability
            )
        
        #step the simulation with proper timing
        for _ in range(5):  #5 physical steps per RL step
            p.stepSimulation()
        
        # Simplified timing - key fix for headless mode
        time.sleep(1./240.)
        
        #gets new state after action
        obs = []
        for i in range(6):
            joint_info = p.getJointState(self.robot, i)
            obs.extend([joint_info[0], joint_info[1]]) #pos, vel

        obs = np.array(obs) #state
        
        #calculates reward based on distance to target
        target_position = np.array([0.5,0.5,0.5]) 
        end_effector_pos = np.array(p.getLinkState(self.robot, 5)[0]) #current pos of end effector (link that interacts with environment)
        reward = -np.linalg.norm(end_effector_pos - target_position) #the closer to the target -> the less negative reward

        #checks if episode is done
        done = -1 * reward < 0.05  #done if end effector is within 5cm of target

        #additional info for debuging
        info = {
            'distance_to_target': np.linalg.norm(end_effector_pos - target_position),
            'joint_positions': current_joint_positions
        }
        
        return obs, reward, done, False, info

    def reset(self, *, seed=None, options=None):
        """resets env to initial conditions, needs when agent restarts an episode """

        super().reset(seed=seed)  #calls parent reset for seeding
        p.resetSimulation()  #resets the simulation
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.81)
        
        # Set physics parameters again
        p.setTimeStep(1./60.)
        p.setPhysicsEngineParameter(numSolverIterations=5)
        
        self.plane = p.loadURDF("plane.urdf") #flat plane
        self.robot = p.loadURDF("envs/ur5.urdf", basePosition = [0,0,0.05], useFixedBase=True)
        
        # Critical fix: stabilize after reset
        for _ in range(3):
            p.stepSimulation()

        obs = []
        for i in range(6):
            joint_info = p.getJointState(self.robot, i)
            obs.extend([joint_info[0], joint_info[1]])  

        return np.array(obs), {}  #gymnasium requires returning a tuple (obs, info)
    
    def render(self, mode ="human"):
        pass #we already have GUI running, still needed to avoid errors

    def close(self):
        if p.isConnected(self.physics_client):
            p.disconnect(self.physics_client)


