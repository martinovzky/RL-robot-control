import gym
import pybullet as p
import numpy as np
from gym import spaces #defines action and observation spaces for RL

class RobotArmEnv(gym.Env):

    def __init__(self):
        super().__init__()
        
        self.physics_client = p.connect(p.GUI) #stores GUI Simulation's ID 
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        #load robot and plane
        self.plane = p.loadURDF("plane.urdf") #flat plane
        self.robot = p.loadURDF("ur5.urdf", basePosition = [0,0,0.5]) #we use Universal Robots's UR5 robotic arm

        #set gravity
        p.setGravity(0,0,-9.81)

        


 