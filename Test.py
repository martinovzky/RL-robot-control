import pybullet as p 
import pybullet_data 
import time


# Start PyBullet in GUI mode
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load a plane and a robot
plane_id = p.loadURDF("plane.urdf")
robot_id = p.loadURDF("r2d2.urdf", basePosition=[0, 0, 0.5])

for _ in range(5000):  # Runs for a longer time (about 20 sec)
    p.stepSimulation()
    time.sleep(1./240.)

# Now disconnect after running for some time
p.disconnect()


