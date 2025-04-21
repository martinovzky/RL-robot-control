import os
import logging
import pybullet as p
import time
import sys
import torch
import numpy as np
from PIL import Image

#avoids library duplication errors, pybullet sometimes crash becaause of this
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from envs.robot_arm_env import RobotArmEnv
from stable_baselines3 import PPO

# configures logging for better monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#hyperparameters
TOTAL_TIMESTEPS = 2000000  # increased for better learning
TENSORBOARD_LOG_DIR = "./ppo_robot_tensorboard_headless/"  
MODEL_SAVE_PATH = "ppo_robot_arm_headless"  

def save_pybullet_screenshot(filename, width=640, height=480):
    """Capture and save a screenshot from the PyBullet camera."""
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[1, 1, 1],
        cameraTargetPosition=[0, 0, 0],
        cameraUpVector=[0, 0, 1]
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=float(width)/height, nearVal=0.1, farVal=100.0
    )
    img_arr = p.getCameraImage(width, height, view_matrix, proj_matrix)
    rgb_array = np.reshape(img_arr[2], (height, width, 4))[:, :, :3]
    img = Image.fromarray(rgb_array)
    img.save(filename)

def main():
    try:
        print("Step 1: Starting...")
        sys.stdout.flush()
        
        #ensures no existing PyBullet connections
        if p.isConnected():
            p.disconnect()
            
        time.sleep(0.5)
            
        print("Step 2: Creating environment")
        sys.stdout.flush()
        #initialize with reduced parameters
        env = RobotArmEnv(headless=True)
        
        print("Step 3: Creating PPO model")
        sys.stdout.flush()
        
        #simpler PPO config 
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            tensorboard_log=TENSORBOARD_LOG_DIR,
            policy_kwargs={
                "net_arch": [dict(pi=[512, 512, 256], vf=[512, 512, 256])],  # deeper/wider net
                "activation_fn": torch.nn.ReLU  # use ReLU for non-linearity
            }, 
            device="cuda",  
            learning_rate=1e-4,  #lower for stability
            n_steps=16384,       #larger rollout buffer
            batch_size=4096,     #larger batch for GPU
            n_epochs=20,         #epochs per update
            gamma=0.995,         
            gae_lambda=0.97,     
            clip_range=0.15,     
            ent_coef=0.01,       #encourages exploration
            normalize_advantage=True  #stabilizes learning
        )

        print("TensorBoard: To monitor training, run the following command in your terminal:")
        print(f"tensorboard --logdir {TENSORBOARD_LOG_DIR}")
        print("Then open http://localhost:6006/ in your browser.\n")
        sys.stdout.flush()

        print("Step 4: Starting training")
        sys.stdout.flush()
        
        #try very small training batch first as test
        print("Running test (100 steps)...")
        sys.stdout.flush()
        model.learn(total_timesteps=100, log_interval=1)

        # Take a screenshot after test run
        screenshot_dir = "./screenshots"
        os.makedirs(screenshot_dir, exist_ok=True)
        screenshot_path = os.path.join(screenshot_dir, "test_run.png")
        save_pybullet_screenshot(screenshot_path)
        print(f"Screenshot saved to {screenshot_path}")
        sys.stdout.flush()

        print("Test successful, starting main training")
        sys.stdout.flush()
        
        #main training
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            log_interval=10,
            callback=lambda _locals, _globals: (
                save_pybullet_screenshot(
                    os.path.join(
                        screenshot_dir,
                        f"step_{_locals['self'].num_timesteps}.png"
                    )
                ) if _locals['self'].num_timesteps % 50000 == 0 else None
            )
        )

        print("Training completed")
        model.save(MODEL_SAVE_PATH)
        logger.info("Model saved")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("Cleaning up...")
        if 'env' in locals():
            env.close()
        if p.isConnected():
            p.disconnect()

if __name__ == "__main__":
    main()