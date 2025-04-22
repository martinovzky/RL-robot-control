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
TOTAL_TIMESTEPS = 2000000  
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

def create_callback(screenshot_dir):
    """Create a callback that saves screenshots and reports progress."""
    last_timesteps = 0
    
    def callback(_locals, _globals):
        nonlocal last_timesteps
        timesteps = _locals['self'].num_timesteps #current training step 
        
        #takes screenshot every 25k steps
        if timesteps % 25000 == 0 and timesteps > 0:
            screenshot_path = os.path.join(
                screenshot_dir,
                f"step_{timesteps}.png"
            )
            save_pybullet_screenshot(screenshot_path)
            print(f"Screenshot saved at step {timesteps}")
            
        #prints progress every 100k steps
        if timesteps - last_timesteps >= 100000:
            last_timesteps = timesteps
            print(f"Progress: {timesteps}/{TOTAL_TIMESTEPS} steps ({timesteps/TOTAL_TIMESTEPS*100:.1f}%)")
            sys.stdout.flush()

        #saves model every 25k step
        if timesteps % 25000 == 0:
            model = _locals['self']
            model.save(MODEL_SAVE_PATH)  s
            print(f"Model saved at step {timesteps}")
            
        return True
    
    return callback

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
        
        #creates env
        env = RobotArmEnv(headless=True)
        
        #screenshot directory
        screenshot_dir = "./screenshots"
        os.makedirs(screenshot_dir, exist_ok=True)
        
        #creates a progress callback (closure: inner function that retains access to outer vars like last_timesteps and screenshot_dir)
        progress_callback = create_callback(screenshot_dir)
        

         #if azure evicts me from training run, use saved model as resuming point
        if os.path.exists(f"{MODEL_SAVE_PATH}.zip"):
            print("Loading existing model...")
            sys.stdout.flush()
            model = PPO.load(MODEL_SAVE_PATH, env=env)

        else:
        
            print("Step 3: Creating model for training")
            sys.stdout.flush()

            #main training model with full hyperparams
            model = PPO(
                policy="MlpPolicy",
                env=env,
                verbose=1, #logs metrics
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
                gamma=0.995,         #discount factor
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
        print(f"Training for {TOTAL_TIMESTEPS} timesteps...")
        sys.stdout.flush()

        
        #main training
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            log_interval=10,
            callback=progress_callback
        )
        
        print("Training completed")
        print("Total timesteps trained:", model.num_timesteps)
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
