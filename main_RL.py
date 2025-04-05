
import os
import logging
import pybullet as p

#sets environment variable to avoid library duplication errors
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#imports custom environment and PPO algorithm
from envs.robot_arm_env import RobotArmEnv
from stable_baselines3 import PPO

#configures logging for better monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#defines training parameters
TOTAL_TIMESTEPS = 10_000  #number of timesteps for training
TENSORBOARD_LOG_DIR = "./ppo_robot_tensorboard/"  #directory for TensorBoard logs (uses 'tensorboard --logdir=./ppo_robot_tensorboard' to view)
MODEL_SAVE_PATH = "ppo_robot_arm"  #path to save the trained model

def main():
    try:
        #ensures no existing PyBullet connections to prevent memory leaks
        if p.isConnected():
            p.disconnect()
            
        #initializes the custom robot arm environment (creates PyBullet instance)
        env = RobotArmEnv()
        
        #creates a PPO (Proximal Policy Optimization) agent with the following configuration:
        model = PPO(
            policy="MlpPolicy",          #multi-layer Perceptron (neural network) policy
            env=env,                     #our custom robot environment
            verbose=1,                   #shows training statistics
            tensorboard_log=TENSORBOARD_LOG_DIR,  #where to save training metrics
            device="auto",               #uses GPU if available, otherwise CPU
            #hyperparameters for training stability:
            learning_rate=3e-4,          #step size for optimization (Adam optimizer)
            n_steps=2048,               #number of steps to run for each environment per update
            batch_size=64,              #minibatch size for training
            n_epochs=10,                #number of epochs when optimizing the surrogate loss
            gamma=0.99,                 #discount factor for future rewards
            gae_lambda=0.95,            #factor for trade-off of bias vs variance for GAE
            clip_range=0.2              #clipping parameter for PPO loss
        )

        logger.info("Starting training...")
        #trains the agent and saves metrics to tensorboard
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,  #total number of interaction steps
            progress_bar=True                 #show progress during training
        )

        logger.info("Training completed. Saving model...")
        #save the trained model for later use
        model.save(MODEL_SAVE_PATH)
        logger.info("Model saved successfully.")

    except Exception as e:
        #logs any errors that occur during training
        logger.error(f"An error occurred: {e}")
        raise

    finally:
        #ensures proper cleanup of resources
        env.close()                     #closes the environment and releases resources
        if p.isConnected():
            p.disconnect()              #ensures PyBullet is properly disconnected

if __name__ == "__main__":
    main()


