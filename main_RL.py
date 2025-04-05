# Import necessary libraries and modules
import os
import logging
import pybullet as p

# Set environment variable to avoid library duplication errors
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Import custom environment and PPO algorithm
from envs.robot_arm_env import RobotArmEnv
from stable_baselines3 import PPO

# Configure logging for better monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define training parameters
TOTAL_TIMESTEPS = 10_000  # Number of timesteps for training
TENSORBOARD_LOG_DIR = "./ppo_robot_tensorboard/"  # Directory for TensorBoard logs (use 'tensorboard --logdir=./ppo_robot_tensorboard' to view)
MODEL_SAVE_PATH = "ppo_robot_arm"  # Path to save the trained model

def main():
    try:
        # Ensure no existing PyBullet connections to prevent memory leaks
        if p.isConnected():
            p.disconnect()
            
        # Initialize the custom robot arm environment (creates PyBullet instance)
        env = RobotArmEnv()
        
        # Create a PPO (Proximal Policy Optimization) agent with the following configuration:
        model = PPO(
            policy="MlpPolicy",          # Multi-layer Perceptron (neural network) policy
            env=env,                     # Our custom robot environment
            verbose=1,                   # Show training statistics
            tensorboard_log=TENSORBOARD_LOG_DIR,  # Where to save training metrics
            device="auto",               # Use GPU if available, otherwise CPU
            # Hyperparameters for training stability:
            learning_rate=3e-4,          # Step size for optimization (Adam optimizer)
            n_steps=2048,               # Number of steps to run for each environment per update
            batch_size=64,              # Minibatch size for training
            n_epochs=10,                # Number of epochs when optimizing the surrogate loss
            gamma=0.99,                 # Discount factor for future rewards
            gae_lambda=0.95,            # Factor for trade-off of bias vs variance for GAE
            clip_range=0.2              # Clipping parameter for PPO loss
        )

        logger.info("Starting training...")
        # Train the agent and save metrics to tensorboard
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,  # Total number of interaction steps
            progress_bar=True                 # Show progress during training
        )

        logger.info("Training completed. Saving model...")
        # Save the trained model for later use
        model.save(MODEL_SAVE_PATH)
        logger.info("Model saved successfully.")

    except Exception as e:
        # Log any errors that occur during training
        logger.error(f"An error occurred: {e}")
        raise

    finally:
        # Ensure proper cleanup of resources
        env.close()                     # Close the environment and release resources
        if p.isConnected():
            p.disconnect()              # Ensure PyBullet is properly disconnected

if __name__ == "__main__":
    main()


