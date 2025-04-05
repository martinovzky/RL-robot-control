# Import necessary libraries and modules
import os
import logging

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
TENSORBOARD_LOG_DIR = "./ppo_robot_tensorboard/"  # Directory for TensorBoard logs
MODEL_SAVE_PATH = "ppo_robot_arm"  # Path to save the trained model

def main():
    try:
        # Initialize the custom robot arm environment
        env = RobotArmEnv()

        # Create a PPO agent with MLP policy and GPU support if available
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,  # Enable detailed logging during training
            tensorboard_log=TENSORBOARD_LOG_DIR,  # Log training metrics for TensorBoard
            device="auto"  # Automatically use GPU if available, otherwise CPU
        )

        logger.info("Starting training...")
        # Train the agent for the specified number of timesteps
        model.learn(total_timesteps=TOTAL_TIMESTEPS)

        logger.info("Training completed. Saving model...")
        # Save the trained model to the specified path
        model.save(MODEL_SAVE_PATH)
        logger.info("Model saved successfully.")

    except Exception as e:
        # Log any errors that occur during training or saving
        logger.error(f"An error occurred: {e}")

    finally:
        # Ensure the environment is closed properly
        env.close()

if __name__ == "__main__":
    main()


