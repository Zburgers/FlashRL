"""
Configuration file for DQN training and evaluation of the Dino game agent.
"""
import os
import torch

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "data", "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Environment parameters
FRAME_STACK = 4           # Number of frames to stack for creating the state
FRAME_SIZE = (84, 84)     # Size of preprocessed frames (height, width)

# DQN parameters
BATCH_SIZE = 32           # Batch size for training
GAMMA = 0.99              # Discount factor
REPLAY_MEMORY_SIZE = 50000 # Replay buffer size
TARGET_UPDATE = 10        # Update target network every N episodes
LEARNING_RATE = 1e-4      # Learning rate for optimizer

# Training parameters
EPISODES = 3000           # Total number of episodes to train
SAVE_EVERY = 50           # Save model every N episodes
EVAL_EVERY = 50           # Evaluate model every N episodes

# Exploration parameters
EPSILON_START = 1.0       # Starting epsilon for exploration
EPSILON_END = 0.01        # Minimum epsilon
EPSILON_DECAY = 500       # Number of episodes to decay epsilon over

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Define model paths
MODEL_PATH = os.path.join(MODEL_DIR, "dqn_dino.pth")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "dqn_dino_best.pth")

# Evaluation parameters
EVAL_EPISODES = 5         # Number of episodes to evaluate
RENDER_EVAL = True        # Whether to render during evaluation 