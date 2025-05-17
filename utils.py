"""
Utility functions for DQN training and evaluation of Dino game agent.
Includes frame preprocessing, stacking, and visualization.
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from collections import deque
from datetime import datetime
from PIL import Image
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("dino-utils")

def create_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data/models", exist_ok=True)
    os.makedirs("data/debug_frames", exist_ok=True)

def preprocess_frame(frame, size=(84, 84)):
    """
    Preprocess a frame for input to DQN:
    - Convert RGB to grayscale
    - Resize to given dimensions
    - Normalize pixel values
    
    Args:
        frame: RGB frame as numpy array (height, width, 3)
        size: Target size as (height, width)
        
    Returns:
        Preprocessed frame as numpy array (height, width, 1)
    """
    # If frame is a PIL Image, convert to numpy array
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)
    
    # Handle different frame types
    if isinstance(frame, bytes) or isinstance(frame, str):
        # Convert bytes or base64 to numpy array
        try:
            nparr = np.frombuffer(frame, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except:
            # Try to load as PIL Image
            try:
                image = Image.open(io.BytesIO(frame))
                frame = np.array(image)
            except:
                logger.error("Could not convert frame to numpy array")
                return np.zeros((*size, 1), dtype=np.float32)
    
    # Convert RGB to grayscale
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Resize to target size
    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    
    # Normalize pixel values to range [0, 1]
    frame = frame.astype(np.float32) / 255.0
    
    # Return as (height, width, 1)
    return frame.reshape(*size, 1)

def stack_frames(stacked_frames, frame, is_new_episode, stack_size=4):
    """
    Stack frames to provide temporal information.
    
    Args:
        stacked_frames: Deque of stacked frames or None
        frame: Current preprocessed frame
        is_new_episode: Whether this is a new episode
        stack_size: Number of frames to stack
        
    Returns:
        Stacked frames as numpy array (stack_size, height, width)
    """
    # Preprocess frame if needed
    if frame.shape != (84, 84, 1):
        frame = preprocess_frame(frame)
    
    if is_new_episode or stacked_frames is None:
        # Create new stacked frames
        stacked_frames = deque([np.zeros_like(frame) for _ in range(stack_size)], maxlen=stack_size)
        # Fill with current frame
        for _ in range(stack_size):
            stacked_frames.append(frame)
    else:
        # Add current frame to stacked frames
        stacked_frames.append(frame)
    
    # Stack frames along first dimension and return
    return np.stack(stacked_frames, axis=0).squeeze()

def create_state_tensor(state, device):
    """
    Convert a state (stacked frames) to a PyTorch tensor.
    
    Args:
        state: Stacked frames as numpy array (stack_size, height, width)
        device: PyTorch device
        
    Returns:
        State as PyTorch tensor (1, stack_size, height, width)
    """
    return torch.FloatTensor(state).unsqueeze(0).to(device)

def save_debug_frame(frame, step, episode, dir_path="data/debug_frames"):
    """
    Save a frame for debugging purposes.
    
    Args:
        frame: Frame to save
        step: Current step
        episode: Current episode
        dir_path: Directory to save frame
    """
    os.makedirs(dir_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{dir_path}/ep{episode}_step{step}_{timestamp}.png"
    
    # Convert to suitable format for saving
    if isinstance(frame, np.ndarray):
        if frame.dtype == np.float32 and frame.max() <= 1.0:
            # Normalize to 0-255 range
            frame = (frame * 255).astype(np.uint8)
        
        if len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
            # Grayscale image
            if len(frame.shape) == 3:
                frame = frame.squeeze()
            cv2.imwrite(filename, frame)
        else:
            # RGB image
            cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    else:
        logger.warning(f"Could not save frame of type {type(frame)}")

def plot_rewards(rewards, save_path="data/rewards_plot.png", window_size=10):
    """
    Plot rewards from training and save the figure.
    
    Args:
        rewards: List of rewards
        save_path: Path to save the figure
        window_size: Window size for moving average
    """
    plt.figure(figsize=(12, 6))
    
    # Plot rewards
    plt.plot(rewards, alpha=0.6, label="Rewards")
    
    # Plot moving average
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(np.arange(window_size-1, len(rewards)), moving_avg, label=f"{window_size}-episode Moving Average")
    
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Rewards")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.savefig(save_path)
    plt.close()
    
    logger.info(f"Rewards plot saved to {save_path}")

class ReplayBuffer:
    """
    Experience replay buffer for DQN.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Batch of transitions
        """
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in batch])
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

def update_progress(episode, total_episodes, avg_reward, epsilon, loss=None):
    """
    Print training progress.
    
    Args:
        episode: Current episode
        total_episodes: Total number of episodes
        avg_reward: Average reward
        epsilon: Current epsilon
        loss: Current loss
    """
    progress = int(20 * episode / total_episodes)
    progress_bar = "█" * progress + "░" * (20 - progress)
    loss_str = f", Loss: {loss:.4f}" if loss is not None else ""
    logger.info(f"Episode: {episode}/{total_episodes} [{progress_bar}] | Avg Reward: {avg_reward:.2f} | Epsilon: {epsilon:.2f}{loss_str}")

# Initialize directories when module is imported
create_directories() 