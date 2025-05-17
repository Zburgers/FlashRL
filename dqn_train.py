"""
Deep Q-Network training script for Chrome Dino game.
This script trains a DQN agent to play the Chrome Dino game.
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import argparse
import random
import logging
import colorama
from colorama import Fore, Style
from collections import deque

# Import local modules
from dino_env import DinoEnv
from utils import (
    preprocess_frame, stack_frames, create_state_tensor, 
    save_debug_frame, plot_rewards, ReplayBuffer, update_progress
)
import config

# Initialize colorama
colorama.init(autoreset=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("dqn-train")

class DQN(nn.Module):
    """
    Deep Q-Network for the Dino game.
    Architecture:
    - Convolutional layers to process images
    - Fully connected layers for Q-value prediction
    """
    def __init__(self, input_shape, n_actions):
        """
        Initialize DQN.
        
        Args:
            input_shape: Shape of input frames (channels, height, width)
            n_actions: Number of possible actions
        """
        super(DQN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate size of conv output
        conv_out_size = self._get_conv_output(input_shape)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    
    def _get_conv_output(self, shape):
        """
        Calculate output size of convolutional layers.
        
        Args:
            shape: Input shape
            
        Returns:
            Size of flattened convolutional output
        """
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input batch of states (batch_size, channels, height, width)
            
        Returns:
            Q-values for each action (batch_size, n_actions)
        """
        conv_out = self.conv(x)
        conv_out = conv_out.view(conv_out.size(0), -1)
        return self.fc(conv_out)

def select_action(state, policy_net, epsilon, n_actions, device):
    """
    Select an action using epsilon-greedy policy.
    
    Args:
        state: Current state
        policy_net: Policy network
        epsilon: Exploration probability
        n_actions: Number of possible actions
        device: PyTorch device
        
    Returns:
        Selected action
    """
    if random.random() < epsilon:
        # Explore: select random action
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    else:
        # Exploit: select best action according to policy
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)

def optimize_model(policy_net, target_net, optimizer, memory, batch_size, gamma, device):
    """
    Optimize the model by performing a single step of gradient descent.
    
    Args:
        policy_net: Policy network
        target_net: Target network
        optimizer: Optimizer
        memory: Replay buffer
        batch_size: Batch size
        gamma: Discount factor
        device: PyTorch device
        
    Returns:
        Loss
    """
    if len(memory) < batch_size:
        return None
    
    # Sample batch from replay buffer
    states, actions, rewards, next_states, dones = memory.sample(batch_size)
    
    # Convert to PyTorch tensors
    state_batch = torch.cat([create_state_tensor(s, device) for s in states])
    action_batch = torch.tensor([a for a in actions], device=device).unsqueeze(1)
    reward_batch = torch.tensor([r for r in rewards], device=device, dtype=torch.float32)
    next_state_batch = torch.cat([create_state_tensor(s, device) for s in next_states])
    done_batch = torch.tensor([d for d in dones], device=device, dtype=torch.float32)
    
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    # Compute V(s_{t+1}) for all next states
    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values = target_net(next_state_batch).max(1)[0]
    
    # Compute the expected Q values
    expected_state_action_values = reward_batch + gamma * next_state_values * (1 - done_batch)
    
    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # Clip gradients to avoid exploding gradients
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    
    return loss.item()

def get_epsilon(episode, epsilon_start, epsilon_end, epsilon_decay):
    """
    Calculate epsilon for epsilon-greedy policy.
    
    Args:
        episode: Current episode
        epsilon_start: Starting epsilon
        epsilon_end: Minimum epsilon
        epsilon_decay: Number of episodes to decay epsilon over
        
    Returns:
        Current epsilon
    """
    return epsilon_end + (epsilon_start - epsilon_end) * np.exp(-episode / epsilon_decay)

def train(args):
    """
    Train the DQN agent.
    
    Args:
        args: Command-line arguments
    """
    # Set up logging
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs", f"dqn_{run_id}")
    writer = SummaryWriter(log_dir)
    logger.info(f"{Fore.CYAN}Tensorboard logs will be saved to {log_dir}")
    
    # Set up environment
    logger.info(f"{Fore.CYAN}Setting up environment...")
    env = DinoEnv()
    n_actions = env.action_space.n
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info(f"{Fore.GREEN}Using device: {device}")
    
    # Set up networks
    logger.info(f"{Fore.CYAN}Setting up networks...")
    input_shape = (config.FRAME_STACK, *config.FRAME_SIZE)
    policy_net = DQN(input_shape, n_actions).to(device)
    target_net = DQN(input_shape, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    # Set up optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=5e-5)
    
    # Set up replay buffer
    memory = ReplayBuffer(config.REPLAY_MEMORY_SIZE)
    
    # Set up tracking variables
    episode_rewards = []
    best_avg_reward = -float('inf')
    frame_idx = 0
    
    # Start training
    logger.info(f"{Fore.GREEN}Starting training for {config.EPISODES} episodes...")
    for episode in range(1, config.EPISODES + 1):
        # Reset environment
        frame = env.reset()
        
        # Preprocess frame
        preprocessed_frame = preprocess_frame(frame)
        stacked_frames = None
        state = stack_frames(stacked_frames, preprocessed_frame, True, config.FRAME_STACK)
        stacked_frames = deque([preprocessed_frame for _ in range(config.FRAME_STACK)], maxlen=config.FRAME_STACK)
        
        # Initialize episode variables
        episode_reward = 0
        losses = []
        
        # Calculate epsilon
        epsilon = get_epsilon(episode, config.EPSILON_START, config.EPSILON_END, config.EPSILON_DECAY)
        
        # Start episode
        done = False
        step = 0
        
        while not done:
            # Select and perform an action
            state_tensor = create_state_tensor(state, device)
            action = select_action(state_tensor, policy_net, epsilon, n_actions, device)
            
            # Convert action to environment format
            env_action = action.item()
            
            # Take action in environment
            next_frame, reward, done, info = env.step(env_action)
            frame_idx += 1
            
            # Preprocess next frame
            next_preprocessed_frame = preprocess_frame(next_frame)
            next_state = stack_frames(stacked_frames, next_preprocessed_frame, False, config.FRAME_STACK)
            
            # Store transition in replay buffer
            memory.push(state, env_action, reward, next_state, done)
            
            # Only start optimization after buffer has 2000 samples
            if len(memory) < 2000:
                state = next_state
                episode_reward += reward
                if args.save_frames and step % 100 == 0:
                    save_debug_frame(next_preprocessed_frame, step, episode)
                step += 1
                if step > 10000:
                    logger.warning(f"{Fore.YELLOW}Episode {episode} too long, breaking...")
                    break
                continue

            # Move to the next state
            state = next_state
            episode_reward += reward

            # Perform optimization step
            loss = optimize_model(policy_net, target_net, optimizer, memory, config.BATCH_SIZE, config.GAMMA, device)
            if loss is not None:
                losses.append(loss)

            # Debug: save frame occasionally
            if args.save_frames and step % 100 == 0:
                save_debug_frame(next_preprocessed_frame, step, episode)

            # Update target network every 1000 environment steps
            if frame_idx % 1000 == 0:
                target_net.load_state_dict(policy_net.state_dict())

            step += 1

            # Break if episode is too long
            if step > 10000:
                logger.warning(f"{Fore.YELLOW}Episode {episode} too long, breaking...")
                break
        
        # Log episode statistics
        episode_rewards.append(episode_reward)
        avg_loss = np.mean(losses) if losses else 0
        avg_reward = np.mean(episode_rewards[-100:])
        
        # Log to tensorboard
        writer.add_scalar("train/episode_reward", episode_reward, episode)
        writer.add_scalar("train/avg_reward", avg_reward, episode)
        writer.add_scalar("train/epsilon", epsilon, episode)
        if losses:
            writer.add_scalar("train/loss", avg_loss, episode)
        
        # Update progress
        update_progress(episode, config.EPISODES, avg_reward, epsilon, avg_loss)
        
        # Save model if it's the best so far
        if avg_reward > best_avg_reward and episode > 10:
            best_avg_reward = avg_reward
            torch.save({
                'episode': episode,
                'frame_idx': frame_idx,
                'policy_net': policy_net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, config.BEST_MODEL_PATH)
            logger.info(f"{Fore.GREEN}New best model saved with avg reward: {best_avg_reward:.2f}")
        
        # Save model periodically
        if episode % config.SAVE_EVERY == 0:
            torch.save({
                'episode': episode,
                'frame_idx': frame_idx,
                'policy_net': policy_net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, config.MODEL_PATH.replace('.pth', f'_{episode}.pth'))
            logger.info(f"{Fore.CYAN}Model saved at episode {episode}")
        
        # Plot rewards
        if episode % 50 == 0:
            plot_rewards(episode_rewards)
    
    # Save final model
    torch.save({
        'episode': config.EPISODES,
        'frame_idx': frame_idx,
        'policy_net': policy_net.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, config.MODEL_PATH)
    logger.info(f"{Fore.GREEN}Final model saved after {config.EPISODES} episodes")
    
    # Plot final rewards
    plot_rewards(episode_rewards)
    
    # Close environment and tensorboard writer
    env.close()
    writer.close()

def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Train a DQN agent to play Chrome Dino')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disable CUDA training')
    parser.add_argument('--episodes', type=int, default=config.EPISODES,
                        help='number of episodes to train (default: 1000)')
    parser.add_argument('--save-frames', action='store_true', default=False,
                        help='save frames during training for debugging')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # Override config if provided in args
    if args.episodes != config.EPISODES:
        config.EPISODES = args.episodes
        
    try:
        train(args)
    except KeyboardInterrupt:
        logger.info(f"{Fore.YELLOW}Training interrupted by user")
    except Exception as e:
        logger.error(f"{Fore.RED}Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info(f"{Fore.GREEN}Training session ended") 