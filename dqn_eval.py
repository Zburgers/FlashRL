"""
Deep Q-Network evaluation script for Chrome Dino game.
This script evaluates a trained DQN agent playing the Chrome Dino game.
"""
import os
import time
import numpy as np
import torch
import argparse
import logging
import colorama
from colorama import Fore, Style
from collections import deque

# Import local modules
from dino_env import DinoEnv
from utils import preprocess_frame, stack_frames, create_state_tensor, save_debug_frame
import config
from dqn_train import DQN

# Initialize colorama
colorama.init(autoreset=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("dqn-eval")

def print_header(msg):
    """Print a formatted header message."""
    logger.info(f"\n{Fore.CYAN}{Style.BRIGHT}" + "=" * 60)
    logger.info(f"{Fore.CYAN}{Style.BRIGHT}{msg:^60}")
    logger.info(f"{Fore.CYAN}{Style.BRIGHT}" + "=" * 60)

def print_step_info(step, action, reward, total_reward, info):
    """Print step information."""
    action_name = f"{Fore.GREEN}JUMP" if action == 1 else f"{Fore.BLUE}WAIT"
    logger.info(f"Step {step:4d} | Action: {action_name} | " +
                f"Reward: {Fore.YELLOW}{reward:+6.1f} | " +
                f"Total: {Fore.YELLOW}{total_reward:+7.1f} | " +
                f"Score: {Fore.MAGENTA}{info.get('score', 0):7.1f}")

def evaluate(model_path, episodes=config.EVAL_EPISODES, render=config.RENDER_EVAL, save_frames=False):
    """
    Evaluate a trained DQN agent.
    
    Args:
        model_path: Path to model checkpoint
        episodes: Number of episodes to evaluate
        render: Whether to render frames
        save_frames: Whether to save frames for visualization
    """
    print_header("EVALUATING DQN AGENT FOR CHROME DINO GAME")
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"{Fore.GREEN}Using device: {device}")
    
    # Set up environment
    logger.info(f"{Fore.CYAN}Setting up environment...")
    env = DinoEnv()
    n_actions = env.action_space.n
    
    # Load model
    logger.info(f"{Fore.CYAN}Loading model from {model_path}...")
    
    # Set up networks
    input_shape = (config.FRAME_STACK, *config.FRAME_SIZE)
    model = DQN(input_shape, n_actions).to(device)
    
    # Load model weights
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['policy_net'])
        trained_episodes = checkpoint.get('episode', 0)
        logger.info(f"{Fore.GREEN}Model loaded successfully (trained for {trained_episodes} episodes)")
    else:
        logger.error(f"{Fore.RED}Model file not found: {model_path}")
        return
    
    model.eval()  # Set model to evaluation mode
    
    # Evaluation stats
    episode_rewards = []
    episode_scores = []
    episode_steps = []
    
    # Evaluate for specified number of episodes
    for episode in range(1, episodes + 1):
        logger.info(f"\n{Fore.YELLOW}Episode {episode}/{episodes}")
        
        # Reset environment
        frame = env.reset()
        
        # Preprocess frame
        preprocessed_frame = preprocess_frame(frame)
        stacked_frames = None
        state = stack_frames(stacked_frames, preprocessed_frame, True, config.FRAME_STACK)
        stacked_frames = deque([preprocessed_frame for _ in range(config.FRAME_STACK)], maxlen=config.FRAME_STACK)
        
        # Initialize episode variables
        episode_reward = 0
        done = False
        step = 0
        
        # Start episode
        while not done:
            # Select action (no randomness in evaluation)
            state_tensor = create_state_tensor(state, device)
            with torch.no_grad():
                action = model(state_tensor).max(1)[1].view(1, 1).item()
            
            # Take action in environment
            next_frame, reward, done, info = env.step(action)
            
            # Print step information
            episode_reward += reward
            print_step_info(step, action, reward, episode_reward, info)
            
            # If enabled, save frame for visualization
            if save_frames and step % 10 == 0:
                save_debug_frame(next_frame, step, episode, dir_path="data/eval_frames")
            
            # Preprocess next frame
            next_preprocessed_frame = preprocess_frame(next_frame)
            next_state = stack_frames(stacked_frames, next_preprocessed_frame, False, config.FRAME_STACK)
            
            # Update state
            state = next_state
            step += 1
            
            # Add short delay for visualization
            if render:
                time.sleep(0.01)
            
            # Break if episode is too long
            if step > 10000:
                logger.warning(f"{Fore.YELLOW}Episode too long, breaking...")
                break
        
        # Log episode statistics
        score = info.get('score', 0)
        episode_rewards.append(episode_reward)
        episode_scores.append(score)
        episode_steps.append(step)
        
        logger.info(f"\n{Fore.GREEN}Episode {episode} finished:")
        logger.info(f"{Fore.GREEN}  Steps: {step}")
        logger.info(f"{Fore.GREEN}  Reward: {episode_reward:.2f}")
        logger.info(f"{Fore.GREEN}  Score: {score:.2f}")
    
    # Calculate and print evaluation statistics
    avg_reward = np.mean(episode_rewards)
    avg_score = np.mean(episode_scores)
    avg_steps = np.mean(episode_steps)
    
    print_header("EVALUATION RESULTS")
    logger.info(f"{Fore.GREEN}Average reward: {avg_reward:.2f}")
    logger.info(f"{Fore.GREEN}Average score: {avg_score:.2f}")
    logger.info(f"{Fore.GREEN}Average steps: {avg_steps:.2f}")
    logger.info(f"{Fore.GREEN}Best score: {max(episode_scores):.2f}")
    
    # Close environment
    env.close()
    
    return {
        'avg_reward': avg_reward,
        'avg_score': avg_score,
        'avg_steps': avg_steps,
        'best_score': max(episode_scores)
    }

def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Evaluate a DQN agent playing Chrome Dino')
    parser.add_argument('--model', type=str, default=config.BEST_MODEL_PATH,
                        help=f'path to model file (default: {config.BEST_MODEL_PATH})')
    parser.add_argument('--episodes', type=int, default=config.EVAL_EPISODES,
                        help=f'number of episodes to evaluate (default: {config.EVAL_EPISODES})')
    parser.add_argument('--no-render', action='store_false', dest='render',
                        help='disable rendering')
    parser.add_argument('--save-frames', action='store_true', default=False,
                        help='save frames during evaluation for visualization')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    try:
        evaluate(args.model, args.episodes, args.render, args.save_frames)
    except KeyboardInterrupt:
        logger.info(f"{Fore.YELLOW}Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"{Fore.RED}Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info(f"{Fore.GREEN}Evaluation session ended") 