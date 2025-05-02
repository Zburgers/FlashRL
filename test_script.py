from dino_env import DinoEnv
import time
import random
import traceback
import sys
import logging
import colorama
from colorama import Fore, Back, Style

# Initialize colorama for cross-platform colored terminal output
colorama.init(autoreset=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("dino-rl")

def print_header(text):
    """Print a styled header"""
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'=' * 60}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{text.center(60)}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{'=' * 60}{Style.RESET_ALL}")

def print_subheader(text):
    """Print a styled subheader"""
    print(f"\n{Fore.YELLOW}{Style.BRIGHT}{text}")
    print(f"{Fore.YELLOW}{Style.BRIGHT}{'-' * len(text)}{Style.RESET_ALL}")

def print_game_stats(step, action, reward, total_reward, score, done, obs):
    """Print game statistics in a clean format"""
    action_str = f"{Fore.GREEN}JUMP{Style.RESET_ALL}" if action == 1 else f"{Fore.BLUE}WAIT{Style.RESET_ALL}"
    status = f"{Fore.RED}CRASHED{Style.RESET_ALL}" if done else f"{Fore.GREEN}ALIVE{Style.RESET_ALL}"
    
    # Create a simple visualization of distance to obstacle
    obstacle_dist = int(obs[1] / 10) if obs[1] < 600 else 0
    dino_height = max(0, min(3, int(obs[0] / 30)))
    
    if obstacle_dist > 0:
        vis = "🦖" + " " * (obstacle_dist - 1) + "🌵"
    else:
        vis = "🦖"
    
    # Format the stats line
    stats = (
        f"Step {Fore.CYAN}{step:4d}{Style.RESET_ALL} | "
        f"Action: {action_str} | "
        f"Score: {Fore.MAGENTA}{score:7.1f}{Style.RESET_ALL} | "
        f"Reward: {Fore.YELLOW}{reward:+6.1f}{Style.RESET_ALL} | "
        f"Total: {Fore.YELLOW}{total_reward:+7.1f}{Style.RESET_ALL} | "
        f"Status: {status}"
    )
    
    # Format the observation line
    obs_line = (
        f"       Dino Y: {Fore.CYAN}{obs[0]:5.1f}{Style.RESET_ALL} | "
        f"Obstacle X: {Fore.CYAN}{obs[1]:5.1f}{Style.RESET_ALL} | "
        f"Width: {Fore.CYAN}{obs[2]:4.1f}{Style.RESET_ALL} | "
        f"{vis}"
    )
    
    print(stats)
    print(obs_line)

# Display welcome message
print_header("CHROME DINO GAME RL ENVIRONMENT")
print(f"{Fore.WHITE}A browser window will open and remain open.")
print(f"{Fore.WHITE}DO NOT close the browser window unless you want to exit the program.")
print(f"{Fore.WHITE}Press {Fore.RED}Ctrl+C{Fore.WHITE} in the terminal to exit properly.")

env = None
try:
    print_subheader("Initializing Environment")
    env = DinoEnv()
    logger.info(f"{Fore.GREEN}Environment initialized successfully!{Style.RESET_ALL}")

    print_subheader("Starting Game")
    obs = env.reset()
    logger.info(f"{Fore.GREEN}Game started! Initial position: Y={obs[0]:.1f}{Style.RESET_ALL}")
    
    print_subheader("Running Random Agent")
    print(f"{Fore.WHITE}Dinosaur will make random jumps. Watch the game in the browser window.")
    print(f"{Fore.WHITE}This will continue until you press Ctrl+C to stop.")
    print()
    
    done = False
    total_reward = 0
    games_played = 0
    max_score = 0
    
    for step in range(1000000):  # Practically infinite loop
        # Random action: 0=do nothing, 1=jump
        action = random.randint(0, 1)
        
        # Execute the action
        obs, reward, done, info = env.step(action)
        total_reward += reward
        score = info.get("score", 0)
        
        # Update max score
        max_score = max(max_score, score)
        
        # Print status every 10 steps or when important events happen
        if step % 10 == 0 or done or action == 1:  # Show on jumps too
            print_game_stats(step, action, reward, total_reward, score, done, obs)
        
        # Handle game over
        if done:
            games_played += 1
            print_subheader(f"Game Over - Session #{games_played}")
            logger.info(f"{Fore.YELLOW}Final score: {score:.1f} | Max height reached: {max_score:.1f}{Style.RESET_ALL}")
            
            # Wait a bit before resetting
            print(f"{Fore.CYAN}Resetting in 3 seconds...{Style.RESET_ALL}")
            time.sleep(3)
            obs = env.reset()
            print(f"{Fore.GREEN}Game reset! Starting again...{Style.RESET_ALL}")
            done = False
            total_reward = 0
    
    print_subheader("Test Completed")

except KeyboardInterrupt:
    print_subheader("Test Interrupted")
    logger.info(f"{Fore.YELLOW}Test interrupted by user (Ctrl+C){Style.RESET_ALL}")
    
except Exception as e:
    print_subheader("Error Occurred")
    logger.error(f"{Fore.RED}Error during test: {e}{Style.RESET_ALL}")
    traceback.print_exc()
    
finally:
    # Make sure we close the environment properly
    print_subheader("Cleaning Up")
    try:
        if env:
            env.close()
            logger.info(f"{Fore.GREEN}Environment closed successfully{Style.RESET_ALL}")
    except Exception as close_error:
        logger.error(f"{Fore.RED}Error closing environment: {close_error}{Style.RESET_ALL}")
    
    print_header("THANK YOU FOR USING DINO RL")
