import gym
import numpy as np
import os
from gym import spaces
from playwright.sync_api import sync_playwright
import time
import atexit
import logging
import colorama
from colorama import Fore, Style

# Initialize colorama
colorama.init(autoreset=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("dino-env")

class DinoEnv(gym.Env):
    def __init__(self):
        super(DinoEnv, self).__init__()
        logger.info(f"{Fore.CYAN}Starting Chrome Dino environment...")
        logger.info(f"{Fore.YELLOW}Note: DO NOT CLOSE BROWSER WINDOW unless you want to exit")
        
        self.playwright = sync_playwright().start()
        
        # Launch browser with persistent window
        self.browser = self.playwright.chromium.launch(
            headless=False,
            args=[
                '--disable-domain-reliability',
                '--disable-component-update',
                '--disable-background-networking',
                '--window-size=800,600',
                '--start-maximized'
            ]
        )
        
        self.context = self.browser.new_context(
            viewport={'width': 800, 'height': 450},
            ignore_https_errors=True
        )
        
        # Register close handler
        atexit.register(self.cleanup)
        
        # Track if browser was manually closed
        self.browser_closed = False
        
        # Track browser status and handle close event
        self.browser.on("disconnected", self._on_browser_closed)
        
        self.page = self.context.new_page()
        
        # Try to load chrome://dino directly
        logger.info(f"{Fore.CYAN}Loading Chrome Dino game...")
        try:
            # Force offline mode to make chrome://dino accessible
            self.page.route('**/*', lambda route: route.abort())
            self.page.goto('chrome://dino')
            logger.info("Page loaded, waiting for initialization...")
            # Longer wait time to ensure the page loads
            time.sleep(3)
            
            # Start the game
            logger.info("Starting game with spacebar...")
            self.page.keyboard.press('Space')
            time.sleep(1)
            
            # Verify game is running
            if self._verify_game():
                logger.info(f"{Fore.GREEN}Successfully loaded chrome://dino")
            
        except Exception as e:
            logger.warning(f"{Fore.YELLOW}Initial load attempt failed: {str(e).split(':')[0]}")
            # Don't close the browser or context, just try another approach
            logger.info(f"{Fore.CYAN}Trying alternative approach...")
            
            try:
                # Directly navigate to chrome://dino with a different approach
                self.page.goto("chrome://dino")
                logger.info("Page loaded (second attempt), waiting...")
                time.sleep(3)
                
                # Start the game
                logger.info("Starting game (second attempt)...")
                self.page.keyboard.press('Space')
                time.sleep(1.5)
                
                # Verify game is running
                if self._verify_game():
                    logger.info(f"{Fore.GREEN}Successfully loaded chrome://dino on second attempt")
            except Exception as e2:
                logger.warning(f"{Fore.YELLOW}Second attempt also failed: {str(e2).split(':')[0]}")
                logger.info(f"{Fore.CYAN}Continuing with best-effort approach...")
        
        # Set up the action and observation spaces
        self.action_space = spaces.Discrete(2)  # 0 = do nothing, 1 = jump
        self.observation_space = spaces.Box(low=0, high=1000, shape=(3,), dtype=np.float32)
        
        logger.info(f"{Fore.GREEN}Environment ready! {Fore.CYAN}Game window will remain open.")
        logger.info(f"{Fore.CYAN}Close the browser window manually to terminate the program.")

    def _on_browser_closed(self):
        """Handle browser closed event"""
        logger.info(f"{Fore.YELLOW}Browser window was closed manually. Terminating...")
        self.browser_closed = True
    
    def cleanup(self):
        """Clean up resources when the program exits"""
        try:
            if hasattr(self, 'context') and self.context and not self.browser_closed:
                self.context.close()
            if hasattr(self, 'browser') and self.browser and not self.browser_closed:
                self.browser.close()
            if hasattr(self, 'playwright') and self.playwright:
                self.playwright.stop()
            logger.info(f"{Fore.GREEN}Resources cleaned up")
        except Exception as e:
            logger.error(f"{Fore.RED}Error during cleanup: {e}")
    
    def _verify_game(self):
        """Verify the game is running correctly"""
        try:
            # Check if we can access the Runner object
            is_runner_ready = self.page.evaluate('''
            () => {
                try {
                    // Check if Runner exists
                    if (typeof Runner === 'undefined' || !Runner.instance_) {
                        console.log('Runner not found');
                        return false;
                    }
                    
                    // Check if the game is ready
                    if (!Runner.instance_.tRex) {
                        console.log('T-Rex not initialized');
                        return false;
                    }
                    
                    console.log('Game ready:', {
                        playing: Runner.instance_.playing,
                        crashed: Runner.instance_.crashed
                    });
                    
                    return true;
                } catch (e) {
                    console.error('Error checking game:', e);
                    return false;
                }
            }
            ''')
            
            if not is_runner_ready:
                logger.warning(f"{Fore.YELLOW}Game not properly initialized, but continuing anyway")
                return False
            
            logger.info(f"{Fore.GREEN}Game verified and ready")
            return True
        except Exception as e:
            logger.warning(f"{Fore.YELLOW}Error verifying game: {str(e).split(':')[0]}")
            return False

    def get_state(self):
        if self.browser_closed:
            logger.warning(f"{Fore.YELLOW}Browser was closed. Cannot get state.")
            return {"y": 93, "x": 600, "width": 0, "crashed": True, "score": 0}
            
        try:
            return self.page.evaluate('''
            () => {
                try {
                    // Safety check
                    if (typeof Runner === 'undefined' || !Runner.instance_) {
                        return { y: 93, x: 600, width: 0, crashed: false, score: 0 };
                    }
                    
                    const runner = Runner.instance_;
                    const tRex = runner.tRex;
                    const horizon = runner.horizon;
                    
                    // Get the first obstacle if any
                    const obstacle = horizon.obstacles.length > 0 ? horizon.obstacles[0] : null;
                    
                    return {
                        y: tRex.yPos,
                        x: obstacle ? obstacle.xPos : 600,
                        width: obstacle ? obstacle.width : 0,
                        crashed: runner.crashed || false,
                        score: runner.distanceRan || 0
                    };
                } catch (e) {
                    console.error('Error getting state:', e);
                    return { y: 93, x: 600, width: 0, crashed: false, score: 0 };
                }
            }
            ''')
        except Exception as e:
            # This is expected during resets, so make it a debug message
            if "net::ERR_INTERNET_DISCONNECTED" in str(e):
                pass  # This is normal and expected
            else:
                logger.debug(f"Error getting state: {str(e).split(':')[0]}")
            # Return a default state that won't immediately end the game
            return { "y": 93, "x": 600, "width": 0, "crashed": False, "score": 0 }

    def step(self, action):
        if self.browser_closed:
            logger.warning(f"{Fore.YELLOW}Browser was closed. Cannot perform step.")
            return np.array([0, 0, 0], dtype=np.float32), -100, True, {"score": 0}
            
        if action == 1:
            try:
                # Press spacebar to jump
                self.page.keyboard.press("Space")
            except Exception as e:
                logger.debug(f"Error performing jump action: {str(e).split(':')[0]}")
                
        # Short delay to let the game update
        time.sleep(0.05)
        
        # Get the current state
        state = self.get_state()
        
        # Convert to the observation format expected by gym
        obs = np.array([state["y"], state["x"], state["width"]], dtype=np.float32)
        
        # Calculate reward
        reward = 1.0  # Small reward for surviving
        done = state["crashed"]
        if done:
            reward = -100.0  # Penalty for crashing
            
        # Additional info dictionary with score
        info = {"score": state.get("score", 0)}
        
        return obs, reward, done, info

    def reset(self):
        if self.browser_closed:
            logger.warning(f"{Fore.YELLOW}Browser was closed. Cannot reset.")
            return np.array([0, 0, 0], dtype=np.float32)
            
        logger.info(f"{Fore.CYAN}Resetting environment...")
        try:
            # Reload the page to restart
            self.page.reload()
            time.sleep(2)
            
            # Start the game with a spacebar press
            self.page.keyboard.press('Space')
            time.sleep(1)
            
            # Get initial state
            state = self.get_state()
            logger.info(f"{Fore.GREEN}Environment reset complete")
            return np.array([state["y"], state["x"], state["width"]], dtype=np.float32)
            
        except Exception as e:
            # This error is expected and normal for chrome://dino
            if "net::ERR_INTERNET_DISCONNECTED" in str(e):
                logger.debug("Expected network disconnection during reset (this is normal)")
            else:
                logger.warning(f"{Fore.YELLOW}Error during reset: {str(e).split(':')[0]}")
            
            # Try to restart the game with a space press anyway
            try:
                self.page.keyboard.press('Space')
                time.sleep(0.5)
            except:
                pass
                
            # Return a default observation if reset fails
            return np.array([93, 600, 0], dtype=np.float32)

    def close(self):
        # Only close resources if the browser wasn't manually closed
        if not self.browser_closed:
            self.cleanup()
        logger.info(f"{Fore.GREEN}Environment closed") 