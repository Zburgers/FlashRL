# FlashRL: Chrome Dino Game RL Agent

This project implements a Deep Q-Network (DQN) agent to play the Chrome Dino game using reinforcement learning. The agent learns to play the game by observing screenshots of the game and deciding when to jump to avoid obstacles.

## Project Structure

```
FlashRL/
├── chromedriver-win64/    # Chrome WebDriver files
├── assets/                # Game assets
│   ├── game.js            # Dino game JavaScript code
│   └── ...
├── data/                  # Data directory
│   ├── models/            # Saved model checkpoints
│   ├── debug_frames/      # Debug frames saved during training
│   └── videos/            # Optional gameplay recordings
├── logs/                  # TensorBoard logs
├── dino_env.py            # Gym environment for the Dino game
├── utils.py               # Utility functions for preprocessing, etc.
├── config.py              # Configuration parameters
├── dqn_train.py           # DQN training script
├── dqn_eval.py            # DQN evaluation script
├── test_script.py         # Simple test script for the environment
└── requirements.txt       # Dependencies
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/FlashRL.git
   cd FlashRL
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install Playwright browsers:
   ```bash
   playwright install
   ```

## Usage

### Testing the Environment

Before training, you can test the environment to ensure it's working correctly:

```bash
python test_script.py
```

This will open a browser window with the Chrome Dino game and run a random agent to make sure the environment is functional.

### Training the DQN Agent

To train the DQN agent:

```bash
python dqn_train.py
```

Optional arguments:
- `--episodes`: Number of episodes to train (default: 1000)
- `--no-cuda`: Disable CUDA training
- `--save-frames`: Save frames during training for debugging

The training script will:
1. Initialize the DQN model and environment
2. Train the agent using experience replay and epsilon-greedy exploration
3. Save model checkpoints to `data/models/`
4. Log training metrics to TensorBoard

You can monitor training progress with TensorBoard:

```bash
tensorboard --logdir=logs
```

### Evaluating the Trained Agent

To evaluate a trained model:

```bash
python dqn_eval.py
```

Optional arguments:
- `--model`: Path to model file (default: best model)
- `--episodes`: Number of episodes to evaluate (default: 5)
- `--no-render`: Disable rendering
- `--save-frames`: Save frames during evaluation

## Model Architecture

The DQN model consists of:
- Convolutional layers to process images
- Fully connected layers for Q-value prediction

Input is a stack of 4 preprocessed frames (grayscale, 84x84) to capture temporal information.

## Configuration

You can customize the training parameters in `config.py`, including:
- Frame stack size
- Batch size
- Learning rate
- Epsilon decay
- Replay buffer size
- Target network update frequency
- Model saving frequency

## Troubleshooting

### Browser Issues
- If the browser crashes or freezes, try restarting the script.
- Make sure you have sufficient memory for the browser and DQN model.

### Training Issues
- If training is too slow, consider reducing the frame size or using a more powerful GPU.
- If the model doesn't learn well, try adjusting the reward function or hyperparameters in `config.py`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Chrome Dino game code is based on the original game developed by Google
- DQN implementation inspired by the paper "Human-level control through deep reinforcement learning" by Mnih et al. 