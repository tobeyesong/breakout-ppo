# Breakout PPO Reinforcement Learning

This repository contains a set of Python scripts for training and evaluating a reinforcement learning agent to play the Atari game Breakout using Proximal Policy Optimization (PPO).

## Overview

The project consists of three main components:

1. **Training script**: Train a PPO agent to play Breakout
2. **Visual evaluation**: Watch the trained agent play in real-time 
3. **Headless evaluation**: Assess agent performance without visual rendering

## Requirements

- Python 3.11+
- PyTorch
- Stable Baselines 3
- Gymnasium
- ALE-Py (Arcade Learning Environment)
- NumPy
- tqdm

## Installation

  ```bash
   git clone https://github.com/tobeyesong/ppo-breakout.git
   cd ppo-breakout
   ```

```bash
# Create and activate a virtual environment
python -3.11 -m venv 3.11env

# Windows
3.11env\Scripts\activate
# macOS/Linux
source 3.11env/bin/activate

# Install dependencies
pip install torch stable-baselines3 gymnasium[atari] ale-py numpy tqdm

# Install Atari ROMs
python -m ale_py.import_roms
```

## Usage

### Training an Agent

Train a new agent from scratch:

```bash
python train_breakout.py --steps 1000000 --envs 8
```

Resume training from an existing model:

```bash
python train_breakout.py --model models/ppo_breakout_final.zip --steps 500000
```

Parameters:
- `--model`: Path to a model to resume training (optional)
- `--steps`: Number of timesteps to train (default: 10,000,000)
- `--envs`: Number of parallel environments for training (default: 8)

Training progress is displayed with a progress bar and logged via TensorBoard.

### Watching the Agent Play

Visualize how your trained agent performs:

```bash
python watch_breakout.py --model models/ppo_breakout_final.zip --episodes 5
```

Parameters:
- `--model`: Path to the trained model (required)
- `--episodes`: Number of episodes to play (default: 5)

### Evaluating Performance (Headless)

Run fast, headless evaluation to get performance metrics:

```bash
python headless_evaluation.py --model models/ppo_breakout_final.zip --episodes 100 --verbose
```

Parameters:
- `--model`: Path to the trained model (required)
- `--episodes`: Number of episodes to evaluate (default: 100)
- `--verbose`: Print detailed per-episode information

## TensorBoard Visualization

View training metrics with TensorBoard:

```bash
tensorboard --logdir=logs
```

Then open http://localhost:6006 in your browser.

## Project Structure

- `train_breakout.py`: Main training script with PPO implementation
- `watch_breakout.py`: Script to watch the agent play with visual rendering
- `headless_evaluation.py`: Fast performance evaluation without rendering
- `models/`: Directory where trained models are saved
- `logs/`: Directory for TensorBoard logs

## Model Hyperparameters

The PPO agent uses the following hyperparameters:
- Learning rate: 2.5e-4
- Steps per environment before update: 128
- Batch size: 256
- Epochs per update: 4
- Discount factor (gamma): 0.99
- GAE lambda: 0.95
- Clip range: 0.1
- Entropy coefficient: 0.01

## License

[MIT License]

## Acknowledgements

This project uses the following libraries:
- [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3)
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)
- [ALE-Py](https://github.com/Farama-Foundation/Arcade-Learning-Environment)