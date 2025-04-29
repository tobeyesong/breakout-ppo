# Breakout PPO Reinforcement Learning

A PyTorch implementation of Proximal Policy Optimization (PPO) for playing Atari Breakout.

## Overview

This project implements a Reinforcement Learning agent using the PPO algorithm to play Atari Breakout. The implementation uses Stable Baselines3, Gymnasium, and the Arcade Learning Environment (ALE).

## Features

- Uses PPO algorithm with CNN policy for Atari games
- Supports parallel environments for faster training
- Includes progress tracking during training
- GPU acceleration when available
- Resumable training from saved models

## Dependencies

- Python 3.7+
- PyTorch
- stable-baselines3
- ale-py
- gymnasium
- gymnasium[atari]

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/ppo-breakout.git
   cd ppo-breakout
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install torch gymnasium[atari] ale-py stable-baselines3
   ```

4. Install Atari ROMs for Gymnasium:
   ```bash
   ale-import-roms
   ```
   Note: You need to provide your own Atari ROM files due to copyright restrictions.

## Usage

### Training a new model

To train a new model with default settings:

```bash
python train.py
```

### Customizing training parameters

You can customize various training parameters:

```bash
python train.py --steps 5000000 --envs 16
```

Parameters:
- `--steps`: Number of timesteps to train (default: 2,000,000)
- `--envs`: Number of parallel environments (default: 8)

### Resuming training from a saved model

To resume training from a previously saved model:

```bash
python train.py --model models/your_saved_model.zip --steps 3000000
```

## Model Architecture

- **Policy**: CNN Policy optimized for Atari games
- **Frame processing**: Frames are stacked (4 frames) and properly transposed for CNN input
- **Hyperparameters**: 
  - Learning rate: 2.5e-4
  - Batch size: 256
  - N steps: 128
  - N epochs: 4
  - Gamma: 0.99
  - GAE Lambda: 0.95
  - Clip range: 0.1
  - Entropy coefficient: 0.01

## Troubleshooting

### Common Issues

1. **CUDA out of memory errors**
   - Reduce the number of parallel environments with `--envs`
   - Try running on CPU if GPU memory is insufficient

2. **ROM loading issues**
   - Ensure ROMs are properly imported with `ale-import-roms`
   - Check ROM paths and permissions

3. **Training instability**
   - Try modifying hyperparameters like learning rate or batch size
   - Ensure you have enough training steps (Atari games typically need millions of steps)

### Performance Tips

- Use GPU acceleration when available
- Increase parallel environments for faster training, but be mindful of memory usage
- For best results, train for at least 10 million timesteps

## License

[MIT License](LICENSE)

## Acknowledgements

- This implementation is based on the Stable Baselines3 framework
- PPO algorithm as described in the paper "Proximal Policy Optimization Algorithms" by Schulman et al.