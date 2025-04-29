#!/usr/bin/env python3
import os
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

import torch
import ale_py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import argparse

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Register ALE and make sure folders exist
gym.register_envs(ale_py)
for d in ("logs", "models"):
    os.makedirs(d, exist_ok=True)

# Simple progress callback
class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, log_freq=10000):
        super().__init__()
        self.total = total_timesteps
        self.log_freq = log_freq
        
    def _on_step(self):
        if self.n_calls % self.log_freq == 0:
            print(f"Progress: {self.num_timesteps}/{self.total} steps")
        return True

def train(model_path=None, total_timesteps=10_000_000, n_envs=8):
    """Train a new model or resume training"""
    
    # Create training environment
    train_env = make_atari_env("BreakoutNoFrameskip-v4", n_envs=n_envs, seed=42, 
                               vec_env_cls=SubprocVecEnv)
    train_env = VecFrameStack(train_env, n_stack=4)
    train_env = VecTransposeImage(train_env)
    
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model = PPO.load(model_path, env=train_env)
        resume = True
    else:
        print("Starting new training")
        model = PPO(
            "CnnPolicy",
            train_env,
            learning_rate=2.5e-4,
            n_steps=128,
            batch_size=256,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log="./logs/",
            device=device,
        )
        resume = False
    
    # Setup callback
    progress_callback = ProgressCallback(total_timesteps=total_timesteps)
    
    # Train model
    print(f"\n{'Resuming' if resume else 'Starting'} training for {total_timesteps} steps")
    model.learn(
        total_timesteps=total_timesteps,
        callback=progress_callback,
        reset_num_timesteps=not resume
    )
    
    # Save final model
    final_path = "models/ppo_breakout_final"
    model.save(final_path)
    print(f"Saved final model to {final_path}")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Path to model to resume from (optional)")
    parser.add_argument("--steps", type=int, default=2_000_000, help="Number of steps to train")
    parser.add_argument("--envs", type=int, default=8, help="Number of parallel environments")
    
    args = parser.parse_args()
    
    # Train or resume
    train(args.model, total_timesteps=args.steps, n_envs=args.envs)