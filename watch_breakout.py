#!/usr/bin/env python3
import os
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

import torch
import ale_py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage, SubprocVecEnv
import argparse
import time

# Register ALE environments
gym.register_envs(ale_py)

def watch_agent(model_path, num_episodes=5, render_mode="human"):
    """
    Load a trained model and watch it play
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model
    num_episodes : int
        Number of episodes to play
    render_mode : str
        'human' for visual rendering, 'rgb_array' for programmatic access
    """
    # Create a single environment for evaluation
    env = gym.make("BreakoutNoFrameskip-v4", render_mode=render_mode)
    env = gym.wrappers.AtariPreprocessing(
        env, 
        frame_skip=4, 
        screen_size=84,
        grayscale_obs=True,
        scale_obs=False,
        terminal_on_life_loss=False
    )
    env = gym.wrappers.FrameStackObservation(env, 4)
    
    # Load the trained model
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path)
    
    total_reward = 0
    
    # Play episodes
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        step = 0
        
        print(f"Episode {episode+1}/{num_episodes}")
        
        while not done:
            # Get action from the agent
            action, _ = model.predict(obs)
            
            # Execute action in the environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            step += 1
            
            # Sleep a bit to slow down the visual rendering
            if render_mode == "human":
                time.sleep(0.01)  # Adjust as needed for smooth viewing
            
            # Print progress every 100 steps
            if step % 100 == 0:
                print(f"Step {step}, Current Score: {episode_reward}")
        
        total_reward += episode_reward
        print(f"Episode {episode+1} finished. Score: {episode_reward}")
    
    # Calculate average reward
    avg_reward = total_reward / num_episodes
    print(f"Average score over {num_episodes} episodes: {avg_reward:.2f}")
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to play")
    args = parser.parse_args()
    
    watch_agent(args.model, num_episodes=args.episodes)