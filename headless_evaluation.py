#!/usr/bin/env python3
import os
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

import torch
import ale_py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
import argparse
import numpy as np
from tqdm import tqdm

# Register ALE environments
gym.register_envs(ale_py)

def evaluate_agent(model_path, num_episodes=100, verbose=False):
    """
    Evaluate a trained model without rendering
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model
    num_episodes : int
        Number of episodes to evaluate
    verbose : bool
        Whether to print detailed progress
    """
    # Create environment with no rendering
    env = gym.make("BreakoutNoFrameskip-v4", render_mode=None)
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
    
    # Track scores and episode lengths
    scores = []
    steps_list = []
    
    # Set up progress bar (always show progress)
    episode_iter = tqdm(range(num_episodes), desc="Evaluating episodes")
    
    # Run evaluation episodes
    for episode in episode_iter:
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            # Get action from the agent
            action, _ = model.predict(obs, deterministic=True)
            
            # Execute action in environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            steps += 1
        
        scores.append(episode_reward)
        steps_list.append(steps)
        
        if verbose:
            print(f"Episode {episode+1}: Score = {episode_reward}, Steps = {steps}")
    
    # Calculate statistics
    avg_score = np.mean(scores)
    std_score = np.std(scores)
    max_score = np.max(scores)
    min_score = np.min(scores)
    median_score = np.median(scores)
    avg_steps = np.mean(steps_list)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Episodes: {num_episodes}")
    print(f"Average Score: {avg_score:.2f} ± {std_score:.2f}")
    print(f"Median Score: {median_score:.2f}")
    print(f"Min/Max Score: {min_score:.2f}/{max_score:.2f}")
    print(f"Average Steps per Episode: {avg_steps:.2f}")
    
    # Close environment
    env.close()
    
    return {
        'avg_score': avg_score,
        'std_score': std_score,
        'median_score': median_score,
        'min_score': min_score,
        'max_score': max_score,
        'avg_steps': avg_steps,
        'scores': scores
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to evaluate")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress")
    args = parser.parse_args()
    
    evaluate_agent(args.model, num_episodes=args.episodes, verbose=args.verbose)