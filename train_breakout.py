#!/usr/bin/env python3
"""
train_breakout.py

Train a PPO agent on Breakout using Gymnasium (ALE/Breakout-v5) and Stable-Baselines3.
"""

import os
import ale_py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.wrappers import RecordVideo, AtariPreprocessing, FrameStackObservation

# Register ALE environments
gym.register_envs(ale_py)

# Directories
LOG_DIR    = "logs"
MODEL_DIR  = "models"
VIDEO_DIR  = "videos"
for d in (LOG_DIR, MODEL_DIR, VIDEO_DIR):
    os.makedirs(d, exist_ok=True)

def train():
    env_id = "ALE/Breakout-v5"
    # create vectorized training env
    train_env = make_atari_env(env_id, n_envs=8, seed=42)
    train_env = VecFrameStack(train_env, n_stack=4)
    # create vectorized evaluation env
    eval_env = make_atari_env(env_id, n_envs=1, seed=43)
    eval_env = VecFrameStack(eval_env, n_stack=4)

    # evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
    )

    # PPO model
    model = PPO(
        policy="CnnPolicy",
        env=train_env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=2.5e-4,
        n_steps=128,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )

    # quick smoke test
    model.learn(total_timesteps=100_000, callback=eval_callback)
    # full training
    model.learn(total_timesteps=10_000_000, callback=eval_callback)

    model.save(os.path.join(MODEL_DIR, "ppo_breakout_final"))
    print("✅ Training complete; model saved to models/ppo_breakout_final.zip")

def record_and_evaluate(model, n_episodes=5):
    # build evaluation env with video recording
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    env = AtariPreprocessing(env, frame_skip=4, screen_size=84, grayscale_obs=True)
    env = FrameStackObservation(env, stack_size=4)
    env = RecordVideo(env, video_folder=VIDEO_DIR, episode_trigger=lambda _: True)

    total_reward = 0
    for i in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        print(f"Episode {i+1} reward: {episode_reward}")
        total_reward += episode_reward

    env.close()
    print(f"🎬 Videos saved in '{VIDEO_DIR}'")
    print(f"🔢 Mean reward over {n_episodes} episodes: {total_reward / n_episodes:.2f}")

if __name__ == "__main__":
    train()
    final_model = PPO.load(os.path.join(MODEL_DIR, "ppo_breakout_final"))
    record_and_evaluate(final_model)
