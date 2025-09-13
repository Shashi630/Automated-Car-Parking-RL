import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env.car_parking_obstacle_env import parking2dEnvWithObstacle  # use continuous env

def make_env():
    return parking2dEnvWithObstacle(grid_size=10, target=(7.0, 3.0), render_mode=None)

env = DummyVecEnv([make_env])  # SB3 wrapper

# --- Set up PPO model with TensorBoard logging ---
log_dir = "./results/tensorboard/"
os.makedirs(log_dir, exist_ok=True)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    tensorboard_log=log_dir  # logs go here
)

# --- Train the model ---
TIMESTEPS = 100_000
model.learn(total_timesteps=TIMESTEPS, tb_log_name="ppo_parking_obstacle")

# --- Save trained model ---
os.makedirs("models", exist_ok=True)
model.save("models/ppo_parking_obstacle")

print("PPO Training finished and model saved")
print(f"TensorBoard logs saved in {log_dir}")
