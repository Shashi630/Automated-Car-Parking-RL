import os
import gymnasium as gym
from stable_baselines3 import PPO
from env.car_parking_physics_env import CarParkingPhysicsEnv

# Create results folder if not exists
os.makedirs("results/models", exist_ok=True)

# Register / wrap the environment
env = CarParkingPhysicsEnv(grid_size=10, render_mode=None)

# Define PPO model
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
    ent_coef=0.01,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    n_epochs=10,
    tensorboard_log="results/tensorboard/"
)

# Train the agent
print("Training PPO agent on CarParkingPhysicsEnv...")
model.learn(total_timesteps=50_000)

# Save model
save_path = "results/models/ppo_car_parking_physics"
model.save(save_path)
print(f"Model saved at {save_path}")
