import time
import numpy as np
import imageio
import os
from stable_baselines3 import PPO
from env.car_parking_obstacle_env import parking2dEnvWithObstacle

# --- Load environment with rendering ---
env = parking2dEnvWithObstacle(grid_size=10, target=(7.0, 3.0), render_mode="human")

# --- Load trained PPO model ---
model = PPO.load("models/ppo_parking_obstacle")

# --- Video storage ---
os.makedirs("results/videos", exist_ok=True)

num_episodes = 2  # record 2 episodes for demo
for ep in range(1, num_episodes + 1):
    obs, _ = env.reset()
    done, total_reward = False, 0
    frames = []  # store episode frames
    fixed_size = None  # will detect once per episode

    print(f"\n🚗 Episode {ep} starting")

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()

        # --- Capture current frame from Matplotlib ---
        env.fig.canvas.draw()
        buf = np.frombuffer(env.fig.canvas.tostring_rgb(), dtype=np.uint8)

        width, height = env.fig.canvas.get_width_height()
        total_pixels = buf.size // 3

        # Detect actual size only once
        if fixed_size is None:
            if width * height != total_pixels:
                height = total_pixels // width
                fixed_size = (width, height)
                print(f"⚠️ DPI mismatch detected, using {fixed_size}")
            else:
                fixed_size = (width, height)
                print(f"Using reported size {fixed_size}")

        # Reshape correctly
        frame = buf.reshape((fixed_size[1], fixed_size[0], 3))
        frames.append(frame)

        time.sleep(0.15)
        done = terminated or truncated
        total_reward += reward

    # --- Save video ---
    video_path = f"results/videos/episode_{ep}.mp4"
    imageio.mimsave(video_path, frames, fps=6)
    print(f"Saved episode {ep} video to {video_path}")
    print(f"Episode {ep} finished | Total Reward: {total_reward}")

env.close()
