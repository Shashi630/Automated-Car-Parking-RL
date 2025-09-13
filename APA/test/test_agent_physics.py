import time
from stable_baselines3 import PPO
from env.car_parking_physics_env import CarParkingPhysicsEnv

# Load trained PPO model
model_path = "results/models/ppo_car_parking_physics"
model = PPO.load(model_path)

# Create environment with rendering
env = CarParkingPhysicsEnv(grid_size=10, render_mode="human")

# Run one test episode
state, _ = env.reset()
done = False
total_reward = 0
print("🚗 Testing PPO agent in CarParkingPhysicsEnv with Obstacles & Multiple Spots")

for step in range(300):  # run max 300 steps
    if done:
        break

    # Agent picks action
    action, _ = model.predict(state)
    state, reward, terminated, truncated, info = env.step(action)
    env.render()

    total_reward += reward
    done = terminated or truncated

    print(f"Step {step}: State={state}, Reward={reward:.2f}")

# Final outcome check
if total_reward >= 80:
    print(f"Correct Parking! Total Reward: {total_reward:.2f}")
elif total_reward <= -40:
    print(f"Collision or Wrong Parking! Total Reward: {total_reward:.2f}")
else:
    print(f"Episode ended (timeout). Total Reward: {total_reward:.2f}")

print("🏁 Test finished")
env.close()
