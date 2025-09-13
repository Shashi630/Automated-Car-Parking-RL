from env.car_parking_obstacle_env import parking2dEnvWithObstacle
import pickle, time, numpy as np

env = parking2dEnvWithObstacle(grid_size=10, target=(7, 3), render_mode="human")

with open("q_table_obstacles.pkl", "rb") as f:
    Q = pickle.load(f)
print("📂 Loaded trained Q-table")

def choose_action(state):
    return int(np.argmax(Q.get(tuple(state), [0,0,0,0])))

num_episodes, success = 5, 0

for ep in range(1, num_episodes + 1):
    state, _ = env.reset()
    done, total_reward = False, 0
    print(f"\n🚗 Episode {ep} starting at {state}")

    while not done:
        action = choose_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        env.render()
        time.sleep(0.2) 
        total_reward += reward
        done = terminated or truncated

    if reward == 100: 
        success += 1
    print(f"Episode {ep} finished | Total Reward: {total_reward}")

print(f"\nSuccess Rate: {success}/{num_episodes} episodes")
env.close()
