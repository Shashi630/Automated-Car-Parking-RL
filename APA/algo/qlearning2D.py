import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle
from env.car_parking2D_evn import parking2dEnv

env = parking2dEnv(grid_size=10, target=(7, 3))

Q = defaultdict(lambda: [0, 0, 0, 0])

alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.05
episodes = 1000

episode_rewards = []

for ep in range(episodes):
    state, _ = env.reset()
    state = tuple(state)  
    done = False
    total_reward = 0

    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(Q[state]))

        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = tuple(next_state) 
        done = terminated or truncated

        best_next_q = max(Q[next_state])
        Q[state][action] += alpha * (reward + gamma * best_next_q - Q[state][action])

        state = next_state
        total_reward += reward

    if epsilon > min_epsilon:
        epsilon *= epsilon_decay

    episode_rewards.append(total_reward)

print("Training finished")

with open("q_table.pkl", "wb") as f:
    pickle.dump(dict(Q), f) 
print("Q-table saved as q_table.pkl")


plt.figure(figsize=(10, 5))
plt.plot(episode_rewards, label="Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Q-learning Rewards over Episodes (Parking2DEnv)")
plt.legend()
plt.grid(True)
plt.show()
