import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from car_parking_env import parking1dEnv

env = parking1dEnv()

Q = defaultdict(lambda: [0, 0])

alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.99
min_epsilon = 0.05
episodes = 500

episode_rewards = []

episode_rewards = []

for ep in range(episodes):
    state, _ = env.reset()
    state = int(state[0])
    done = False
    total_reward = 0

    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(Q[state]))

        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = int(next_state[0])
        done = terminated or truncated

        best_next_q = max(Q[next_state])
        Q[state][action] += alpha * (reward + gamma * best_next_q - Q[state][action])

        state = next_state
        total_reward += reward

    if epsilon > min_epsilon:
        epsilon *= epsilon_decay

    episode_rewards.append(total_reward)

print("Training finished")

plt.figure(figsize=(10, 5))
plt.plot(episode_rewards, label="Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Q-learning Rewards over Episodes (Parking1DEnv)")
plt.legend()
plt.grid(True)
plt.show()

