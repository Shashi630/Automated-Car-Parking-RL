import numpy as np, random, pickle, matplotlib.pyplot as plt
from collections import defaultdict
from env.car_parking_obstacle_env import parking2dEnvWithObstacle

# --- Initialize environment ---
env = parking2dEnvWithObstacle(grid_size=10, target=(7, 3))

# --- Initialize Q-table ---
# Each state is a tuple (x, y) rounded to 1 decimal
# Each action has a value, here assuming 4 discrete actions
Q = defaultdict(lambda: [0, 0, 0, 0])

# --- Hyperparameters ---
alpha = 0.1        # learning rate
gamma = 0.9        # discount factor
epsilon = 1.0      # exploration rate
epsilon_decay = 0.995
min_epsilon = 0.05
episodes = 1000

episode_rewards = []

# --- Training loop ---
for ep in range(episodes):
    state, _ = env.reset()
    state = tuple(np.round(state, 1))  # discretize continuous state
    total_reward, done = 0, False

    while not done:
        # --- Epsilon-greedy action selection ---
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(Q[state]))

        # --- Take action ---
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = tuple(np.round(next_state, 1))  # discretize
        done = terminated or truncated

        # --- Update Q-table ---
        best_next = max(Q[next_state]) if next_state in Q else 0
        Q[state][action] += alpha * (reward + gamma * best_next - Q[state][action])

        state, total_reward = next_state, total_reward + reward

    # --- Decay epsilon ---
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    episode_rewards.append(total_reward)

    if ep % 100 == 0:
        print(f"Episode {ep}, Reward: {total_reward}, Epsilon: {epsilon:.3f}")

print("Training finished")

# --- Save Q-table ---
with open("q_table_obstacles.pkl", "wb") as f:
    pickle.dump(dict(Q), f)

# --- Plot learning curve ---
plt.plot(episode_rewards)
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.title("Q-learning with Obstacles")
plt.grid()
plt.show()
