from env.car_parking2D_evn import parking2dEnv
import numpy as np
import pickle
import time

env = parking2dEnv(grid_size=10, target=(7, 3), render_mode="human")


with open("q_table.pkl", "rb") as f:
    Q = pickle.load(f)
print("📂 Loaded trained Q-table")

def choose_action(state):
    state = tuple(state)
    if state in Q:
        return int(np.argmax(Q[state]))
    else:
        return env.action_space.sample() 
    
num_episodes = 5

for ep in range(1, num_episodes + 1):
    state, _ = env.reset()
    done = False
    total_reward = 0

    print("\nStarting Test Episode")
    print(f"Car starts at {state}, Target at ({env.target_x}, {env.target_y})")

    while not done:
        action = choose_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        env.render()
        time.sleep(0.3) 

        state = next_state
        done = terminated or truncated

        # print(f"Action: {action} | Moved to {next_state} | Reward: {reward}")
        # state = next_state
        # done = terminated or truncated

    print(f"Episode finished | Total Reward: {total_reward}")

env.close()




# from car_parking_env import parking1dEnv
# import numpy as np

# env = parking1dEnv()

# Q = {i: [0, 0] for i in range(11)}  

# state, _ = env.reset()
# done = False
# while not done:
#     action = int(np.argmax(Q[int(state[0])]))
#     state, reward, terminated, truncated, _ = env.step(action)
#     env.render()
#     done = terminated or truncated