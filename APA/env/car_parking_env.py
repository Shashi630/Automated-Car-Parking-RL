import gymnasium as gym
from gymnasium import spaces
import numpy as np


class parking1dEnv(gym.Env):
    def __init__(self, target = 5, max_steps = 20):
        super().__init__()
        self.action_space = spaces.Discrete(2) # 0 - backward, 1 - forward
        self.observation_space = spaces.Box(low = 0, high = 10, shape = (1,), dtype = np.int32)
        self.target = target
        self.max_steps = max_steps
        self.current_pos = 0
        self.steps = 0

    def reset(self, seed = None, options = None):
        super().reset(seed = seed)
        self.current_pos = np.random.randint(0, 10)
        self.steps = 0
        return np.array([self.current_pos], dtype = np.int32), {}
    
    def step(self, action):
        self.steps += 1
        if action == 0 and self.current_pos > 0:
            self.current_pos -= 1
        elif action == 0 and self.current_pos < 10:
            self.current_pos += 1
        

        reward = -1
        terminated = self.current_pos == self.target
        if terminated:
            reward = 100
        truncated = self.steps >= self.max_steps

        return np.array([self.current_pos], dtype=np.int32), reward, terminated, truncated, {}

    def render(self):
        print(f"Car at position {self.current_pos}, Target at {self.target}")



