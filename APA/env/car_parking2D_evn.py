import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class parking2dEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, grid_size=10, target=(5, 5), max_steps=50, render_mode=None):
        super().__init__()

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(
            low=0,
            high=grid_size,
            shape=(2,),
            dtype=np.int32
        )
        
        self.grid_size = grid_size
        self.target_x, self.target_y = target
        self.max_steps = max_steps
        
        self.x = 0
        self.y = 0
        self.steps = 0
        self.render_mode = render_mode

        if render_mode == "human":
            self.fig, self.ax = plt.subplots()
            self.car_marker, = self.ax.plot([], [], "ro", markersize=12, label="Car")
            self.target_marker, = self.ax.plot([], [], "gs", markersize=12, label="Target")
            self.ax.set_xlim(-1, grid_size + 1)
            self.ax.set_ylim(-1, grid_size + 1)
            self.ax.set_xticks(range(grid_size + 1))
            self.ax.set_yticks(range(grid_size + 1))
            self.ax.grid(True)
            self.ax.legend()
            plt.ion()
            plt.show()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.x = np.random.randint(0, self.grid_size + 1)
        self.y = np.random.randint(0, self.grid_size + 1)
        self.steps = 0
        return np.array([self.x, self.y], dtype=np.int32), {}

    def step(self, action):
        self.steps += 1
        
        if action == 0 and self.y < self.grid_size:   # up
            self.y += 1
        elif action == 1 and self.y > 0:             # down
            self.y -= 1
        elif action == 2 and self.x > 0:             # left
            self.x -= 1
        elif action == 3 and self.x < self.grid_size: # right
            self.x += 1

        reward = -1
        terminated = False

        if self.x == self.target_x and self.y == self.target_y:
            reward = 100
            terminated = True

        truncated = self.steps >= self.max_steps

        return np.array([self.x, self.y], dtype=np.int32), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human":
            self.car_marker.set_data(self.x, self.y)
            self.target_marker.set_data(self.target_x, self.target_y)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def close(self):
        if self.render_mode == "human":
            plt.ioff()
            plt.close()
