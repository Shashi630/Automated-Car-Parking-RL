import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class CarParkingPhysicsEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 15}

    def __init__(self, grid_size=10, render_mode=None, max_steps=200):
        super().__init__()
        self.grid_size = grid_size
        self.max_speed = 1.0
        self.dt = 0.1
        self.max_steps = max_steps

        # state = [x, y, theta, velocity]
        high = np.array([grid_size, grid_size, np.pi, self.max_speed])
        low = np.array([0.0, 0.0, -np.pi, -self.max_speed])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # action = [steering, throttle]
        self.action_space = spaces.Box(
            low=np.array([-0.5, -1.0]),
            high=np.array([0.5, 1.0]),
            dtype=np.float32
        )

        # Environment elements
        self.target = np.array([7.0, 3.0])  # one correct parking spot
        self.parking_spots = [(3.0, 3.0), (7.0, 3.0), (8.0, 8.0)]  # multiple options
        self.obstacles = [(4.5, 4.5), (6.0, 6.0), (2.0, 7.0)]
        self.obstacle_radius = 0.5
        self.parking_radius = 0.7

        self.render_mode = render_mode
        if render_mode == "human":
            self.fig, self.ax = plt.subplots()

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.x, self.y = np.random.uniform(0, self.grid_size, 2)
        self.theta = 0.0
        self.v = 0.0
        self.steps = 0
        return np.array([self.x, self.y, self.theta, self.v], dtype=np.float32), {}

    def step(self, action):
        steering, throttle = action
        self.v = np.clip(self.v + throttle * self.dt, -self.max_speed, self.max_speed)
        self.theta += steering * self.dt
        self.x = np.clip(self.x + self.v * np.cos(self.theta) * self.dt, 0, self.grid_size)
        self.y = np.clip(self.y + self.v * np.sin(self.theta) * self.dt, 0, self.grid_size)

        self.steps += 1
        reward, terminated = -0.1, False

        # Check obstacle collision
        for (ox, oy) in self.obstacles:
            if np.linalg.norm([self.x - ox, self.y - oy]) < self.obstacle_radius:
                reward, terminated = -50, True
                break

        # Check parking spots
        for (px, py) in self.parking_spots:
            if np.linalg.norm([self.x - px, self.y - py]) < self.parking_radius:
                if np.linalg.norm([self.x - self.target[0], self.y - self.target[1]]) < self.parking_radius:
                    reward, terminated = 100, True  # correct spot
                else:
                    reward, terminated = -20, True  # wrong spot
                break

        if self.steps >= self.max_steps:
            terminated = True

        return np.array([self.x, self.y, self.theta, self.v], dtype=np.float32), reward, terminated, False, {}

    def render(self):
        if self.render_mode == "human":
            self.ax.clear()
            self.ax.set_xlim(0, self.grid_size)
            self.ax.set_ylim(0, self.grid_size)

            # Draw obstacles (black squares)
            for ox, oy in self.obstacles:
                circ = plt.Circle((ox, oy), self.obstacle_radius, color="black")
                self.ax.add_patch(circ)

            # Draw parking spots (blue rectangles)
            for px, py in self.parking_spots:
                rect = patches.Rectangle((px-0.5, py-0.5), 1, 1,
                                         linewidth=2, edgecolor="blue", facecolor="none")
                self.ax.add_patch(rect)

            # Draw target (green rectangle)
            tx, ty = self.target
            rect = patches.Rectangle((tx-0.5, ty-0.5), 1, 1,
                                     linewidth=2, edgecolor="green", facecolor="green", alpha=0.3)
            self.ax.add_patch(rect)

            # Draw car (red dot)
            self.ax.plot(self.x, self.y, "ro", label="Car")

            self.ax.legend(loc="upper right")
            plt.pause(0.01)

    def close(self):
        if self.render_mode == "human":
            plt.ioff()
            plt.close()
