import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

class parking2dEnvWithObstacle(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, grid_size=10, target=(7.0, 3.0), obstacles=None, parking_spots=None, max_steps=100, render_mode=None):
        super().__init__()

        self.car_img = mpimg.imread("assets/car.png")
        self.wall_img = mpimg.imread("assets/wall.png")
        self.parked_car_img = mpimg.imread("assets/parked_car.png")

        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)  
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0]), high=np.array([grid_size, grid_size]), dtype=np.float32)

        self.grid_size = grid_size
        self.target = np.array(target, dtype = np.float32)
        self.max_steps = max_steps
        self.steps = 0
        self.x, self.y = 0, 0
        self.render_mode = render_mode

        self.obstacles = obstacles if obstacles else [(2.5, 3.5), (4.0, 4.0), (6.5, 2.2)]
        self.obstacle_radius = 0.5

        self.parking_spots = parking_spots if parking_spots else [(3.0, 3.0), (7.0, 3.0), (8.0, 8.0)]
        self.parking_radius = 0.7

        if render_mode == "human":
            self.fig, self.ax = plt.subplots()
            self.ax.set_xlim(0, grid_size)
            self.ax.set_ylim(0, grid_size)
            self.ax.set_xticks(range(grid_size + 1))
            self.ax.set_yticks(range(grid_size + 1))
            self.ax.grid(True)

            self.obs_scatter = self.ax.scatter([], [], c="black", s=200, label="Obstacle")

            self.spot_scatter = self.ax.scatter([], [], c="blue", s=200, label="Parking Spot")

            self.target_marker = self.ax.scatter([], [], c="green", s=200, label="Target")

            self.car_marker, = self.ax.plot([], [], "ro", markersize=10, label="Car")

            self.ax.legend(loc="upper right")
            plt.ion()
            plt.show()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.x = np.random.uniform(0, self.grid_size)
        self.y = np.random.uniform(0, self.grid_size)
        self.car_pos = (self.x, self.y) 
        self.steps = 0
        return np.array([self.x, self.y], dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        dx, dy = action
        self.x = np.clip(self.x + dx, 0, self.grid_size)
        self.y = np.clip(self.y + dy, 0, self.grid_size)
        self.car_pos = (self.x, self.y)

        reward = -0.5  # small step penalty
        terminated = False

        # Check obstacle collision
        for (ox, oy) in self.obstacles:
            if np.linalg.norm([self.x - ox, self.y - oy]) < self.obstacle_radius:
                reward = -20
                terminated = True
                break

        # Check parking
        for (px, py) in self.parking_spots:
            if np.linalg.norm([self.x - px, self.y - py]) < self.parking_radius:
                if np.linalg.norm([self.x - self.target[0], self.y - self.target[1]]) < self.parking_radius:
                    reward = 100  # correct parking spot
                else:
                    reward = -10  # wrong parking spot
                terminated = True
                break

        truncated = self.steps >= self.max_steps
        return np.array([self.x, self.y], dtype=np.float32), reward, terminated, truncated, {}


    def render(self):
        self.ax.clear()
        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)
        self.ax.grid(True)

        # Draw obstacles as wall images
        for ox, oy in self.obstacles:
            self.ax.imshow(self.wall_img, extent=[ox-0.5, ox+0.5, oy-0.5, oy+0.5], zorder=1)

        # Draw parking spots as parked cars inside blue rectangles
        for px, py in self.parking_spots:
            if (px, py) == tuple(self.target):
                continue  # skip if this is the target

            rect = patches.Rectangle((px-0.5, py-0.5), 1, 1,
                                    linewidth=2, edgecolor="blue",
                                    facecolor="none", zorder=2)
            self.ax.add_patch(rect)
            self.ax.imshow(self.parked_car_img, extent=[px-0.5, px+0.5, py-0.5, py+0.5], zorder=3)


        # Draw target as green rectangle
        tx, ty = self.target
        target_rect = patches.Rectangle((tx-0.5, ty-0.5), 1, 1, linewidth=2, edgecolor="green", facecolor="green", alpha=0.3, zorder=2)
        self.ax.add_patch(target_rect)

        # Draw car as image
        cx, cy = self.car_pos
        self.ax.imshow(self.car_img, extent=[cx-0.5, cx+0.5, cy-0.5, cy+0.5], zorder=4)

        self.ax.legend(
            handles=[
                patches.Patch(color="green", label="Target"),
                patches.Patch(color="blue", label="Parking Spot"),
                patches.Patch(color="black", label="Obstacle"),
                patches.Patch(color="red", label="Car"),
            ],
            loc="upper right"
        )

        plt.pause(0.01)

        if self.render_mode == "human":
            # Update obstacles
            if self.obstacles:
                ox, oy = zip(*self.obstacles)
                self.obs_scatter.set_offsets(np.c_[ox, oy])

            # Update parking spots
            if self.parking_spots:
                px, py = zip(*self.parking_spots)
                self.spot_scatter.set_offsets(np.c_[px, py])

            # Update target
            self.target_marker.set_offsets([self.target])

            # Update car
            self.car_marker.set_data(self.x, self.y)

            # Redraw
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def close(self):
        if self.render_mode == "human":
            plt.ioff()
            plt.close()
