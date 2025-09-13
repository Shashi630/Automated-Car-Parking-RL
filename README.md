# 🚗 Automated-Car-Parking-RL

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29-brightgreen.svg)](https://gymnasium.farama.org/)
[![Stable Baselines3](https://img.shields.io/badge/SB3-2.3.2-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Build](https://github.com/<your-username>/<your-repo>/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/<your-repo>/actions)

An experimental project to train agents for **automated car parking** using **Reinforcement Learning (RL)**.  

## 📂 Project Structure

APA/
│
├── env/
│ ├── car_parking_env.py # 1D discrete parking
│ ├── car_parking_obstacle_env.py # 2D with obstacles (continuous)
│ ├── car_parking_2D_env.py # 2D with obstacles (discrete)
│
├── algo/
│ ├── qlearning_obstacle.py
│ ├── dqn_parking.py 
| |__ qlearning.py
| |__ qlearning2D.py
│ ├── ppo_parking_obstacle.py
│
├── test/
│ ├── test_agent.py
│ ├── test_agent_dqn.py
│ ├── test_agent_ppo_obstacle.py
│
├── results/
│ ├── plots/ # Training reward curves
│ ├── videos/ # Episode recordings (.mp4, .gif)
│ └── tensorboard/ # PPO training logs
│
├── requirements.txt
└── README.md


# Credits

Gymnasium for environment interface.
Stable-Baselines3 for RL algorithms.
