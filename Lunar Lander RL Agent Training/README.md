# Assignment 7: Lunar Lander RL Agent Training

## Overview
This assignment implements a Deep Q-Network (DQN) Reinforcement Learning agent in PyTorch designed to solve the Lunar Lander environment.

## Technical Details
* **Algorithm**: Deep Q-Learning (DQN) with Experience Replay
* **Framework**: PyTorch & Gymnasium
* **State Space**: Continuous 8-dimensional state vector (position, velocity, orientation, leg contact)
* **Action Space**: Discrete 4-action space (do nothing, fire left engine, fire main engine, fire right engine)

## How to Run
```bash
python 07_lunar_lander.py