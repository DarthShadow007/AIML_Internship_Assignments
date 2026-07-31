"""
Cart-Pole RL Agent Training
Trains a small DQN (Deep Q-Network) agent to solve the CartPole-v1 environment.
Requires: pip install gymnasium torch
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import gymnasium as gym

# -------------------- 1. Q-Network --------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# -------------------- 2. Replay buffer --------------------
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward, dtype=np.float32),
                np.array(next_state), np.array(done, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


# -------------------- 3. Training setup --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = QNetwork(state_dim, action_dim).to(device)
target_net = QNetwork(state_dim, action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
buffer = ReplayBuffer()

gamma = 0.99
batch_size = 64
epsilon_start, epsilon_end, epsilon_decay = 1.0, 0.05, 500
target_update_freq = 10
num_episodes = 300

epsilon = epsilon_start
episode_rewards = []

def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    with torch.no_grad():
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        return policy_net(state_t).argmax().item()

# -------------------- 4. Training loop --------------------
for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = select_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.push(state, action, reward, next_state, float(done))
        state = next_state
        total_reward += reward

        if len(buffer) >= batch_size:
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
            states_t = torch.FloatTensor(states).to(device)
            actions_t = torch.LongTensor(actions).unsqueeze(1).to(device)
            rewards_t = torch.FloatTensor(rewards).to(device)
            next_states_t = torch.FloatTensor(next_states).to(device)
            dones_t = torch.FloatTensor(dones).to(device)

            q_values = policy_net(states_t).gather(1, actions_t).squeeze(1)
            with torch.no_grad():
                next_q_values = target_net(next_states_t).max(1)[0]
                target_q = rewards_t + gamma * next_q_values * (1 - dones_t)

            loss = nn.functional.mse_loss(q_values, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    epsilon = max(epsilon_end, epsilon_start - episode / epsilon_decay)
    episode_rewards.append(total_reward)

    if episode % target_update_freq == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if episode % 20 == 0:
        avg_reward = np.mean(episode_rewards[-20:])
        print(f"Episode {episode:4d} | Avg reward (last 20): {avg_reward:6.2f} | Epsilon: {epsilon:.3f}")

env.close()

avg_last_50 = np.mean(episode_rewards[-50:])
print(f"\nTraining complete. Average reward over last 50 episodes: {avg_last_50:.2f}")
print("CartPole-v1 is considered solved at an average reward of 475 over 100 episodes.")

torch.save(policy_net.state_dict(), "cartpole_dqn.pth")
print("Model saved to cartpole_dqn.pth")