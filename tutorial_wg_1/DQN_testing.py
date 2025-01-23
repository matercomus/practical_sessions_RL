import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os
from agent_base import BaseAgent

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class DeepQLearningAgent(BaseAgent):
    def __init__(
        self,
        env,
        discount_rate=0.99,
        learning_rate=0.005,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.99,
        batch_size=128,
        memory_size=20000,
        target_update=500,
        model_path=None
    ):
        self.env = env
        self.discount_rate = discount_rate
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        # Normalization parameters
        self.storage_scale = 170.0
        self.price_scale = np.percentile(self.env.price_values.flatten(), 99)

        # Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(4, 21).to(self.device)
        self.target_net = DQN(4, 21).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=memory_size)
        self.steps = 0
        self.target_update = target_update

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def _normalize_state(self, state):
        storage, price, hour, day = state
        return np.array([
            storage / self.storage_scale,
            price / self.price_scale,
            hour / 24.0,
            day / 7.0
        ], dtype=np.float32)

    def _get_storage_reward(self, storage):
        return 0.5 if 30 <= storage <= 50 else 0

    def choose_action(self, state):
        if random.random() < self.epsilon:
            action_idx = random.randint(0, 20)
        else:
            state_tensor = torch.FloatTensor(self._normalize_state(state)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                action_idx = q_values.max(1)[1].item()

        return action_idx * 0.1 - 1.0

    def update(self, state, action, reward, next_state, done):
        action_idx = int((action + 1.0) * 10)
        reward += self._get_storage_reward(state[0])
        reward = reward / 10.0  # Normalize reward

        self.memory.append((
            self._normalize_state(state),
            action_idx,
            reward,
            self._normalize_state(next_state),
            done
        ))

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        if len(self.memory) >= self.batch_size:
            self._update_network()

        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def _update_network(self):
        batch = random.sample(self.memory, self.batch_size)
        batch = list(zip(*batch))

        states = torch.FloatTensor(np.array(batch[0])).to(self.device)
        actions = torch.LongTensor(batch[1]).to(self.device)
        rewards = torch.FloatTensor(batch[2]).to(self.device)
        next_states = torch.FloatTensor(np.array(batch[3])).to(self.device)
        dones = torch.BoolTensor(batch[4]).to(self.device)

        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + self.discount_rate * max_next_q_values * (~dones)

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, env, episodes, validate_every=None, val_env=None):
        training_rewards = []
        validation_rewards = []
        state_action_history = []

        for episode in range(episodes):
            state = env.observation()
            terminated = False
            episode_reward = 0
            episode_history = []

            while not terminated:
                action = self.choose_action(state)
                next_state, reward, terminated = env.step(action)

                self.update(state, action, reward, next_state, terminated)
                episode_history.append((state, action))
                episode_reward += reward
                state = next_state

            training_rewards.append(episode_reward)
            state_action_history.append(episode_history)

            if validate_every and val_env and (episode + 1) % validate_every == 0:
                val_reward = self.validate(val_env)
                validation_rewards.append((episode + 1, val_reward))

            env.reset()

            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Epsilon = {self.epsilon:.4f}")

        return training_rewards, validation_rewards, state_action_history

    def validate(self, env, num_episodes=10):
        total_reward = 0
        original_epsilon = self.epsilon
        self.epsilon = 0

        for _ in range(num_episodes):
            state = env.observation()
            terminated = False
            episode_reward = 0

            while not terminated:
                action = self.choose_action(state)
                next_state, reward, terminated = env.step(action)
                episode_reward += reward
                state = next_state

            total_reward += episode_reward
            env.reset()

        self.epsilon = original_epsilon
        return total_reward / num_episodes

    def save(self, path):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
