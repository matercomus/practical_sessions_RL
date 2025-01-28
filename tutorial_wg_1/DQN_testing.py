import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import json
import logging
from agent_base import BaseAgent

# Configure logging - only enable info and error by default for performance
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        # Simplified network architecture with proper initialization for better training
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

        # Initialize weights using Xavier initialization
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.network(x)


class ExperienceReplayBuffer:
    def __init__(self, capacity):
        self.buffer = np.zeros(capacity, dtype=object)  # Pre-allocate buffer
        self.capacity = capacity
        self.position = 0
        self.size = 0

    def add(self, transition):
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.choice(self.size, batch_size, replace=False)
        batch = self.buffer[indices]
        return zip(*batch)

    def __len__(self):
        return self.size


class DeepQLearningAgent(BaseAgent):
    def __init__(
        self,
        env,
        discount_rate=0.75,
        learning_rate=0.001,  # Reduced learning rate for stability
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_episodes=3,
        batch_size=128,
        memory_size=20000,
        target_update=500,
        update_frequency=4,  # New parameter for controlling update frequency
        model_path=None,
    ):
        self.env = env
        self.discount_rate = discount_rate
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.epsilon = epsilon_start
        self.batch_size = batch_size
        self.update_frequency = update_frequency

        # Cache environment parameters
        self.daily_energy_demand = 120.0
        self.max_power_rate = 10.0
        self.storage_scale = 170.0
        self.price_scale = np.percentile(self.env.price_values.flatten(), 99)

        # Pre-calculate day statistics
        self.day_stats_cache = self._precalculate_day_statistics()

        # Initialize networks and optimize for GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(4, 3).to(self.device)
        self.target_net = DQN(4, 3).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set target network to evaluation mode

        # Use Adam optimizer with improved parameters
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8
        )

        self.memory = ExperienceReplayBuffer(memory_size)
        self.steps = 0
        self.target_update = target_update

        if model_path and os.path.exists(model_path):
            self.load(model_path)

        logger.info(f"DeepQLearningAgent initialized on device: {self.device}")

    def _precalculate_day_statistics(self):
        """Pre-calculate statistics for each day to avoid repeated calculations"""
        stats_cache = {}
        for day in range(len(self.env.price_values)):
            day_prices = self.env.price_values[day]
            stats_cache[day + 1] = {
                "mean": np.mean(day_prices),
                "median": np.median(day_prices),
                "percentile_75": np.percentile(day_prices, 75),
                "percentile_25": np.percentile(day_prices, 25),
            }
        return stats_cache

    def _normalize_state(self, state):
        """Vectorized state normalization"""
        return np.array(
            [
                state[0] / self.storage_scale,
                state[1] / self.price_scale,
                state[2] / 24.0,
                state[3] / 7.0,
            ],
            dtype=np.float32,
        )

    def choose_action(self, state):
        storage = state[0]
        hour = int(state[2])
        hours_left = 24 - hour
        shortfall = self.daily_energy_demand - storage
        max_possible_buy = hours_left * self.max_power_rate

        # Force buy condition
        if shortfall > max_possible_buy:
            return 1  # Buy action

        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            return random.randint(-1, 1)

        # Get action from policy network
        with torch.no_grad():
            state_tensor = (
                torch.FloatTensor(self._normalize_state(state))
                .unsqueeze(0)
                .to(self.device)
            )
            q_values = self.policy_net(state_tensor)
            action = q_values.max(1)[1].item() - 1

        # Check if selling is disallowed
        if action == -1:
            sell_amount = self.max_power_rate
            potential_storage = storage - sell_amount
            potential_shortfall = self.daily_energy_demand - potential_storage
            hours_left_after = hours_left - 1
            max_buy_after = hours_left_after * self.max_power_rate

            if potential_shortfall > max_buy_after:
                action = 0  # Disallow sell

        return action

    def reward_shaping(self, state, action, reward, next_state):
        """Simplified reward shaping with minimal logging"""
        storage, price, hour, day = state
        next_storage, next_price, next_hour, next_day = next_state
        reward_scale = 1000.0

        # Base reward normalization
        shaped_reward = reward / reward_scale

        # Critical components only
        hours_left = 24 - hour if hour < 24 else 0
        shortfall = max(0, self.daily_energy_demand - storage)

        # Storage management penalty
        storage_ratio = storage / self.daily_energy_demand
        target_ratio = hour / 24.0
        storage_penalty = -abs(storage_ratio - target_ratio) * 0.2
        shaped_reward += storage_penalty

        # Emergency prevention
        safety_margin = self.max_power_rate * hours_left
        if shortfall > safety_margin:
            shaped_reward -= 0.5

        # End-of-day completion bonus
        if next_hour == 1 and hour == 24:
            completion_ratio = min(storage / self.daily_energy_demand, 1.0)
            shaped_reward += completion_ratio * 0.5

        return shaped_reward, reward

    def update(self, state, action, reward, next_state, done):
        # Apply reward shaping
        reward, original_reward = self.reward_shaping(state, action, reward, next_state)
        action_idx = action + 1

        # Store normalized states
        normalized_state = self._normalize_state(state)
        normalized_next_state = self._normalize_state(next_state)

        self.memory.add(
            (normalized_state, action_idx, reward, normalized_next_state, done)
        )

        # Update network less frequently
        if (
            len(self.memory) >= self.batch_size
            and self.steps % self.update_frequency == 0
        ):
            self._update_network()

        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return reward, original_reward

    def _update_network(self):
        # Batch processing with GPU optimization
        with torch.no_grad():
            states, actions, rewards, next_states, dones = self.memory.sample(
                self.batch_size
            )

            # Convert to tensors and move to device
            states = torch.FloatTensor(np.array(states)).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
            dones = torch.BoolTensor(dones).to(self.device)

            # Double Q-learning target calculation
            next_q_values = self.policy_net(next_states)
            next_actions = next_q_values.max(1)[1].unsqueeze(1)
            max_next_q_values = (
                self.target_net(next_states).gather(1, next_actions).squeeze(1)
            )
            target_q_values = rewards + self.discount_rate * max_next_q_values * (
                ~dones
            )

        # Current Q-values and loss calculation
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Optimization step
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
                updated_reward, original_reward = self.update(
                    state, action, reward, next_state, terminated
                )
                episode_history.append((state, action, original_reward, updated_reward))
                episode_reward += updated_reward
                state = next_state

            training_rewards.append(episode_reward)
            state_action_history.append(episode_history)

            if validate_every and val_env and (episode + 1) % validate_every == 0:
                val_reward = self.validate(val_env)
                validation_rewards.append((episode + 1, val_reward))

            env.reset()

            # Update epsilon with linear decay
            if episode < self.epsilon_decay_episodes:
                self.epsilon = self.epsilon_start - (
                    episode / self.epsilon_decay_episodes
                ) * (self.epsilon_start - self.epsilon_end)
            else:
                self.epsilon = self.epsilon_end

            # Log only every 10 episodes for performance
            if (episode + 1) % 10 == 0:
                logger.info(
                    f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Epsilon = {self.epsilon:.4f}"
                )

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
                updated_reward, _ = self.reward_shaping(
                    state, action, reward, next_state
                )
                episode_reward += updated_reward
                state = next_state

            total_reward += episode_reward
            env.reset()

        self.epsilon = original_epsilon
        return total_reward / num_episodes

    def save(self, path):
        torch.save(
            {
                "policy_net_state_dict": self.policy_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "steps": self.steps,
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.steps = checkpoint["steps"]

    def save_state_action_history(self, state_action_history, save_path):
        serializable_history = []
        for episode in state_action_history:
            episode_data = []
            for state, action, original_reward, updated_reward in episode:
                episode_data.append(
                    (state.tolist(), action, original_reward, updated_reward)
                )
            serializable_history.append(episode_data)

        with open(save_path, "w") as f:
            json.dump(serializable_history, f)
