import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os
import json
import logging
from agent_base import BaseAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.network(x)


class ExperienceReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return zip(*batch)

    def __len__(self):
        return len(self.buffer)


class DeepQLearningAgent(BaseAgent):
    def __init__(
        self,
        env,
        discount_rate=0.99,
        learning_rate=0.005,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_episodes=400,
        batch_size=128,
        memory_size=20000,
        target_update=500,
        model_path=None,
    ):
        self.env = env
        self.discount_rate = discount_rate
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.epsilon = epsilon_start
        self.batch_size = batch_size

        # Environment-specific parameters
        self.daily_energy_demand = 120.0  # MWh
        self.max_power_rate = 10.0  # MW

        # Normalization parameters
        self.storage_scale = 170.0
        self.price_scale = np.percentile(self.env.price_values.flatten(), 99)

        # Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(4, 3).to(
            self.device
        )  # 3 actions: sell all, do nothing, buy all
        self.target_net = DQN(4, 3).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ExperienceReplayBuffer(memory_size)
        self.steps = 0
        self.target_update = target_update

        if model_path and os.path.exists(model_path):
            self.load(model_path)

        logger.info("DeepQLearningAgent initialized")

    def _normalize_state(self, state):
        storage, price, hour, day = state
        return np.array(
            [
                storage / self.storage_scale,
                price / self.price_scale,
                hour / 24.0,
                day / 7.0,
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
            action = 1  # Maps to action index 2 (buy)
            logger.debug("Force buy required. Choosing action 1.")
        else:
            # Epsilon-greedy action selection
            if random.random() < self.epsilon:
                action_idx = random.randint(0, 2)
                action = action_idx - 1
                logger.debug(f"Choosing random action: {action}")
            else:
                state_tensor = (
                    torch.FloatTensor(self._normalize_state(state))
                    .unsqueeze(0)
                    .to(self.device)
                )
                with torch.no_grad():
                    q_values = self.policy_net(state_tensor)
                    action_idx = q_values.max(1)[1].item()
                action = action_idx - 1
                logger.debug(f"Choosing action from policy: {action}")

            # Check if selling is disallowed
            if action == -1:
                sell_amount = self.max_power_rate
                potential_storage = storage - sell_amount
                potential_shortfall = self.daily_energy_demand - potential_storage
                hours_left_after = hours_left - 1
                max_buy_after = hours_left_after * self.max_power_rate

                if potential_shortfall > max_buy_after:
                    action = 0  # Disallow sell, set to do nothing
                    logger.debug("Disallowed sell. Setting action to 0.")

        logger.debug(f"Chose action: {action}")
        return action

    def reward_shaping(self, state, action, reward, next_state):
        storage, price, hour, day = state
        next_storage, next_price, next_hour, next_day = next_state
        original_reward = reward
        reward_scale = 1000.0  # Base scaling factor

        # Initialize logging dictionary to track all components
        reward_components = {
            "original_reward": original_reward,
            "normalized_reward": reward / reward_scale,
            "storage_penalty": 0,
            "time_multiplier": 1.0,
            "emergency_penalty": 0,
            "price_bonus": 0,
            "completion_bonus": 0,
        }

        logger.debug(
            f"\nProcessing reward shaping for state: Storage={storage:.2f}, Price={price:.2f}, Hour={hour}, Day={day}"
        )
        logger.debug(
            f"Action taken: {action:.2f}, Original reward: {original_reward:.2f}"
        )

        # Normalize the base reward (financial return)
        reward = reward / reward_scale

        # Calculate remaining hours in the day
        hours_left = 24 - hour if hour < 24 else 0
        logger.debug(f"Hours left in day: {hours_left}")

        # Calculate current shortfall from daily requirement
        shortfall = max(0, self.daily_energy_demand - storage)
        logger.debug(f"Current shortfall: {shortfall:.2f} MWh")

        # 1. Storage Level Management Component
        storage_ratio = storage / self.daily_energy_demand
        target_ratio = hour / 24.0  # Linear target ratio throughout the day
        storage_deviation = abs(storage_ratio - target_ratio)
        storage_penalty = -storage_deviation * 0.2
        reward_components["storage_penalty"] = storage_penalty

        logger.debug(
            f"Storage ratio: {storage_ratio:.2f}, Target ratio: {target_ratio:.2f}"
        )
        logger.debug(
            f"Storage deviation: {storage_deviation:.2f}, Penalty: {storage_penalty:.2f}"
        )

        # 2. Time-Aware Buying Incentive
        buy_hours = set(range(1, 9))  # Hours 1-8
        time_multiplier = 1.2 if hour in buy_hours else 1.0
        reward_components["time_multiplier"] = time_multiplier

        logger.debug(
            f"Hour {hour} {'is' if hour in buy_hours else 'is not'} in buy hours"
        )
        logger.debug(f"Time multiplier: {time_multiplier:.2f}")

        # 3. Emergency Prevention Reward
        safety_margin = self.max_power_rate * hours_left
        emergency_threshold = shortfall - safety_margin
        emergency_penalty = -0.5 if emergency_threshold > 0 else 0
        reward_components["emergency_penalty"] = emergency_penalty

        logger.debug(
            f"Safety margin: {safety_margin:.2f}, Emergency threshold: {emergency_threshold:.2f}"
        )
        logger.debug(f"Emergency penalty: {emergency_penalty:.2f}")

        # 4. Price-Aware Action Reward
        day_prices = self.env.price_values[int(day) - 1]
        price_percentile = np.percentile(day_prices, 75)

        price_bonus = 0
        if action > 0:  # Buying
            price_bonus = 0.2 if price < price_percentile else -0.1
            logger.debug(
                f"Buying: Current price {price:.2f} vs 75th percentile {price_percentile:.2f}"
            )
        elif action < 0:  # Selling
            price_bonus = 0.2 if price > price_percentile else -0.1
            logger.debug(
                f"Selling: Current price {price:.2f} vs 75th percentile {price_percentile:.2f}"
            )
        reward_components["price_bonus"] = price_bonus

        logger.debug(f"Price bonus: {price_bonus:.2f}")

        # 5. End-of-Day Completion Reward
        completion_bonus = 0
        if next_hour == 1 and hour == 24:  # Day transition
            completion_ratio = min(storage / self.daily_energy_demand, 1.0)
            completion_bonus = completion_ratio * 0.5
            reward_components["completion_bonus"] = completion_bonus
            logger.debug(
                f"Day completed. Completion ratio: {completion_ratio:.2f}, Bonus: {completion_bonus:.2f}"
            )

        # Combine all components
        shaped_reward = (
            reward * time_multiplier
            + storage_penalty
            + emergency_penalty
            + price_bonus
            + completion_bonus
        )

        # Log final reward composition
        logger.debug("Reward composition:")
        logger.debug(f"Base reward (normalized): {reward:.2f}")
        logger.debug(f"Storage penalty: {storage_penalty:.2f}")
        logger.debug(
            f"Time multiplier effect: {(reward * time_multiplier - reward):.2f}"
        )
        logger.debug(f"Emergency penalty: {emergency_penalty:.2f}")
        logger.debug(f"Price bonus: {price_bonus:.2f}")
        logger.debug(f"Completion bonus: {completion_bonus:.2f}")
        logger.debug(f"Final shaped reward: {shaped_reward:.2f}")

        # Store reward components for potential analysis
        self.last_reward_components = reward_components

        return shaped_reward, original_reward

    def _get_day_statistics(self, day):
        """Helper method to calculate day-specific price statistics"""
        day_prices = self.env.price_values[day - 1]
        stats = {
            "mean": np.mean(day_prices),
            "median": np.median(day_prices),
            "percentile_75": np.percentile(day_prices, 75),
            "percentile_25": np.percentile(day_prices, 25),
        }

        logger.debug(f"Day {day} price statistics:")
        for key, value in stats.items():
            logger.debug(f"{key}: {value:.2f}")

        return stats

    def print_reward_components(self):
        """Helper method to print the last reward components"""
        if hasattr(self, "last_reward_components"):
            logger.debug("\nLast reward components:")
            for component, value in self.last_reward_components.items():
                logger.debug(f"{component}: {value:.4f}")
        else:
            logger.debug("No reward components available yet")

    def update(self, state, action, reward, next_state, done):
        # Apply reward shaping
        reward, original_reward = self.reward_shaping(state, action, reward, next_state)

        # Convert action to index (0, 1, 2)
        action_idx = action + 1

        self.memory.add(
            (
                self._normalize_state(state),
                action_idx,
                reward,
                self._normalize_state(next_state),
                done,
            )
        )

        if len(self.memory) >= self.batch_size:
            self._update_network()

        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return reward, original_reward

    def _update_network(self):
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # Current Q-values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Double Q-Learning target Q-values
        with torch.no_grad():
            next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            max_next_q_values = (
                self.target_net(next_states).gather(1, next_actions).squeeze(1)
            )
            target_q_values = rewards + self.discount_rate * max_next_q_values * (
                ~dones
            )

        # Loss calculation
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        logger.debug(f"Updated network with loss: {loss.item()}")

    def train(self, env, episodes, validate_every=None, val_env=None):
        logger.info("Starting training")
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

            logger.info(
                f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Epsilon = {self.epsilon:.4f}"
            )

            # Update epsilon linearly
            if episode < self.epsilon_decay_episodes:
                self.epsilon = self.epsilon_start - (
                    episode / self.epsilon_decay_episodes
                ) * (self.epsilon_start - self.epsilon_end)
            else:
                self.epsilon = self.epsilon_end

        logger.info("Training completed")
        return training_rewards, validation_rewards, state_action_history

    def validate(self, env, num_episodes=10):
        logger.info("Starting validation")
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
                updated_reward, original_reward = self.update(
                    state, action, reward, next_state, terminated
                )
                episode_reward += updated_reward
                state = next_state

            total_reward += episode_reward
            env.reset()

        self.epsilon = original_epsilon
        avg_reward = total_reward / num_episodes
        logger.info(f"Validation: Average Reward = {avg_reward:.2f}")
        return avg_reward

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
        logger.info(f"Model saved to {path}")

    def load(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.steps = checkpoint["steps"]
        logger.info(f"Model loaded from {path}")

    def save_state_action_history(self, state_action_history, save_path):
        # Convert NumPy arrays to lists
        serializable_history = [
            [
                (state.tolist(), action, original_reward, updated_reward)
                for state, action, original_reward, updated_reward in episode
            ]
            for episode in state_action_history
        ]
        with open(save_path, "w") as f:
            json.dump(serializable_history, f)
        logger.info(f"State-action history saved to {save_path}")
