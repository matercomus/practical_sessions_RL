import numpy as np
import h5py
import os
import json
from typing import List, Tuple, Any, Optional
from agent_base import BaseAgent


class QLearningAgent(BaseAgent):
    def __init__(
        self,
        env,
        discount_rate=0.95,
        learning_rate=0.1,
        epsilon_start=1.0,
        epsilon_end=0.1,
        model_path: str = None,
    ):
        """Initialize Q-Learning Agent with calculated linear epsilon decay"""
        super().__init__()
        self.env = env
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate

        # Epsilon parameters
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon = epsilon_start

        # Calculate total training steps from data
        self.total_train_steps = len(env.price_values) * 24  # days * hours per day
        self.current_step = 0

        # Initialize price binning using percentile-based approach
        all_prices = env.price_values.flatten()
        price_percentiles = [0, 20, 40, 60, 80, 100]
        self.price_bins = np.percentile(all_prices, price_percentiles)

        # Create non-uniform storage bins with finer granularity around daily target
        daily_target = 120  # Your daily energy demand
        critical_storage_bins = np.linspace(daily_target - 20, daily_target + 20, 10)
        lower_storage_bins = np.linspace(0, daily_target - 20, 5)
        upper_storage_bins = np.linspace(daily_target + 20, 170, 5)
        self.storage_bins = np.unique(
            np.concatenate(
                [lower_storage_bins, critical_storage_bins, upper_storage_bins]
            )
        )

        # Time-based bins
        self.hour_bins = np.arange(1, 25, 2)  # 12 bins (2-hour periods)
        self.day_bins = np.arange(1, 8)  # 7 bins (days of week)

        # Calculate dimensions for Q-table
        self.n_storage_bins = len(self.storage_bins) - 1
        self.n_price_bins = len(self.price_bins) - 1
        self.n_hour_bins = len(self.hour_bins)
        self.n_day_bins = len(self.day_bins)
        self.n_actions = 3
        self.action_space = np.linspace(-1, 1, self.n_actions)

        # Initialize or load Q-table
        if model_path and os.path.exists(model_path):
            self.load(model_path)
        else:
            self._initialize_q_table()

    def print_and_save_q_table_stats(self, path: str):
        """Print Q-table statistics and save to JSON"""
        stats = {
            "Storage bins": self.n_storage_bins,
            "Price bins": self.n_price_bins,
            "Hour bins": self.n_hour_bins,
            "Day bins": self.n_day_bins,
            "Actions": self.n_actions,
            "Q-table shape": self.Q_table.shape,
            "Total parameters": self.Q_table.size,
        }

        print("Q-table dimensions:")
        for key, value in stats.items():
            print(f"{key}: {value}")

        if path:
            stats_path = os.path.join(path, "q_table_stats.json")
            with open(stats_path, "w") as json_file:
                json.dump(stats, json_file, indent=4)

    def _initialize_q_table(self):
        """Initialize new Q-table"""
        dimensions = (
            self.n_storage_bins,
            self.n_price_bins,
            self.n_hour_bins,
            self.n_day_bins,
            self.n_actions,
        )
        self.Q_table = np.zeros(dimensions)

    def decay_epsilon(self):
        """Linear decay schedule based on actual steps"""
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start
            - (self.epsilon_start - self.epsilon_end)
            * (self.current_step / self.total_train_steps),
        )
        self.current_step += 1

    def discretize_state(self, state):
        """Convert continuous state values into discrete bins"""
        storage, price, hour, day = state

        storage_idx = np.digitize(storage, self.storage_bins) - 1
        storage_idx = min(storage_idx, self.n_storage_bins - 1)

        price_idx = np.digitize(price, self.price_bins) - 1
        price_idx = min(price_idx, self.n_price_bins - 1)

        hour_idx = np.digitize(hour, self.hour_bins) - 1
        hour_idx = min(hour_idx, self.n_hour_bins - 1)

        day_idx = min(int(day - 1), self.n_day_bins - 1)

        return storage_idx, price_idx, hour_idx, day_idx

    def choose_action(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            storage_idx, price_idx, hour_idx, day_idx = self.discretize_state(state)
            return np.argmax(self.Q_table[storage_idx, price_idx, hour_idx, day_idx])

    def update(
        self, state: Any, action: Any, reward: float, next_state: Any, done: bool
    ):
        """Update Q-table and decay epsilon"""
        storage_idx, price_idx, hour_idx, day_idx = self.discretize_state(state)
        next_storage_idx, next_price_idx, next_hour_idx, next_day_idx = (
            self.discretize_state(next_state)
        )

        current_q = self.Q_table[storage_idx, price_idx, hour_idx, day_idx, action]
        if done:
            target = reward
        else:
            next_max_q = np.max(
                self.Q_table[
                    next_storage_idx, next_price_idx, next_hour_idx, next_day_idx
                ]
            )
            target = reward + self.discount_rate * next_max_q

        self.Q_table[
            storage_idx, price_idx, hour_idx, day_idx, action
        ] += self.learning_rate * (target - current_q)

        # Decay epsilon after each step
        self.decay_epsilon()

    def train(
        self, env, episodes: int, validate_every: Optional[int] = None, val_env=None
    ) -> Tuple[list, list, list]:
        """Train the agent on the environment"""
        # Reset steps and epsilon at start of training
        self.current_step = 0
        self.epsilon = self.epsilon_start

        # Adjust total steps based on episodes
        self.total_train_steps = len(env.price_values) * 24 * episodes

        training_rewards = []
        validation_rewards = []
        state_action_history = []

        for episode in range(episodes):
            state = env.observation()
            terminated = False
            episode_reward = 0
            episode_history = []

            while not terminated:
                action_idx = self.choose_action(state)
                action = self.action_space[action_idx]
                next_state, reward, terminated = env.step(action)
                episode_reward += reward

                self.update(state, action_idx, reward, next_state, terminated)
                episode_history.append((state, action))
                state = next_state

            training_rewards.append(episode_reward)
            state_action_history.append(episode_history)

            print(
                f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Epsilon = {self.epsilon:.4f}"
            )

            if validate_every and val_env and (episode + 1) % validate_every == 0:
                avg_validation_reward = self.validate(val_env)
                validation_rewards.append((episode + 1, avg_validation_reward))
                print(
                    f"Validation at Episode {episode + 1}: Avg Reward = {avg_validation_reward:.2f}"
                )

            env.reset()

        return training_rewards, validation_rewards, state_action_history

    def validate(self, env, num_episodes: int = 10) -> float:
        """Run validation episodes with exploration disabled"""
        total_reward = 0
        original_epsilon = self.epsilon
        self.epsilon = 0  # Disable exploration during validation

        for _ in range(num_episodes):
            state = env.observation()
            terminated = False
            episode_reward = 0

            while not terminated:
                action_idx = self.choose_action(state)
                action = self.action_space[action_idx]
                next_state, reward, terminated = env.step(action)
                episode_reward += reward
                state = next_state

            total_reward += episode_reward
            env.reset()

        self.epsilon = original_epsilon  # Restore original epsilon
        return total_reward / num_episodes

    def save(self, path: str):
        """Save the Q-table and training state"""
        with h5py.File(path, "w") as f:
            f.create_dataset("q_table", data=self.Q_table)
            f.attrs["epsilon"] = self.epsilon
            f.attrs["current_step"] = self.current_step
            f.attrs["total_train_steps"] = self.total_train_steps

    def save_state_action_history(self, state_action_history: List, path: str):
        """Save the state-action history to a JSON file"""
        with open(path, "w") as f:
            json.dump(state_action_history, f)

    def load(self, path: str):
        """Load the Q-table and training state"""
        with h5py.File(path, "r") as f:
            self.Q_table = f["q_table"][:]
            self.epsilon = f.attrs.get("epsilon", self.epsilon)
            self.current_step = f.attrs.get("current_step", 0)
            self.total_train_steps = f.attrs.get(
                "total_train_steps", self.total_train_steps
            )
