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
        learning_rate=0.8,
        epsilon_start=1.0,
        epsilon_end=0.0,
        daily_demand=120,  # Typically matches the env's daily demand
        model_path: str = None,
    ):
        """
        Simplified Q-Learning Agent:
        - Ignores day/weekend entirely.
        - Uses shortfall/time-left bins as an 'urgency' feature.
        """

        super().__init__()
        self.env = env
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate

        # Epsilon-greedy exploration settings
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon = epsilon_start

        # Daily demand for shortfall calculations
        self.daily_demand = daily_demand

        # Assume env.price_values has shape (num_days, 24)
        self.total_train_steps = len(env.price_values) * 24
        self.current_step = 0

        # -------------- PRICE BINS --------------
        # Example: 4 bins. Adjust as needed.
        # You could do percentiles, fixed thresholds, etc.
        all_prices = env.price_values.flatten()
        # Here we do a quick percentile-based approach:
        price_percentiles = [0, 25, 50, 75, 100]
        self.price_bins = np.percentile(all_prices, price_percentiles)
        self.n_price_bins = len(self.price_bins) - 1

        # -------------- STORAGE BINS --------------
        # Example: 4 intervals from 0 to 170
        self.storage_bins = np.linspace(0, 170, 5)  # => [0, 42.5, 85, 127.5, 170]
        self.n_storage_bins = len(self.storage_bins) - 1

        # -------------- HOUR BINS --------------
        # Example: 4 intervals => [1..6], [7..12], [13..18], [19..24]
        self.hour_bins = np.arange(1, 25, 6)  # => [1, 7, 13, 19]
        self.n_hour_bins = len(self.hour_bins)

        # -------------- SHORTFALL/TIME-LEFT BINS --------------
        # Bins for ratio = (daily_demand - storage) / (24 - hour)
        # We'll define 3 intervals => [0,2,5,∞]
        self.shortfall_time_bins = [0, 2, 5, float("inf")]
        self.n_shortfall_time_bins = len(self.shortfall_time_bins) - 1

        # -------------- ACTION SPACE --------------
        # 3 actions => [-1, 0, +1]
        self.n_actions = 3
        self.action_space = np.linspace(-1, 1, self.n_actions)

        # -------------- Q-TABLE --------------
        # shape = (storage_bins, price_bins, hour_bins, shortfall_time_bins, actions)
        if model_path and os.path.exists(model_path):
            self.load(model_path)
        else:
            self._initialize_q_table()

    def _initialize_q_table(self):
        """Initialize a zero Q-table."""
        dimensions = (
            self.n_storage_bins,
            self.n_price_bins,
            self.n_hour_bins,
            self.n_shortfall_time_bins,
            self.n_actions,
        )
        self.Q_table = np.zeros(dimensions)

    def print_and_save_q_table_stats(self, path: str):
        """Print and optionally save Q-table stats to JSON."""
        stats = {
            "Storage bins": self.n_storage_bins,
            "Price bins": self.n_price_bins,
            "Hour bins": self.n_hour_bins,
            "Shortfall/Time bins": self.n_shortfall_time_bins,
            "Actions": self.n_actions,
            "Q-table shape": self.Q_table.shape,
            "Total parameters": self.Q_table.size,
        }
        print("Q-table dimensions:")
        for k, v in stats.items():
            print(f"{k}: {v}")

        if path:
            out_file = os.path.join(path, "q_table_stats.json")
            with open(out_file, "w") as json_file:
                json.dump(stats, json_file, indent=4)

    def decay_epsilon(self):
        """Linearly decay epsilon from epsilon_start to epsilon_end over total steps."""
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start
            - (self.epsilon_start - self.epsilon_end)
            * (self.current_step / self.total_train_steps),
        )
        self.current_step += 1

    def _shortfall_time_left_bin(self, storage: float, hour: float) -> int:
        """
        Calculate ratio = (daily_demand - storage) / (24 - hour), then bin it.
        If (24 - hour) <= 0 but shortfall > 0, ratio=∞ => high urgency.
        Bins are [0,2,5,∞].
        """
        shortfall = max(0.0, self.daily_demand - storage)
        time_left = max(0.0, 24 - hour)
        if time_left <= 0:
            ratio = float("inf") if shortfall > 0 else 0.0
        else:
            ratio = shortfall / time_left

        bin_idx = np.digitize(ratio, self.shortfall_time_bins) - 1
        return min(bin_idx, self.n_shortfall_time_bins - 1)

    def discretize_state(self, state):
        """
        State from env = (storage, price, hour, day)
        We ignore 'day'. 
        We map:
          - storage -> storage_idx
          - price   -> price_idx
          - hour    -> hour_idx
          - ratio   -> shortfall_time_idx
        """
        storage, price, hour, _ = state  # ignore day

        # Storage
        storage_idx = np.digitize(storage, self.storage_bins) - 1
        storage_idx = min(storage_idx, self.n_storage_bins - 1)

        # Price
        price_idx = np.digitize(price, self.price_bins) - 1
        price_idx = min(price_idx, self.n_price_bins - 1)

        # Hour
        hour_idx = np.digitize(hour, self.hour_bins) - 1
        hour_idx = min(hour_idx, self.n_hour_bins - 1)

        # Shortfall/time-left ratio
        shortfall_time_idx = self._shortfall_time_left_bin(storage, hour)

        return storage_idx, price_idx, hour_idx, shortfall_time_idx

    def choose_action(self, state):
        """
        Epsilon-greedy action selection: random with prob epsilon, else best from Q-table.
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)  # random among {0,1,2}
        else:
            s_idx, p_idx, h_idx, st_idx = self.discretize_state(state)
            return np.argmax(self.Q_table[s_idx, p_idx, h_idx, st_idx])

    def update(self, state, action, reward, next_state, done):
        """
        Update Q-value with Bellman equation:
         Q(s,a) ← Q(s,a) + α * [r + γ*max_a' Q(s',a') - Q(s,a)]
        """
        s_idx, p_idx, h_idx, st_idx = self.discretize_state(state)
        ns_idx, np_idx, nh_idx, nst_idx = self.discretize_state(next_state)

        current_q = self.Q_table[s_idx, p_idx, h_idx, st_idx, action]

        if done:
            target = reward
        else:
            next_max_q = np.max(self.Q_table[ns_idx, np_idx, nh_idx, nst_idx])
            target = reward + self.discount_rate * next_max_q

        # Update
        self.Q_table[s_idx, p_idx, h_idx, st_idx, action] += \
            self.learning_rate * (target - current_q)

        # Epsilon decay each step
        self.decay_epsilon()

    def train(
        self,
        env,
        episodes: int,
        validate_every: Optional[int] = None,
        val_env=None,
    ) -> Tuple[list, list, list]:
        """
        Main training loop over multiple episodes. Optionally validate at intervals.
        """
        # Reset
        self.current_step = 0
        self.epsilon = self.epsilon_start
        self.total_train_steps = len(env.price_values) * 24 * episodes

        training_rewards = []
        validation_rewards = []
        state_action_history = []

        for episode in range(episodes):
            state = env.observation()
            terminated = False
            episode_reward = 0.0
            episode_history = []

            while not terminated:
                action_idx = self.choose_action(state)
                action = self.action_space[action_idx]  # convert idx to -1,0,1
                next_state, reward, terminated = env.step(action)

                self.update(state, action_idx, reward, next_state, terminated)

                episode_reward += reward
                episode_history.append((state, action))
                state = next_state

            training_rewards.append(episode_reward)
            state_action_history.append(episode_history)

            print(
                f"Episode {episode + 1}: Reward = {episode_reward:.2f}, "
                f"Epsilon = {self.epsilon:.4f}"
            )

            if validate_every and val_env and (episode + 1) % validate_every == 0:
                avg_val_reward = self.validate(val_env)
                validation_rewards.append((episode + 1, avg_val_reward))
                print(
                    f"Validation at Episode {episode + 1}: "
                    f"Avg Reward = {avg_val_reward:.2f}"
                )

            env.reset()

        return training_rewards, validation_rewards, state_action_history

    def validate(self, env, num_episodes: int = 5) -> float:
        """
        Evaluate performance with epsilon=0 (no random exploration).
        """
        original_epsilon = self.epsilon
        self.epsilon = 0.0

        total_reward = 0.0
        for _ in range(num_episodes):
            state = env.observation()
            terminated = False
            episode_reward = 0.0

            while not terminated:
                action_idx = self.choose_action(state)
                action = self.action_space[action_idx]
                next_state, reward, terminated = env.step(action)
                episode_reward += reward
                state = next_state

            total_reward += episode_reward
            env.reset()

        self.epsilon = original_epsilon
        return total_reward / num_episodes

    def save(self, path: str):
        """Save the Q-table and training state to an HDF5 file."""
        with h5py.File(path, "w") as f:
            f.create_dataset("q_table", data=self.Q_table)
            f.attrs["epsilon"] = self.epsilon
            f.attrs["current_step"] = self.current_step
            f.attrs["total_train_steps"] = self.total_train_steps

    def load(self, path: str):
        """Load the Q-table and training state from an HDF5 file."""
        with h5py.File(path, "r") as f:
            self.Q_table = f["q_table"][:]
            self.epsilon = f.attrs.get("epsilon", self.epsilon)
            self.current_step = f.attrs.get("current_step", 0)
            self.total_train_steps = f.attrs.get(
                "total_train_steps", self.total_train_steps
            )

    def save_state_action_history(self, state_action_history: list, path: str):
        """Save (state, action) history per episode to JSON."""
        with open(path, "w") as f:
            json.dump(state_action_history, f, indent=2)