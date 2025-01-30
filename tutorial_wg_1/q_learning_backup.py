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
        discount_rate=0.8,
        learning_rate=0.9,
        epsilon_start=1.0,
        epsilon_end=0.0,
        model_path: str = None,
    ):
        """
        Q-Learning Agent that ignores the 'day' dimension, uses fewer bins
        for storage, and sets maximum storage to 170 MWh in its discretization.
        """
        super().__init__()
        self.env = env
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate

        # Epsilon parameters (exploration)
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon = epsilon_start

        # Calculate total training steps (for epsilon decay)
        # Assuming env.price_values has shape (#days, 24)
        self.total_train_steps = len(env.price_values) * 24
        self.current_step = 0

        # -------------------------
        # 1) Price binning (fewer bins)
        # -------------------------
        # Flatten all prices and create percentiles
        all_prices = env.price_values.flatten()
        print(all_prices)
        price_percentiles = [0, 25, 50, 75, 100]  # 4 intervals
        self.price_bins = np.percentile(all_prices, price_percentiles)
        print(self.price_bins)
        self.n_price_bins = len(self.price_bins) - 1

        # -------------------------
        # 2) Storage binning (0..170 MWh)
        # -------------------------
        # e.g., 0..170 in 10 intervals
        self.storage_bins = np.linspace(0, 170, 11)
        self.n_storage_bins = len(self.storage_bins) - 1

        # -------------------------
        # 3) Hour binning (fewer bins)
        # -------------------------
        # Example: 4 intervals -> [1..6], [7..12], [13..18], [19..24]
        # (Note that np.arange(1, 25, 6) => [1, 7, 13, 19])
        self.hour_bins = np.arange(1, 25, 6)
        self.n_hour_bins = len(self.hour_bins)

        # -------------------------
        # 4) Action space
        # -------------------------
        # 3 discrete actions mapped to -1.0, 0.0, +1.0
        self.n_actions = 3
        self.action_space = np.linspace(-1, 1, self.n_actions)

        # -------------------------
        # 5) Q-table initialization
        # -------------------------
        # Day dimension is REMOVED, so shape = (storage_bins, price_bins, hour_bins, actions)
        if model_path and os.path.exists(model_path):
            self.load(model_path)
        else:
            self._initialize_q_table()

    def _initialize_q_table(self):
        """Initialize Q-table with zeros."""
        # No day dimension
        dimensions = (
            self.n_storage_bins,
            self.n_price_bins,
            self.n_hour_bins,
            self.n_actions,
        )
        self.Q_table = np.zeros(dimensions)

    def print_and_save_q_table_stats(self, path: str):
        """Print Q-table statistics and save to JSON"""
        stats = {
            "Storage bins": self.n_storage_bins,
            "Price bins": self.n_price_bins,
            "Hour bins": self.n_hour_bins,
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

    def decay_epsilon(self):
        """
        Linear decay schedule based on the current step vs total steps.
        Epsilon can't go below epsilon_end.
        """
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start
            - (self.epsilon_start - self.epsilon_end)
            * (self.current_step / self.total_train_steps),
        )
        self.current_step += 1

    def discretize_state(self, state):
        """
        Convert continuous state values into discrete bins.
        State = (storage, price, hour, day)
        We IGNORE 'day' by not creating a day index.
        """
        storage, price, hour, _ = state  # day is ignored

        # Storage index
        storage_idx = np.digitize(storage, self.storage_bins) - 1
        storage_idx = min(storage_idx, self.n_storage_bins - 1)

        # Price index
        price_idx = np.digitize(price, self.price_bins) - 1
        price_idx = min(price_idx, self.n_price_bins - 1)

        # Hour index
        hour_idx = np.digitize(hour, self.hour_bins) - 1
        hour_idx = min(hour_idx, self.n_hour_bins - 1)

        return storage_idx, price_idx, hour_idx

    def choose_action(self, state):
        """
        Epsilon-greedy action selection.
        Either pick a random action (exploration) or the best known action
        from the Q-table (exploitation).
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            s_idx, p_idx, h_idx = self.discretize_state(state)
            return np.argmax(self.Q_table[s_idx, p_idx, h_idx])

    def update(self, state, action, reward, next_state, done):
        """
        Update Q-table using the Bellman equation:
        Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]
        """
        s_idx, p_idx, h_idx = self.discretize_state(state)
        ns_idx, np_idx, nh_idx = self.discretize_state(next_state)

        current_q = self.Q_table[s_idx, p_idx, h_idx, action]

        if done:
            target = reward
        else:
            next_max_q = np.max(self.Q_table[ns_idx, np_idx, nh_idx])
            target = reward + self.discount_rate * next_max_q

        # Update rule
        self.Q_table[s_idx, p_idx, h_idx, action] += self.learning_rate * (
            target - current_q
        )

        # Decay epsilon after each update step
        self.decay_epsilon()

    def train(
        self, env, episodes: int, validate_every: Optional[int] = None, val_env=None
    ) -> Tuple[list, list, list]:
        """
        Train the agent over multiple episodes.
        Optionally run validation at a specified interval.
        """
        # Reset for new training run
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
                action = self.action_space[action_idx]  # -1, 0, +1
                next_state, reward, terminated = env.step(action)

                # Update Q-table
                self.update(state, action_idx, reward, next_state, terminated)

                # Log
                episode_reward += reward
                episode_history.append((state, action))
                state = next_state

            # After episode ends
            training_rewards.append(episode_reward)
            state_action_history.append(episode_history)

            print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Epsilon = {self.epsilon:.4f}")

            # Validation
            if validate_every and val_env and (episode + 1) % validate_every == 0:
                avg_validation_reward = self.validate(val_env)
                validation_rewards.append((episode + 1, avg_validation_reward))
                print(
                    f"Validation at Episode {episode + 1}: Avg Reward = {avg_validation_reward:.2f}"
                )

            env.reset()

        return training_rewards, validation_rewards, state_action_history

    def validate(self, env, num_episodes: int = 5) -> float:
        """
        Evaluate the agent's performance with exploration disabled (epsilon=0).
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
        """Save Q-table and training state to an HDF5 file."""
        with h5py.File(path, "w") as f:
            f.create_dataset("q_table", data=self.Q_table)
            f.attrs["epsilon"] = self.epsilon
            f.attrs["current_step"] = self.current_step
            f.attrs["total_train_steps"] = self.total_train_steps

    def load(self, path: str):
        """Load Q-table and training state from an HDF5 file."""
        with h5py.File(path, "r") as f:
            self.Q_table = f["q_table"][:]
            self.epsilon = f.attrs.get("epsilon", self.epsilon)
            self.current_step = f.attrs.get("current_step", 0)
            self.total_train_steps = f.attrs.get(
                "total_train_steps", self.total_train_steps
            )

    def save_state_action_history(self, state_action_history: List, path: str):
        """Save state-action pairs for each episode to a JSON file."""
        with open(path, "w") as f:
            json.dump(state_action_history, f)