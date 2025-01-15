import numpy as np
import h5py
import os
import csv
from dataclasses import dataclass
from typing import Union, List, Tuple, Any, Optional
from agent_base import BaseAgent


@dataclass
class BinningConfig:
    """Configuration for feature binning with min/max bins"""

    min_threshold: float
    max_threshold: float
    n_bins: int  # Total number of bins including min/max bins
    min_label: str = None  # Optional custom label for min bin
    max_label: str = None  # Optional custom label for max bin


def create_adaptive_bins(config: BinningConfig) -> Tuple[np.ndarray, List[str]]:
    """
    Create bins with min/max bins for outlier handling.

    Args:
        config: BinningConfig object containing binning parameters

    Returns:
        Tuple containing:
        - np.ndarray of bin edges including -inf and inf
        - List of string labels for each bin
    """
    if config.n_bins < 3:
        raise ValueError(
            "n_bins must be at least 3 (min bin, at least one regular bin, and max bin)"
        )

    # Create the bin edges
    regular_bins = np.linspace(
        config.min_threshold, config.max_threshold, config.n_bins - 1
    )
    bins = np.concatenate([[-np.inf], regular_bins, [np.inf]])

    # Create labels for each bin
    labels = []
    for i in range(len(bins) - 1):
        if i == 0:
            label = config.min_label or f"< {config.min_threshold}"
        elif i == len(bins) - 2:
            label = config.max_label or f"> {config.max_threshold}"
        else:
            label = f"{bins[i]:.1f}-{bins[i+1]:.1f}"
        labels.append(label)

    return bins, labels


class QLearningAgent(BaseAgent):
    def __init__(
        self,
        env,
        discount_rate=0.95,
        learning_rate=0.1,
        epsilon=0.1,
        epsilon_decay=0.999,
        storage_bin_size=3,
        price_config: BinningConfig = None,
        model_path: str = None,
    ):
        """Initialize Q-Learning Agent with adaptive binning capabilities."""
        super().__init__()
        self.env = env
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        # Initialize price binning configuration if not provided
        if price_config is None:
            price_config = BinningConfig(min_threshold=20, max_threshold=150, n_bins=5)

        # Create price bins and labels using the utility function
        self.price_bins, self.price_bin_labels = create_adaptive_bins(price_config)

        # Define other bins and action space
        self.action_space = np.linspace(-1, 1, 21)
        self.storage_bins = np.linspace(0, 170, storage_bin_size)
        self.hour_bins = np.arange(1, 25)
        self.day_bins = np.arange(1, len(env.price_values) + 1)

        # Calculate dimensions for Q-table
        self.n_storage_bins = len(self.storage_bins)
        self.n_price_bins = len(self.price_bins) - 1
        self.n_hour_bins = len(self.hour_bins)
        self.n_day_bins = len(self.day_bins)
        self.n_actions = len(self.action_space)

        # Initialize or load Q-table
        if model_path and os.path.exists(model_path):
            self.load(model_path)
        else:
            self._initialize_q_table()

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

    def train(
        self, env, episodes: int, validate_every: Optional[int] = None, val_env=None
    ) -> Tuple[list, list, list]:
        """Train the agent on the environment"""
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

            self.decay_epsilon()
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
        """Run validation episodes and return average reward"""
        total_reward = 0
        for _ in range(num_episodes):
            state = env.observation()
            terminated = False
            episode_reward = 0

            while not terminated:
                action_idx = np.argmax(self.Q_table[self.discretize_state(state)])
                action = self.action_space[action_idx]
                next_state, reward, terminated = env.step(action)
                episode_reward += reward
                state = next_state

            total_reward += episode_reward
            env.reset()

        return total_reward / num_episodes

    def update(
        self, state: Any, action: Any, reward: float, next_state: Any, done: bool
    ):
        """Update the Q-table using the Q-learning formula"""
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

    def save(self, path: str):
        """Save the Q-table to an HDF5 file"""
        with h5py.File(path, "w") as f:
            f.create_dataset("q_table", data=self.Q_table)

    def load(self, path: str):
        """Load the Q-table from an HDF5 file"""
        with h5py.File(path, "r") as f:
            self.Q_table = f["q_table"][:]

    # Keep other existing methods (discretize_state, decay_epsilon, save_q_table_to_csv)...
    def discretize_state(self, state):
        """
        Convert continuous state values into discrete bins.

        Args:
            state: Tuple of (storage, price, hour, day)

        Returns:
            Tuple of discretized indices
        """
        storage, price, hour, day = state

        # Handle storage discretization
        storage_idx = min(
            np.digitize(storage, self.storage_bins) - 1, self.n_storage_bins - 1
        )

        # Handle price discretization using the new binning
        price_idx = min(np.digitize(price, self.price_bins) - 1, self.n_price_bins - 1)

        # Handle hour and day discretization
        hour_idx = min(int(hour - 1), self.n_hour_bins - 1)
        day_idx = min(int(day - 1), self.n_day_bins - 1)

        return storage_idx, price_idx, hour_idx, day_idx

    def choose_action(self, state):
        """
        Choose an action using epsilon-greedy policy.

        Args:
            state: Current state tuple

        Returns:
            Integer index of chosen action
        """
        if np.random.random() < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            storage_idx, price_idx, hour_idx, day_idx = self.discretize_state(state)
            return np.argmax(self.Q_table[storage_idx, price_idx, hour_idx, day_idx])

    def update_Q(self, state, action, reward, next_state, done):
        """
        Update Q-table using the Q-learning formula.

        Args:
            state: Current state tuple
            action: Action taken
            reward: Reward received
            next_state: Resulting state tuple
            done: Whether episode is done
        """
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

    def decay_epsilon(self):
        """Decay epsilon to encourage exploitation over time."""
        self.epsilon *= self.epsilon_decay

    def save_q_table(self):
        """Save the Q-table to an HDF5 file."""
        with h5py.File(self.q_table_file, "w") as f:
            f.create_dataset("q_table", data=self.Q_table)

    def save_q_table_to_csv(self, filename="price_action_q_values.csv"):
        """
        Save price, action, and Q-value to a CSV file with descriptive bin labels.

        Args:
            filename: Output CSV file path
        """
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["price_range", "action", "q_value"])

            for price_idx in range(self.n_price_bins):
                for action_idx, action_value in enumerate(self.action_space):
                    q_value = np.mean(self.Q_table[:, price_idx, :, :, action_idx])
                    writer.writerow(
                        [self.price_bin_labels[price_idx], action_value, q_value]
                    )
