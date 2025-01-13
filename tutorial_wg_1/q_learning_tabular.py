import numpy as np
import h5py
import os
import csv
from dataclasses import dataclass
from typing import Union, List, Tuple


@dataclass
class BinningConfig:
    """Configuration for feature binning with min/max bins"""
    min_threshold: float
    max_threshold: float
    n_bins: int  # Total number of bins including min/max bins
    min_label: str = None  # Optional custom label for min bin
    max_label: str = None  # Optional custom label for max bin


def create_adaptive_bins(
    config: BinningConfig
) -> Tuple[np.ndarray, List[str]]:
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
        raise ValueError("n_bins must be at least 3 (min bin, at least one regular bin, and max bin)")
    
    # Create the bin edges
    regular_bins = np.linspace(
        config.min_threshold, 
        config.max_threshold, 
        config.n_bins - 1
    )
    bins = np.concatenate([[-np.inf], regular_bins, [np.inf]])
    
    # Create labels for each bin
    labels = []
    for i in range(len(bins)-1):
        if i == 0:
            label = config.min_label or f"< {config.min_threshold}"
        elif i == len(bins)-2:
            label = config.max_label or f"> {config.max_threshold}"
        else:
            label = f"{bins[i]:.1f}-{bins[i+1]:.1f}"
        labels.append(label)
    
    return bins, labels


class QLearningAgent:
    def __init__(
        self,
        env,
        discount_rate=0.95,
        learning_rate=0.1,
        epsilon=0.1,
        epsilon_decay=0.999,
        storage_bin_size=3,
        price_config: BinningConfig = None,
        q_table_file="q_table.h5",
    ):
        """
        Initialize Q-Learning Agent with adaptive binning capabilities.
        
        Args:
            env: Environment object containing price_values
            discount_rate: Future reward discount factor
            learning_rate: Q-learning rate
            epsilon: Exploration probability
            epsilon_decay: Rate at which epsilon decays
            storage_bin_size: Number of bins for storage feature
            price_config: BinningConfig for price feature binning
            q_table_file: File path for saving/loading Q-table
        """
        self.env = env
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        
        # Initialize price binning configuration if not provided
        if price_config is None:
            price_config = BinningConfig(
                min_threshold=20,
                max_threshold=150,
                n_bins=5  # Will create: min bin, 3 regular bins, max bin
            )
        
        # Create price bins and labels using the utility function
        self.price_bins, self.price_bin_labels = create_adaptive_bins(price_config)
        
        # Define other bins and action space
        self.action_space = np.linspace(-1, 1, 21)
        self.storage_bins = np.linspace(0, 170, storage_bin_size)
        self.hour_bins = np.arange(1, 25)  # 24 hours
        self.day_bins = np.arange(1, len(env.price_values) + 1)

        # Calculate dimensions for Q-table
        self.n_storage_bins = len(self.storage_bins)
        self.n_price_bins = len(self.price_bins) - 1  # Number of intervals between bin edges
        self.n_hour_bins = len(self.hour_bins)
        self.n_day_bins = len(self.day_bins)
        self.n_actions = len(self.action_space)

        # Initialize or load Q-table
        self._initialize_q_table(q_table_file)
        self.q_table_file = q_table_file

    def _initialize_q_table(self, q_table_file):
        """Initialize Q-table from file or create new one"""
        dimensions = (
            self.n_storage_bins,
            self.n_price_bins,
            self.n_hour_bins,
            self.n_day_bins,
            self.n_actions,
        )

        if os.path.exists(q_table_file):
            try:
                with h5py.File(q_table_file, "r") as f:
                    existing_q_table = f["q_table"][:]
                    if existing_q_table.shape == dimensions:
                        self.Q_table = existing_q_table
                        return
                    print("Existing Q-table has incorrect dimensions. Creating new Q-table.")
            except Exception as e:
                print(f"Error loading Q-table: {e}. Creating new Q-table.")
        
        self.Q_table = np.zeros(dimensions)

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
            np.digitize(storage, self.storage_bins) - 1, 
            self.n_storage_bins - 1
        )
        
        # Handle price discretization using the new binning
        price_idx = min(
            np.digitize(price, self.price_bins) - 1, 
            self.n_price_bins - 1
        )
        
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
                    writer.writerow([
                        self.price_bin_labels[price_idx],
                        action_value,
                        q_value
                    ])
