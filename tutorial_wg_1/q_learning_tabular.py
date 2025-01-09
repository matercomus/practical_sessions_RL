import numpy as np
import h5py
import os
import csv

class QLearningAgent:
    def __init__(self, env, discount_rate=0.95, learning_rate=0.1, epsilon=0.1, epsilon_decay=0.999, bin_size=20, q_table_file='q_table.h5'):
        self.env = env
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.bin_size = bin_size

        # Define action space and state discretization
        self.action_space = np.linspace(-1, 1, 11)
        self.storage_bins = np.linspace(0, 170, self.bin_size)
        self.price_bins = np.linspace(0, 200, self.bin_size)
        self.hour_bins = np.arange(1, 25)
        self.day_bins = np.arange(1, len(env.price_values) + 1)

        # Check if the Q-table already exists in a file
        if os.path.exists(q_table_file):
            with h5py.File(q_table_file, 'r') as f:
                self.Q_table = f['q_table'][:]
        else:
            # Create a Q-table if it doesn't exist
            self.Q_table = np.zeros(
                (self.bin_size, self.bin_size, len(self.hour_bins), len(self.day_bins), len(self.action_space))
            )
        
        self.q_table_file = q_table_file

    def save_q_table(self):
        """Save the Q-table to an HDF5 file."""
        with h5py.File(self.q_table_file, 'w') as f:
            f.create_dataset('q_table', data=self.Q_table)

    def discretize_state(self, state):
        """Convert continuous state values into discrete bins."""
        storage, price, hour, day = state
        storage_idx = np.digitize(storage, self.storage_bins) - 1
        price_idx = np.digitize(price, self.price_bins) - 1
        hour_idx = int(hour - 1)  # Hours are 1-based
        day_idx = int(day - 1)  # Days are 1-based
        return storage_idx, price_idx, hour_idx, day_idx

    def choose_action(self, state):
        """Choose an action using epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            return np.random.choice(range(len(self.action_space)))
        else:
            storage_idx, price_idx, hour_idx, day_idx = self.discretize_state(state)
            return np.argmax(self.Q_table[storage_idx, price_idx, hour_idx, day_idx])

    def update_Q(self, state, action, reward, next_state, done):
        """Update the Q-table using the Q-learning formula."""
        storage_idx, price_idx, hour_idx, day_idx = self.discretize_state(state)
        next_storage_idx, next_price_idx, next_hour_idx, next_day_idx = self.discretize_state(next_state)

        current_q = self.Q_table[storage_idx, price_idx, hour_idx, day_idx, action]
        if done:
            target = reward
        else:
            next_max_q = np.max(
                self.Q_table[next_storage_idx, next_price_idx, next_hour_idx, next_day_idx]
            )
            target = reward + self.discount_rate * next_max_q

        self.Q_table[storage_idx, price_idx, hour_idx, day_idx, action] += self.learning_rate * (target - current_q)


    def decay_epsilon(self):
        """Decay epsilon to encourage exploitation over time."""
        self.epsilon *= self.epsilon_decay

    def save_q_table_to_csv(self, filename="price_action_q_values.csv"):
        """Save only price, action, and Q-value to a CSV file."""
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["price", "action", "q_value"])
            
            for price_idx in range(self.Q_table.shape[1]):  # Iterate over price bins
                for action_idx, action_value in enumerate(self.action_space):  # Iterate over actions
                    q_value = np.mean(self.Q_table[:, price_idx, :, :, action_idx])  # Average over storage, hour, and day
                    price_value = self.price_bins[price_idx]  # Map bin index to price
                    writer.writerow([price_value, action_value, q_value])


