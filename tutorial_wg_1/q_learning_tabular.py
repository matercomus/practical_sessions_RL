import gym
import numpy as np
import pandas as pd
from env import DataCenterEnv

class QAgent:
    def __init__(self, env, discount_rate=0.95, learning_rate=0.1, epsilon=1.0, epsilon_decay=0.99, min_epsilon=0.01):
        self.env = env
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Define state and action space discretization
        self.storage_bins = np.linspace(0, 170, 20)  # Discretize storage level (0 to 170 MWh)
        self.price_bins = np.linspace(0, 200, 20)    # Discretize price range (example 0 to 200 €/MWh)
        self.hour_bins = np.arange(1, 25)            # Hour of the day (1 to 24)
        
        self.action_space = np.linspace(-1, 1, 21)   # Discretize actions (-1 to 1)

        # Q-table: (storage_bins, price_bins, hour_bins, action_space)
        self.Qtable = np.zeros((len(self.storage_bins), len(self.price_bins), len(self.hour_bins), len(self.action_space)))

    def discretize_state(self, state):
        storage, price, hour, _ = state
        storage_idx = np.digitize(storage, self.storage_bins) - 1
        price_idx = np.digitize(price, self.price_bins) - 1
        hour_idx = np.digitize(hour, self.hour_bins) - 1
        return storage_idx, price_idx, hour_idx

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(self.action_space))  # Explore
        storage_idx, price_idx, hour_idx = self.discretize_state(state)
        return np.argmax(self.Qtable[storage_idx, price_idx, hour_idx])  # Exploit

    def train(self, episodes):
        for episode in range(episodes):
            state = self.env.observation()
            terminated = False
            total_reward = 0

            while not terminated:
                storage_idx, price_idx, hour_idx = self.discretize_state(state)

                action_idx = self.choose_action(state)
                action = self.action_space[action_idx]

                next_state, reward, terminated = self.env.step(action)

                total_reward += reward

                next_storage_idx, next_price_idx, next_hour_idx = self.discretize_state(next_state)
                best_next_action = np.argmax(self.Qtable[next_storage_idx, next_price_idx, next_hour_idx])

                # Update Q-table
                td_target = reward + self.discount_rate * self.Qtable[next_storage_idx, next_price_idx, next_hour_idx, best_next_action]
                td_error = td_target - self.Qtable[storage_idx, price_idx, hour_idx, action_idx]
                self.Qtable[storage_idx, price_idx, hour_idx, action_idx] += self.learning_rate * td_error

                state = next_state

            # Decay epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    def act(self, state):
        storage_idx, price_idx, hour_idx = self.discretize_state(state)
        action_idx = np.argmax(self.Qtable[storage_idx, price_idx, hour_idx])
        return self.action_space[action_idx]