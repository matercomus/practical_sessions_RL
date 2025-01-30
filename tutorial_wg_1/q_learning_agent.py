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
        discount_rate=0.99,
        learning_rate=0.1,
        epsilon_start=1.0,
        epsilon_end=0.005,  # Keep a small non-zero final epsilon
        daily_demand=120,  # Typically matches the env's daily demand
        model_path: str = None,
    ):
        """
        Simplified Q-Learning Agent:
        - Ignores day/weekend entirely.
        - Uses shortfall/time-left bins as an 'urgency' feature.
        - Now has 5 discrete actions: -1, -0.5, 0, 0.5, +1
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

        # Count total steps for epsilon decay
        # Assume env.price_values has shape (num_days, 24)
        self.total_train_steps = len(env.price_values) * 24
        self.current_step = 0

        # ---------------- PRICE BINS ----------------
        # Use clipped data to reduce effect of extreme outliers
        all_prices = env.price_values.flatten()
        capped_prices = np.clip(all_prices, None, 300)
        price_percentiles = [0, 25, 50, 75, 100]
        self.price_bins = np.percentile(capped_prices, price_percentiles)
        self.n_price_bins = len(self.price_bins) - 1

        # ---------------- STORAGE BINS ----------------
        self.storage_bins = np.linspace(0, 170, 5)  # => [0, 42.5, 85, 127.5, 170]
        self.n_storage_bins = len(self.storage_bins) - 1

        # ---------------- HOUR BINS ----------------
        self.hour_bins = np.arange(1, 25, 6)  # => [1, 7, 13, 19]
        self.n_hour_bins = len(self.hour_bins)

        # ---------------- SHORTFALL/TIME BINS ----------------
        # ratio = (daily_demand - storage) / (24 - hour)
        self.shortfall_time_bins = [0, 2, 5, float("inf")]
        self.n_shortfall_time_bins = len(self.shortfall_time_bins) - 1

        # ---------------- ACTION SPACE (5 actions) ----------------
        self.n_actions = 5
        # e.g. [-1, -0.5, 0, 0.5, +1]
        self.action_space = np.linspace(-1, 1, self.n_actions)

        # ---------------- Q-TABLE ----------------
        # shape = (storage_bins, price_bins, hour_bins, shortfall_time_bins, actions)
        if model_path and os.path.exists(model_path):
            self.load(model_path)
        else:
            self._initialize_q_table()

    def _initialize_q_table(self):
        """Initialize Q-table with zeros."""
        dims = (
            self.n_storage_bins,
            self.n_price_bins,
            self.n_hour_bins,
            self.n_shortfall_time_bins,
            self.n_actions,
        )
        self.Q_table = np.zeros(dims)

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
        """
        Exponential decay with a small final epsilon_end floor.
        """
        k = 5.0 / self.total_train_steps  # tweak "5.0" to control the rate
        self.epsilon = self.epsilon_start * np.exp(-k * self.current_step)
        self.current_step += 1
        # Ensure epsilon never goes below epsilon_end
        if self.epsilon < self.epsilon_end:
            self.epsilon = self.epsilon_end

    def _shortfall_time_left_bin(self, storage: float, hour: float) -> int:
        """Bin ratio = (daily_demand - storage) / max(1, (24 - hour))."""
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
        State from env = (storage, price, hour, day).
        We ignore 'day'.
        """
        storage, price, hour, _ = state

        # storage
        storage_idx = np.digitize(storage, self.storage_bins) - 1
        storage_idx = min(storage_idx, self.n_storage_bins - 1)

        # price
        price_idx = np.digitize(price, self.price_bins) - 1
        price_idx = min(price_idx, self.n_price_bins - 1)

        # hour
        hour_idx = np.digitize(hour, self.hour_bins) - 1
        hour_idx = min(hour_idx, self.n_hour_bins - 1)

        # shortfall/time-left
        st_ratio_idx = self._shortfall_time_left_bin(storage, hour)

        return storage_idx, price_idx, hour_idx, st_ratio_idx
    
    def reward_shaping(self, state, reward):
        """
        Example shaping: penalize big shortfall/time-left ratio
        so the agent won't wait too long before buying.
        Customize thresholds/penalties as desired.
        """
        # Extract ratio directly for convenience:
        storage, _, hour, _ = state
        shortfall = max(0.0, self.daily_demand - storage)
        time_left = max(0.0, 24 - hour)
        if time_left <= 0:
            ratio = float("inf") if shortfall > 0 else 0.0
        else:
            ratio = shortfall / time_left

        shaped_reward = reward

        # Example logic:
        if ratio > 3 and ratio <= 5:
            shaped_reward -= 1.0   # small penalty
        elif ratio > 5:
            shaped_reward -= 2.0   # bigger penalty

        return shaped_reward

    def choose_action(self, state):
        """
        Epsilon-greedy action selection.
        With n_actions=5 => [-1, -0.5, 0, +0.5, +1].
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            s_idx, p_idx, h_idx, st_idx = self.discretize_state(state)
            return np.argmax(self.Q_table[s_idx, p_idx, h_idx, st_idx])

    def update(self, state, action, reward, next_state, done):
        """
        Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]
        """
        s_idx, p_idx, h_idx, st_idx = self.discretize_state(state)
        ns_idx, np_idx, nh_idx, nst_idx = self.discretize_state(next_state)


        # >>> REWARD SHAPING MOD HERE <<<
        shaped_reward = self.reward_shaping(state, reward)
        # >>> END REWARD SHAPING MOD <<<

        current_q = self.Q_table[s_idx, p_idx, h_idx, st_idx, action]

        if done:
            target = reward
        else:
            next_max_q = np.max(self.Q_table[ns_idx, np_idx, nh_idx, nst_idx])
            target = shaped_reward + self.discount_rate * next_max_q

        self.Q_table[s_idx, p_idx, h_idx, st_idx, action] += \
            self.learning_rate * (target - current_q)

        # Decay epsilon after each update
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
                action = self.action_space[action_idx]
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

            # Validation if requested
            if validate_every and val_env and (episode + 1) % validate_every == 0:
                avg_val_reward = self.validate(val_env)
                validation_rewards.append((episode + 1, avg_val_reward))
                print(
                    f"Validation at Episode {episode + 1}: "
                    f"Avg Reward = {avg_val_reward:.2f}"
                )

            env.reset()

        return training_rewards, validation_rewards, state_action_history
    
    def reward_shaping(self, state, reward):
        """
        Example shaping: penalize big shortfall/time-left ratio
        so the agent won't wait too long before buying.
        Customize thresholds/penalties as desired.
        """
        # Extract ratio directly for convenience:
        storage, _, hour, _ = state
        shortfall = max(0.0, self.daily_demand - storage)
        time_left = max(0.0, 24 - hour)
        if time_left <= 0:
            ratio = float("inf") if shortfall > 0 else 0.0
        else:
            ratio = shortfall / time_left

        shaped_reward = reward
        if hour < 8 and storage > 50:
            shaped_reward += 1  # small bonus
        # Example logic:
        if ratio > 3 and ratio <= 5:
            shaped_reward -= 1.0   # small penalty
        elif ratio > 5:
            shaped_reward -= 2.0   # bigger penalty

        return shaped_reward

    def validate(self, env, num_episodes: int = 5) -> float:
        """Evaluate with epsilon=0 (no random exploration)."""
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

    def save_state_action_history(self, history, save_path):
        """Save state-action history in a JSON-friendly format."""
        serializable_history = []
        
        for episode in history:
            episode_data = []
            for state, action in episode:
                # Convert numpy arrays/values to Python native types
                state_list = [float(x) for x in state]
                action_val = float(action)
                episode_data.append([state_list, action_val])
            serializable_history.append(episode_data)
        
        with open(save_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)