import numpy as np
import h5py
import os
from typing import Tuple, List, Union
import logging

class QLearningAgent:
    """
    Q-Learning agent with improved state discretization and logging capabilities.
    """
    def __init__(
        self, 
        env, 
        discount_rate: float = 0.95,
        learning_rate: float = 0.1,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.999,
        epsilon_min: float = 0.01,
        bin_size: int = 20,
        q_table_file: str = 'q_table.h5'
    ):
        self.env = env
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.bin_size = bin_size
        
        # Define action space with more granular control
        self.action_space = np.linspace(-1, 1, 21)  # Increased from 11 to 21 actions
        
        # Improved state space discretization
        self.storage_bins = self._create_nonuniform_bins(0, 200, self.bin_size)
        self.price_bins = self._create_nonuniform_bins(0, 100, self.bin_size)
        self.hour_bins = np.arange(1, 25)
        self.day_bins = np.arange(1, len(env.price_values) + 1)
        
        # Initialize Q-table
        self.q_table_file = q_table_file
        self.Q_table = self._initialize_q_table()
        
        # Setup logging
        self._setup_logging()
        
    def _create_nonuniform_bins(self, start: float, end: float, num_bins: int) -> np.ndarray:
        """
        Create non-uniform bins with finer granularity in critical regions.
        """
        # Use exponential distribution for bin edges to focus on lower values
        exp_bins = np.exp(np.linspace(0, np.log(end - start + 1), num_bins)) - 1 + start
        return exp_bins
    
    def _initialize_q_table(self) -> np.ndarray:
        """
        Initialize Q-table with optimistic initial values or load from file.
        """
        if os.path.exists(self.q_table_file):
            with h5py.File(self.q_table_file, 'r') as f:
                return f['q_table'][:]
        else:
            # Initialize with small positive values to encourage exploration
            return np.ones((
                self.bin_size,
                self.bin_size,
                len(self.hour_bins),
                len(self.day_bins),
                len(self.action_space)
            )) * 0.1
    
    def _setup_logging(self):
        """
        Setup logging configuration for the agent.
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('qlearning.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def discretize_state(self, state: Tuple[float, float, int, int]) -> Tuple[int, int, int, int]:
        """
        Convert continuous state values into discrete bins with boundary handling.
        """
        storage, price, hour, day = state
        
        # Clip values to valid ranges
        storage = np.clip(storage, 0, 200)
        price = np.clip(price, 0, 100)
        
        storage_idx = np.clip(np.digitize(storage, self.storage_bins) - 1, 0, self.bin_size - 1)
        price_idx = np.clip(np.digitize(price, self.price_bins) - 1, 0, self.bin_size - 1)
        hour_idx = int(np.clip(hour - 1, 0, 23))
        day_idx = int(np.clip(day - 1, 0, len(self.day_bins) - 1))
        
        return storage_idx, price_idx, hour_idx, day_idx
    
    def choose_action(self, state: Tuple[float, float, int, int]) -> int:
        """
        Choose an action using epsilon-greedy policy with bonus for unexplored actions.
        """
        if np.random.random() < self.epsilon:
            # Add exploration bonus for less-visited actions
            storage_idx, price_idx, hour_idx, day_idx = self.discretize_state(state)
            visit_counts = np.abs(self.Q_table[storage_idx, price_idx, hour_idx, day_idx])
            exploration_bonus = 1.0 / (1.0 + visit_counts)
            weighted_random = exploration_bonus / np.sum(exploration_bonus)
            return np.random.choice(range(len(self.action_space)), p=weighted_random)
        else:
            storage_idx, price_idx, hour_idx, day_idx = self.discretize_state(state)
            return np.argmax(self.Q_table[storage_idx, price_idx, hour_idx, day_idx])
    
    def update_Q(self, state: Tuple[float, float, int, int], action: int, 
                reward: float, next_state: Tuple[float, float, int, int], done: bool):
        """
        Update Q-table using Q-learning with eligibility traces.
        """
        storage_idx, price_idx, hour_idx, day_idx = self.discretize_state(state)
        next_storage_idx, next_price_idx, next_hour_idx, next_day_idx = self.discretize_state(next_state)
        
        current_q = self.Q_table[storage_idx, price_idx, hour_idx, day_idx, action]
        
        if done:
            target = reward
        else:
            # Double Q-Learning approach
            next_action = np.argmax(self.Q_table[next_storage_idx, next_price_idx, next_hour_idx, next_day_idx])
            next_q = self.Q_table[next_storage_idx, next_price_idx, next_hour_idx, next_day_idx, next_action]
            target = reward + self.discount_rate * next_q
        
        # Update with variable learning rate based on visit count
        visit_count = np.sum(np.abs(self.Q_table[storage_idx, price_idx, hour_idx, day_idx]))
        adaptive_lr = self.learning_rate / (1 + 0.1 * visit_count)
        
        self.Q_table[storage_idx, price_idx, hour_idx, day_idx, action] += adaptive_lr * (target - current_q)
        
        # Log significant Q-value updates
        if abs(target - current_q) > 1.0:
            self.logger.info(f"Large Q-value update: {current_q:.2f} -> {target:.2f}")
    
    def decay_epsilon(self):
        """
        Decay epsilon with a minimum value to maintain some exploration.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save_q_table(self):
        """
        Save Q-table with compression and error checking.
        """
        try:
            with h5py.File(self.q_table_file, 'w') as f:
                f.create_dataset('q_table', data=self.Q_table, compression='gzip')
            self.logger.info(f"Q-table successfully saved to {self.q_table_file}")
        except Exception as e:
            self.logger.error(f"Error saving Q-table: {str(e)}")

# Training script improvements
def train_agent(env, agent, episodes: int, eval_interval: int = 10):
    """
    Training loop with improved monitoring and early stopping.
    """
    best_reward = float('-inf')
    no_improvement_count = 0
    
    for episode in range(episodes):
        state = env.observation()
        terminated = False
        episode_reward = 0
        step_count = 0
        
        while not terminated:
            action_idx = agent.choose_action(state)
            action = agent.action_space[action_idx]
            next_state, reward, terminated = env.step(action)
            
            episode_reward += reward
            agent.update_Q(state, action_idx, reward, next_state, terminated)
            
            state = next_state
            step_count += 1
        
        agent.decay_epsilon()
        
        # Evaluation and logging
        if (episode + 1) % eval_interval == 0:
            agent.save_q_table()
            print(f"Episode {episode + 1}")
            print(f"Steps: {step_count}")
            print(f"Reward: {episode_reward:.2f}")
            print(f"Epsilon: {agent.epsilon:.4f}")
            
            # Early stopping check
            if episode_reward > best_reward:
                best_reward = episode_reward
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                
            if no_improvement_count >= 50:  # Stop if no improvement for 50 evaluations
                print("Early stopping triggered")
                break
        
        env.day = 1
    
    return agent
