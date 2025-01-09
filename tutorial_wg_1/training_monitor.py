import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

class TrainingMonitor:
    def __init__(self, save_dir='training_plots'):
        self.save_dir = save_dir
        self.episode_rewards = []
        self.validation_rewards = []
        self.epsilon_history = []
        self.moving_avg_window = 100
        
        # Create directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
    def update(self, episode_reward, epsilon):
        self.episode_rewards.append(episode_reward)
        self.epsilon_history.append(epsilon)
        
    def add_validation_reward(self, reward):
        self.validation_rewards.append(reward)
        
    def plot_training_progress(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
        
        # Plot episode rewards
        ax1.plot(self.episode_rewards, alpha=0.3, label='Episode Reward')
        if len(self.episode_rewards) >= self.moving_avg_window:
            moving_avg = np.convolve(self.episode_rewards, 
                                   np.ones(self.moving_avg_window)/self.moving_avg_window, 
                                   mode='valid')
            ax1.plot(range(self.moving_avg_window-1, len(self.episode_rewards)), 
                    moving_avg, label=f'{self.moving_avg_window}-Episode Moving Average')
        ax1.set_title('Training Rewards per Episode')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.legend()
        ax1.grid(True)
        
        # Plot validation rewards
        if self.validation_rewards:
            ax2.plot(self.validation_rewards, marker='o', label='Validation Reward')
            ax2.set_title('Validation Rewards')
            ax2.set_xlabel('Validation Episode')
            ax2.set_ylabel('Reward')
            ax2.legend()
            ax2.grid(True)
        
        # Plot epsilon decay
        ax3.plot(self.epsilon_history, label='Epsilon')
        ax3.set_title('Epsilon Decay')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Epsilon')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/training_progress_{timestamp}.png')
        plt.close()
