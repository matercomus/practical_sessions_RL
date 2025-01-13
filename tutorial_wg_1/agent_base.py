from abc import ABC, abstractmethod
import os
from typing import Any, Tuple

class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    @abstractmethod
    def train(self, env, episodes: int, validate_every: int = None, val_env = None) -> Tuple[list, list]:
        """
        Train the agent on the environment
        
        Args:
            env: Training environment
            episodes: Number of episodes to train
            validate_every: Optional; Validate every N episodes
            val_env: Optional; Validation environment
            
        Returns:
            Tuple of (training_rewards, validation_rewards)
        """
        pass
    
    @abstractmethod
    def choose_action(self, state: Any) -> Any:
        """Choose an action based on the current state"""
        pass
    
    @abstractmethod
    def update(self, state: Any, action: Any, reward: float, next_state: Any, done: bool):
        """Update the agent's knowledge based on the transition"""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save the agent's state to the specified path"""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load the agent's state from the specified path"""
        pass
    
    @abstractmethod
    def validate(self, env, num_episodes: int = 10) -> float:
        """
        Run validation episodes and return average reward
        
        Args:
            env: Validation environment
            num_episodes: Number of validation episodes to run
            
        Returns:
            Average reward over validation episodes
        """
        pass
