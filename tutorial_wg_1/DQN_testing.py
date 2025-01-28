import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import json
import logging

from agent_base import BaseAgent

# Configure logging - INFO level should be enough for typical training monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        # Simple feedforward network with Xavier initialization
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

        # Initialize weights using Xavier initialization
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.network(x)


class ExperienceReplayBuffer:
    def __init__(self, capacity):
        self.buffer = np.zeros(capacity, dtype=object)  # Pre-allocate buffer
        self.capacity = capacity
        self.position = 0
        self.size = 0

    def add(self, transition):
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.choice(self.size, batch_size, replace=False)
        batch = self.buffer[indices]
        return zip(*batch)

    def __len__(self):
        return self.size


class DeepQLearningAgent(BaseAgent):
    def __init__(
        self,
        env,
        discount_rate=0.99,  # Higher discount factor for longer-term
        learning_rate=0.0001,  # Lower LR for stability
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_episodes=300,  # Slow epsilon decay over episodes
        batch_size=128,
        memory_size=20000,
        update_frequency=4,
        tau=0.01,  # Polyak update rate for target net
        model_path=None,
    ):
        super().__init__()
        self.env = env
        self.discount_rate = discount_rate
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.epsilon = epsilon_start
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.tau = tau

        # Scale parameters (adjust to match your environment)
        self.daily_energy_demand = 120.0
        self.max_power_rate = 10.0
        self.storage_scale = 170.0
        self.price_scale = np.percentile(self.env.price_values.flatten(), 99)

        # Setup networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(4, 3).to(self.device)
        self.target_net = DQN(4, 3).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # We never train target_net directly

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8
        )

        # Replay Buffer
        self.memory = ExperienceReplayBuffer(memory_size)

        # Bookkeeping
        self.steps = 0

        # Optionally load a previously saved model
        if model_path and os.path.exists(model_path):
            self.load(model_path)

        logger.info(f"DeepQLearningAgent initialized on device: {self.device}")

    def _normalize_state(self, state):
        """
        Vectorized state normalization.
        State is assumed: (storage, price, hour, day).
        """
        return np.array(
            [
                state[0] / self.storage_scale,  # normalize storage
                state[1] / self.price_scale,  # normalize price
                state[2] / 24.0,  # hour in [0..24)
                state[3] / 7.0,  # day in [0..7) or adapt to your env
            ],
            dtype=np.float32,
        )

    def choose_action(self, state):
        """
        Epsilon-greedy action selection (actions: -1, 0, +1).
        """
        if random.random() < self.epsilon:
            # Random integer in [-1, 1]
            return random.randint(-1, 1)

        # Greedy action
        with torch.no_grad():
            state_tensor = (
                torch.FloatTensor(self._normalize_state(state))
                .unsqueeze(0)
                .to(self.device)
            )
            q_values = self.policy_net(state_tensor)  # shape [1, 3]
            action_idx = q_values.argmax(dim=1).item()  # 0,1,2
            action = action_idx - 1  # -1..1
        return action

    def reward_shaping(self, state, action, reward, next_state):
        """
        Minimal shaping: mostly rely on environment reward,
        optionally add small bonus at day boundary if storage meets demand.
        """
        shaped_reward = reward
        storage, price, hour, day = state
        next_hour = int(next_state[2])

        # If environment transitions from hour=23->hour=0 as new day
        if hour == 23 and next_hour == 0:
            if storage >= self.daily_energy_demand:
                # small bonus for meeting or exceeding daily demand
                shaped_reward += 1.0

        return shaped_reward, reward

    def update(self, state, action, reward, next_state, done):
        """
        Add transition to replay, do partial training step if needed.
        """
        shaped_reward, original_reward = self.reward_shaping(
            state, action, reward, next_state
        )
        action_idx = action + 1  # -1..1 => 0..2

        # Store transitions (normalized states)
        self.memory.add(
            (
                self._normalize_state(state),
                action_idx,
                shaped_reward,
                self._normalize_state(next_state),
                done,
            )
        )

        # Train if enough samples and at certain frequencies
        if len(self.memory) >= self.batch_size and (
            self.steps % self.update_frequency == 0
        ):
            self._update_network()

        self.steps += 1
        return shaped_reward, original_reward

    def _update_network(self):
        """
        One gradient update step using Double DQN + Huber loss
        and then Polyak (soft) update the target network.
        """
        # Sample from replay
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )

        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # Double DQN: choose best action via policy_net, evaluate via target_net
        with torch.no_grad():
            next_q_values = self.policy_net(next_states)  # shape [B, 3]
            next_actions = next_q_values.argmax(dim=1, keepdim=True)  # shape [B, 1]
            target_q_values_next = (
                self.target_net(next_states).gather(1, next_actions).squeeze(1)
            )

            # Bellman target
            target_q_values = rewards + self.discount_rate * target_q_values_next * (
                ~dones
            )

        # Current Q-values
        current_q_values = (
            self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        )

        # Use Huber (SmoothL1) loss
        loss_fn = nn.SmoothL1Loss()
        loss = loss_fn(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Polyak (soft) update for the target network
        with torch.no_grad():
            for target_param, policy_param in zip(
                self.target_net.parameters(), self.policy_net.parameters()
            ):
                target_param.data.copy_(
                    self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
                )

    def train(self, env, episodes, validate_every=None, val_env=None):
        """
        Main training loop with additional logging.
        Logs every episode’s reward and epsilon.
        """
        logger.info(f"Starting training for {episodes} episodes...")
        training_rewards = []
        validation_rewards = []
        state_action_history = []

        for episode in range(episodes):
            state = env.observation()
            terminated = False
            episode_reward = 0.0
            episode_history = []

            while not terminated:
                action = self.choose_action(state)
                next_state, reward, terminated = env.step(action)
                shaped_reward, original_reward = self.update(
                    state, action, reward, next_state, terminated
                )
                episode_history.append((state, action, original_reward, shaped_reward))
                episode_reward += shaped_reward
                state = next_state

            # Collect episode stats
            training_rewards.append(episode_reward)
            state_action_history.append(episode_history)

            # Validation if requested
            if validate_every and val_env and (episode + 1) % validate_every == 0:
                val_reward = self.validate(val_env)
                validation_rewards.append((episode + 1, val_reward))

            # Environment reset at end of episode
            env.reset()

            # Linear epsilon decay
            if episode < self.epsilon_decay_episodes:
                fraction = episode / float(self.epsilon_decay_episodes)
                self.epsilon = self.epsilon_start + fraction * (
                    self.epsilon_end - self.epsilon_start
                )
            else:
                self.epsilon = self.epsilon_end

            # Log info EVERY episode
            logger.info(
                f"Episode {episode + 1}/{episodes}: "
                f"Reward={episode_reward:.2f}, "
                f"Epsilon={self.epsilon:.4f}, "
                f"TotalSteps={self.steps}, "
                f"BufferSize={len(self.memory)}"
            )

        logger.info("Training complete.")
        return training_rewards, validation_rewards, state_action_history

    def validate(self, env, num_episodes=10):
        """
        Evaluate current policy with epsilon=0 (greedy).
        Returns average reward across num_episodes.
        """
        total_reward = 0.0
        original_epsilon = self.epsilon
        self.epsilon = 0.0

        for _ in range(num_episodes):
            state = env.observation()
            terminated = False
            episode_reward = 0.0

            while not terminated:
                action = self.choose_action(state)
                next_state, reward, terminated = env.step(action)
                shaped_reward, _ = self.reward_shaping(
                    state, action, reward, next_state
                )
                episode_reward += shaped_reward
                state = next_state

            total_reward += episode_reward
            env.reset()

        self.epsilon = original_epsilon
        avg_reward = total_reward / num_episodes
        logger.info(
            f"Validation over {num_episodes} episodes: Average Reward={avg_reward:.2f}"
        )
        return avg_reward

    def save(self, path):
        """
        Saves model, optimizer, and relevant parameters to a checkpoint.
        """
        torch.save(
            {
                "policy_net_state_dict": self.policy_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "steps": self.steps,
            },
            path,
        )
        logger.info(f"Model saved to {path}")

    def load(self, path):
        """
        Loads model, optimizer, and relevant parameters from a checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.steps = checkpoint["steps"]
        logger.info(f"Model loaded from {path}")

    def save_state_action_history(self, state_action_history, save_path):
        """
        Converts training history to a JSON-serializable format and saves to file.
        """
        serializable_history = []
        for episode in state_action_history:
            episode_data = []
            for state, action, original_reward, shaped_reward in episode:
                # Convert any numpy arrays in `state` to lists for JSON
                episode_data.append(
                    (state.tolist(), action, original_reward, shaped_reward)
                )
            serializable_history.append(episode_data)

        with open(save_path, "w") as f:
            json.dump(serializable_history, f)
        logger.info(f"State-action history saved to {save_path}")
