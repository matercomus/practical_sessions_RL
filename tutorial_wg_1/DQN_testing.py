import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Neural network for Q-value approximation.

        Args:
            input_dim: Dimension of state space
            output_dim: Dimension of action space (3 for our case: sell, hold, buy)
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )
        # Xavier initialization for better training dynamics
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.network(x)


class ExperienceReplayBuffer:
    def __init__(self, capacity):
        """
        Experience replay buffer for storing transitions.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = np.zeros(capacity, dtype=object)
        self.capacity = capacity
        self.position = 0
        self.size = 0

    def add(self, transition):
        """Add a transition to the buffer"""
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Sample a random batch of transitions"""
        indices = np.random.randint(0, self.size, size=batch_size)
        batch = self.buffer[indices]
        return zip(*batch)

    def __len__(self):
        return self.size


class DeepQLearningAgent:
    def __init__(
        self,
        env,
        total_episodes,
        discount_rate=0.99,
        learning_rate=0.05,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_episodes=None,
        batch_size=32,
        memory_size=20000,
        update_frequency=4,
        tau=0.04980657167524154,
        model_path=None,
        output_dir=".",
        invalid_action_penalty=-1.0,
    ):
        """
        Initialize the DQN agent with environment and hyperparameters.

        Args:
            env: Training environment
            total_episodes: Number of episodes to train
            discount_rate: Future reward discount factor
            learning_rate: Learning rate for optimizer
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay_episodes: Episodes over which to decay epsilon
            batch_size: Size of training batches
            memory_size: Size of replay buffer
            update_frequency: Steps between target network updates
            tau: Soft update coefficient
            model_path: Path to load pretrained model
            output_dir: Directory for saving outputs
            invalid_action_penalty: Penalty for choosing invalid actions
        """
        self.env = env
        self.total_episodes = total_episodes
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes or int(
            total_episodes * 0.8
        )
        self.epsilon = epsilon_start
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.update_frequency = update_frequency
        self.tau = tau
        self.steps = 0
        self.invalid_action_penalty = invalid_action_penalty

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

        # Environment parameters with proper bounds
        self.daily_energy_demand = 120.0
        self.max_power_rate = 10.0
        self.storage_scale = 170.0  # Maximum storage capacity

        # Set up price scaling based on historical data
        if hasattr(self.env, "price_values"):
            self.price_min = np.min(self.env.price_values)
            self.price_max = np.max(self.env.price_values)
        else:
            self.price_min = 0.0
            self.price_max = 1000.0

        # Setup device and networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize networks
        self.policy_net = DQN(4, 3).to(self.device)
        self.target_net = DQN(4, 3).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Setup optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Initialize replay buffer
        self.memory = ExperienceReplayBuffer(self.memory_size)

        # Save hyperparameters
        self.hparams = {
            "discount_rate": discount_rate,
            "learning_rate": learning_rate,
            "epsilon_start": epsilon_start,
            "epsilon_end": epsilon_end,
            "epsilon_decay_episodes": self.epsilon_decay_episodes,
            "batch_size": batch_size,
            "memory_size": memory_size,
            "update_frequency": update_frequency,
            "tau": tau,
            "invalid_action_penalty": invalid_action_penalty,
        }
        self.hparams_json_path = os.path.join(self.output_dir, "hparams.json")
        self.save_hyperparams()

        # Load model if path provided
        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def detect_forced_action(self, state, chosen_action):
        """
        Detect if an action will be forced by the environment.

        Args:
            state: Current state (storage_level, price, hour, day)
            chosen_action: Action chosen by agent (-1=sell, 0=hold, 1=buy)

        Returns:
            tuple: (executed_action, was_forced, reason)
        """
        storage_level, _, hour, _ = state
        hours_left = 24 - hour
        shortfall = self.daily_energy_demand - storage_level
        max_possible_buy = hours_left * self.max_power_rate

        action = float(np.clip(chosen_action, -1, 1))
        was_forced = False
        reason = None

        if storage_level == 0 and action < 0:
            action = 0.0
            was_forced = True
            reason = "no_storage"

        # Force buy if shortfall can't be met
        if shortfall > max_possible_buy:
            needed_now = shortfall - max_possible_buy
            forced_fraction = min(1.0, needed_now / self.max_power_rate)
            if action < forced_fraction:
                action = forced_fraction
                was_forced = True
                reason = "forced_buy"

        # Prevent selling if it makes meeting demand impossible
        if action < 0:
            sell_mwh = -action * self.max_power_rate
            potential_storage = storage_level - sell_mwh
            potential_shortfall = self.daily_energy_demand - potential_storage
            hours_left_after = hours_left - 1
            max_buy_after = hours_left_after * self.max_power_rate

            if potential_shortfall > max_buy_after:
                action = 0.0
                was_forced = True
                reason = "prevented_sell"

        return float(np.clip(action, -1, 1)), was_forced, reason

    def _normalize_state(self, state):
        """
        Normalize state values to the range [0, 1].

        State components:
        1. storage_level: Bounded by [0, storage_scale]
        2. price: Using min-max normalization based on historical price range
        3. hour: 24-hour format [0, 23]
        4. day: Days of data (1 to total_days)

        Args:
            state: Raw state values (storage_level, price, hour, day)

        Returns:
            numpy.ndarray: Normalized state values, all in range [0, 1]
        """
        storage_level, price, hour, day = state

        # Normalize storage level: Already bounded by [0, storage_scale]
        norm_storage = np.clip(storage_level / self.storage_scale, 0, 1)

        # Normalize price: Use historical min/max for better scaling
        if hasattr(self.env, "price_values"):
            price_min = np.min(self.env.price_values)
            price_max = np.max(self.env.price_values)
            norm_price = np.clip((price - price_min) / (price_max - price_min), 0, 1)
        else:
            # Fallback to simple scaling if no historical prices available
            norm_price = np.clip(price / 1000.0, 0, 1)

        # Normalize hour: [0, 23] -> [0, 1]
        norm_hour = hour / 23.0

        # Normalize day: [1, total_days] -> [0, 1]
        if hasattr(self.env, "test_data"):
            total_days = len(self.env.test_data)
            norm_day = (day - 1) / (total_days - 1)
        else:
            # Fallback if test_data is not available assume 3 years
            norm_day = (day - 1) / (3 * 365)

        return np.array(
            [norm_storage, norm_price, norm_hour, norm_day], dtype=np.float32
        )

    def choose_action(self, state):
        """
        Choose action using epsilon-greedy policy.

        Args:
            state: Current state

        Returns:
            float: Chosen action (-1=sell, 0=hold, 1=buy)
        """
        if np.random.rand() < self.epsilon:
            action_idx = np.random.randint(0, 3)
            return action_idx - 1

        # Ensure state normalization is applied
        normalized_state = self._normalize_state(state)
        state_tensor = (
            torch.tensor(normalized_state, dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            action_idx = q_values.argmax().item()
        return action_idx - 1

    def reward_shaping(self, state, chosen_action, executed_action, reward, next_state):
        """
        Shape the reward by:
          1) Using tanh to keep the final shaped reward in [-1, 1].
          2) Adding a reward for buying between hours 0-7.
        """
        scaled_reward = reward

        # Apply a tanh-based saturating function
        # This will smoothly restrict the final shaped_reward to [-1, 1].
        alpha = 1.0  # scale factor for tanh; adjust as needed
        shaped_reward = float(np.tanh(scaled_reward / alpha) * alpha)

        # Extract hour from the state
        _, _, hour, _ = state

        # Add reward for buying between hours 0-7
        if executed_action > 0 and 0 <= hour <= 7:
            shaped_reward += 1.0

        return shaped_reward, reward

    def _update_network(self):
        """
        Perform one step of training on the Q-network.
        Uses double DQN with soft target updates.
        """
        # Sample batch of transitions
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )

        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # Compute double DQN target
        with torch.no_grad():
            next_q_vals = self.policy_net(next_states)
            next_actions = next_q_vals.argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q_vals = rewards + self.discount_rate * next_q * (~dones)

        # Compute current Q-values
        current_q_vals = (
            self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        )

        # Compute loss and update
        loss_fn = nn.SmoothL1Loss()
        loss = loss_fn(current_q_vals, target_q_vals)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update target network
        with torch.no_grad():
            for tp, pp in zip(
                self.target_net.parameters(), self.policy_net.parameters()
            ):
                tp.data.copy_(self.tau * pp.data + (1.0 - self.tau) * tp.data)

    def update(self, state, chosen_action, executed_action, reward, next_state, done):
        """
        Update agent with a transition, including both chosen and executed actions.
        [Previous docstring content remains the same]
        """
        shaped_r, original_r = self.reward_shaping(
            state, chosen_action, executed_action, reward, next_state
        )

        # Convert executed_action to index (0, 1, 2)
        action_idx = int(executed_action + 1)

        # Ensure both states are normalized
        norm_state = self._normalize_state(state)
        norm_next_state = self._normalize_state(next_state)

        # Store normalized transition
        self.memory.add((norm_state, action_idx, shaped_r, norm_next_state, done))

        if (
            len(self.memory) >= self.batch_size
            and self.steps % self.update_frequency == 0
        ):
            self._update_network()

        self.steps += 1
        return shaped_r, original_r

    def train(self, env, episodes, validate_every=None, val_env=None):
        logger.info(f"Starting training for {episodes} episodes...")
        train_rewards_original = []
        train_rewards_shaped = []
        val_scores = []
        state_action_history = []
        forced_action_stats = {
            "forced_buy": 0,
            "prevented_sell": 0,
            "total_steps": 0,
            "no_storage": 0,
        }

        for ep in range(episodes):
            s = env.observation()
            done = False
            ep_r_shaped = 0.0
            ep_r_original = 0.0
            ep_history = []

            while not done:
                # Choose and execute action
                chosen_action = self.choose_action(s)
                executed_action, was_forced, reason = self.detect_forced_action(
                    s, chosen_action
                )

                # Track statistics
                forced_action_stats["total_steps"] += 1
                if was_forced and reason:
                    forced_action_stats[reason] += 1

                # Environment step
                s_next, r, done = env.step(executed_action)

                # Update agent
                shaped_r, orig_r = self.update(
                    s, chosen_action, executed_action, r, s_next, done
                )

                # Log experience
                ep_history.append(
                    {
                        "state": s,
                        "chosen_action": chosen_action,
                        "executed_action": executed_action,
                        "original_reward": orig_r,
                        "shaped_reward": shaped_r,
                        "was_forced": was_forced,
                        "force_reason": reason,
                    }
                )

                ep_r_shaped += shaped_r
                ep_r_original += orig_r
                s = s_next

            self.save()
            train_rewards_shaped.append(ep_r_shaped)
            train_rewards_original.append(ep_r_original)
            state_action_history.append(ep_history)

            # Periodic validation
            if validate_every and val_env and (ep + 1) % validate_every == 0:
                val_orig, val_shaped, _ = self.validate(val_env)
                val_scores.append((ep + 1, val_orig, val_shaped))
                logger.info(
                    f"Validation at episode {ep+1}: "
                    f"Original Reward: {val_orig:.2f}, "
                    f"Shaped Reward: {val_shaped:.2f}"
                )

            # Epsilon decay
            if ep < self.epsilon_decay_episodes:
                self.epsilon = self.epsilon_start + (
                    ep / self.epsilon_decay_episodes
                ) * (self.epsilon_end - self.epsilon_start)
            else:
                self.epsilon = self.epsilon_end

            # Logging
            forced_rate = (
                forced_action_stats["forced_buy"]
                + forced_action_stats["prevented_sell"]
                + forced_action_stats["no_storage"]
            ) / max(1, forced_action_stats["total_steps"])

            logger.info(
                f"Episode {ep + 1}/{episodes}: "
                f"Original Reward: {ep_r_original:.2f}, "
                f"Shaped Reward: {ep_r_shaped:.2f}, "
                f"Epsilon: {self.epsilon:.4f}, "
                f"Forced Rate: {forced_rate:.2%}"
            )

            env.reset()

        logger.info("Training complete.")

        # Save final rewards
        final_rewards = {
            "final_train_original": (
                train_rewards_original[-1] if train_rewards_original else None
            ),
            "final_train_shaped": (
                train_rewards_shaped[-1] if train_rewards_shaped else None
            ),
            "final_val_original": val_scores[-1][1] if val_scores else None,
            "final_val_shaped": val_scores[-1][2] if val_scores else None,
        }
        with open(os.path.join(self.output_dir, "final_rewards.json"), "w") as f:
            json.dump(final_rewards, f, indent=2)

        return (
            train_rewards_original,
            train_rewards_shaped,
            val_scores,
            state_action_history,
            forced_action_stats,
        )

    def validate(self, env, num_episodes=1):
        logger.info(f"Validating agent for {num_episodes} episodes...")
        total_r_shaped = 0.0
        total_r_original = 0.0
        old_epsilon = self.epsilon
        self.epsilon = 0.0  # No exploration
        validation_history = []  # Collect validation history

        for _ in range(num_episodes):
            s = env.observation()
            done = False
            ep_r_shaped = 0.0
            ep_r_original = 0.0
            ep_history = []

            while not done:
                chosen_action = self.choose_action(s)
                executed_action, was_forced, reason = self.detect_forced_action(
                    s, chosen_action
                )
                s_next, r, done = env.step(executed_action)
                shaped_r, orig_r = self.reward_shaping(
                    s, chosen_action, executed_action, r, s_next
                )

                # Record step details
                ep_history.append(
                    {
                        "state": s,
                        "chosen_action": chosen_action,
                        "executed_action": executed_action,
                        "original_reward": orig_r,
                        "shaped_reward": shaped_r,
                        "was_forced": was_forced,
                        "force_reason": reason,
                    }
                )

                ep_r_shaped += shaped_r
                ep_r_original += orig_r
                s = s_next

            total_r_shaped += ep_r_shaped
            total_r_original += ep_r_original
            validation_history.append(ep_history)
            env.reset()

        self.epsilon = old_epsilon
        avg_original = total_r_original / num_episodes
        avg_shaped = total_r_shaped / num_episodes
        return avg_original, avg_shaped, validation_history

    def save_hyperparams(self):
        """Save hyperparameters to JSON file"""
        with open(self.hparams_json_path, "w") as f:
            json.dump(self.hparams, f, indent=2)
        logger.info(f"Hyperparameters saved to {self.hparams_json_path}")

    def save(self):
        """
        Save model checkpoint including networks, optimizer state, and parameters.

        Args:
            path: Path to save checkpoint
        """
        path = os.path.join(self.output_dir, "model_checkpoint.pth")
        torch.save(
            {
                "policy_net_state_dict": self.policy_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "steps": self.steps,
                "hparams": self.hparams,
            },
            path,
        )
        logger.info(f"Model saved to {path}")

    def load(self, path):
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.steps = checkpoint["steps"]
        if "hparams" in checkpoint:
            self.hparams = checkpoint["hparams"]
        logger.info(f"Model loaded from {path}")

    def save_state_action_history(self, history, filename="state_action_history.json"):
        """Save the state-action history to a JSON file with custom filename"""
        serializable = []
        for ep in history:
            ep_data = []
            for step in ep:
                # Extract values from the step dictionary
                s = step["state"]
                chosen_a = step["chosen_action"]
                executed_a = step["executed_action"]
                orig_r = step["original_reward"]
                shaped_r = step["shaped_reward"]
                forced = step["was_forced"]
                reason = step["force_reason"]

                # Convert state to list
                if isinstance(s, np.ndarray):
                    state_list = s.tolist()
                elif isinstance(s, tuple):
                    state_list = list(s)
                else:
                    state_list = list(s)  # Fallback for other types

                ep_data.append(
                    {
                        "state": state_list,
                        "chosen_action": float(chosen_a),
                        "executed_action": float(executed_a),
                        "original_reward": float(orig_r),
                        "shaped_reward": float(shaped_r),
                        "was_forced": forced,
                        "force_reason": reason,
                    }
                )
            serializable.append(ep_data)

        save_path = os.path.join(self.output_dir, filename)
        with open(save_path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info(f"State-action history saved to {save_path}")
