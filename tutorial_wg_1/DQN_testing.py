import os
import json
import time
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
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
        # Xavier initialization
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.network(x)


class ExperienceReplayBuffer:
    def __init__(self, capacity):
        self.buffer = np.zeros(capacity, dtype=object)
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


class DeepQLearningAgent:
    # Best hyperparams found
    # {'discount_rate': 0.9, 'learning_rate': 0.0006710468773372056, 'tau': 0.04980657167524154, 'batch_size': 32}
    def __init__(
        self,
        env,
        total_episodes,
        discount_rate=0.9,
        learning_rate=0.0006710468773372056,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_episodes=None,
        batch_size=32,
        memory_size=20000,
        update_frequency=4,
        tau=0.04980657167524154,  # Polyak update rate
        model_path=None,
        output_dir=".",
    ):
        """
        :param env: Training environment
        :param total_episodes: Total number of training episodes
        :param discount_rate: Gamma
        :param learning_rate: LR for Adam
        :param epsilon_start: Initial epsilon
        :param epsilon_end: Final epsilon
        :param epsilon_decay_episodes: # episodes to linearly decay epsilon
        :param batch_size: Training batch size
        :param memory_size: Replay buffer capacity
        :param update_frequency: Steps between training updates
        :param tau: Polyak update coefficient
        :param model_path: Path to load a model checkpoint
        :param output_dir: Directory to save logs, hyperparams
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

        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

        # Environment scaling parameters (adapt to your environment)
        self.daily_energy_demand = 120.0
        self.max_power_rate = 10.0
        self.storage_scale = 170.0
        # We'll take a percentile from env for price scaling if env has price_values
        if hasattr(self.env, "price_values"):
            self.price_scale = np.percentile(self.env.price_values.flatten(), 99)
        else:
            self.price_scale = 100.0  # Fallback

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # DQN networks
        self.policy_net = DQN(4, 3).to(self.device)
        self.target_net = DQN(4, 3).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Replay Buffer
        self.memory = ExperienceReplayBuffer(self.memory_size)

        # Save hyperparams JSON immediately
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
        }
        ts = time.strftime("%Y%m%d-%H%M%S")
        self.hparams_json_path = os.path.join(self.output_dir, f"hparams_{ts}.json")
        self.save_hyperparams()

        # Optional: load existing model
        if model_path and os.path.exists(model_path):
            self.load(model_path)

        logger.info("DeepQLearningAgent initialized.")

    def save_hyperparams(self):
        with open(self.hparams_json_path, "w") as f:
            json.dump(self.hparams, f, indent=2)
        logger.info(f"Hyperparameters saved to {self.hparams_json_path}")

    def _normalize_state(self, state):
        """
        Expects state in form (storage, price, hour, day).
        Adjust as needed for your environment.
        """
        return np.array(
            [
                state[0] / self.storage_scale,
                state[1] / self.price_scale,
                state[2] / 24.0,
                state[3] / 7.0,
            ],
            dtype=np.float32,
        )

    def choose_action(self, state):
        """
        Epsilon-greedy policy. Output in {-1, 0, +1}.
        """
        if random.random() < self.epsilon:
            return random.randint(-1, 1)
        with torch.no_grad():
            s = (
                torch.FloatTensor(self._normalize_state(state))
                .unsqueeze(0)
                .to(self.device)
            )
            q_values = self.policy_net(s)
            a_idx = q_values.argmax(dim=1).item()
            # map 0->-1, 1->0, 2->1
            return a_idx - 1

    def reward_shaping(self, state, action, reward, next_state):
        """
        Simple shaping: If hour=23 -> next_hour=0, bonus if storage >= daily_energy_demand.
        """
        shaped_reward = reward
        storage, price, hour, day = state
        next_hour = int(next_state[2])

        if hour == 23 and next_hour == 0:
            if storage >= self.daily_energy_demand:
                shaped_reward += 1.0

        return shaped_reward, reward

    def update(self, state, action, reward, next_state, done):
        """
        Store transition, train periodically.
        """
        shaped_r, original_r = self.reward_shaping(state, action, reward, next_state)
        a_idx = action + 1  # map -1..1 -> 0..2

        self.memory.add(
            (
                self._normalize_state(state),
                a_idx,
                shaped_r,
                self._normalize_state(next_state),
                done,
            )
        )

        if len(self.memory) >= self.batch_size and (
            self.steps % self.update_frequency == 0
        ):
            self._update_network()

        self.steps += 1
        return shaped_r, original_r

    def _update_network(self):
        """
        Double DQN with Huber loss + Polyak (soft) target update.
        """
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        with torch.no_grad():
            next_q_vals = self.policy_net(next_states)
            next_acts = next_q_vals.argmax(dim=1, keepdim=True)
            target_next_q = self.target_net(next_states).gather(1, next_acts).squeeze(1)
            target_q_vals = rewards + self.discount_rate * target_next_q * (~dones)

        current_q_vals = (
            self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        )

        loss_fn = nn.SmoothL1Loss()
        loss = loss_fn(current_q_vals, target_q_vals)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Polyak update
        with torch.no_grad():
            for tp, pp in zip(
                self.target_net.parameters(), self.policy_net.parameters()
            ):
                tp.data.copy_(self.tau * pp.data + (1.0 - self.tau) * tp.data)

    def train(self, env, episodes, validate_every=None, val_env=None):
        logger.info(f"Starting training for {episodes} episodes...")
        train_rewards = []
        val_scores = []
        state_action_history = []

        for ep in range(episodes):
            s = env.observation()
            done = False
            ep_r = 0.0
            ep_history = []

            while not done:
                a = self.choose_action(s)
                s_next, r, done = env.step(a)
                shaped_r, orig_r = self.update(s, a, r, s_next, done)
                ep_history.append((s, a, orig_r, shaped_r))
                ep_r += shaped_r
                s = s_next

            train_rewards.append(ep_r)
            state_action_history.append(ep_history)

            if validate_every and val_env and ((ep + 1) % validate_every == 0):
                val_r = self.validate(val_env)
                val_scores.append((ep + 1, val_r))

            # Epsilon decay
            if ep < self.epsilon_decay_episodes:
                frac = ep / float(self.epsilon_decay_episodes)
                self.epsilon = self.epsilon_start + frac * (
                    self.epsilon_end - self.epsilon_start
                )
            else:
                self.epsilon = self.epsilon_end

            logger.info(
                f"Episode {ep + 1}/{episodes}: "
                f"Reward={ep_r:.2f}, "
                f"Epsilon={self.epsilon:.4f}, "
                f"Steps={self.steps}, "
                f"BufferSize={len(self.memory)}"
            )

            env.reset()

        logger.info("Training complete.")
        return train_rewards, val_scores, state_action_history

    def validate(self, env, num_episodes=5):
        total_r = 0.0
        old_epsilon = self.epsilon
        self.epsilon = 0.0

        for _ in range(num_episodes):
            s = env.observation()
            done = False
            ep_r = 0.0
            while not done:
                a = self.choose_action(s)
                s_next, r, done = env.step(a)
                shaped_r, _ = self.reward_shaping(s, a, r, s_next)
                ep_r += shaped_r
                s = s_next
            total_r += ep_r
            env.reset()

        self.epsilon = old_epsilon
        avg_r = total_r / num_episodes
        logger.info(f"Validation over {num_episodes} episodes: Avg Reward={avg_r:.2f}")
        return avg_r

    def save(self, path):
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
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.steps = checkpoint["steps"]
        if "hparams" in checkpoint:
            self.hparams = checkpoint["hparams"]
        logger.info(f"Model loaded from {path}")

    def save_state_action_history(self, history, save_path):
        serializable = []
        for ep in history:
            ep_data = []
            for s, a, or_r, sh_r in ep:
                ep_data.append((s.tolist(), a, or_r, sh_r))
            serializable.append(ep_data)

        with open(save_path, "w") as f:
            json.dump(serializable, f)
        logger.info(f"State-action history saved to {save_path}")
