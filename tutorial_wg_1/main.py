from env import DataCenterEnv
from q_learning_tabular import QLearningAgent
import numpy as np
import argparse
import matplotlib.pyplot as plt

# Argument parser
args = argparse.ArgumentParser()
args.add_argument(
    "--train-path",
    type=str,
    default="./data/train.xlsx",
    help="Path to the training dataset",
)
args.add_argument(
    "--val-path",
    type=str,
    default="./data/validate.xlsx",
    help="Path to the validation dataset",
)
args.add_argument(
    "--episodes", type=int, default=1000, help="Number of training episodes"
)
args.add_argument(
    "--validate-every", type=int, default=50, help="Validate every N episodes"
)
args.add_argument(
    "--make-graphs", action="store_true", help="Flag to enable graph generation"
)
args = args.parse_args()

# Configure numpy print options
np.set_printoptions(suppress=True, precision=2)

# Initialize environment and agent
train_path = args.train_path
val_path = args.val_path
episodes = args.episodes
validate_every = args.validate_every
make_graphs = args.make_graphs

# Create environment and agent
environment = DataCenterEnv(train_path)
agent = QLearningAgent(environment)

# Metrics storage
training_rewards = []
validation_rewards = []


# Function: Validate the agent
def validate_agent(agent, val_path, num_episodes=10):
    """Run validation episodes and return average reward."""
    validation_env = DataCenterEnv(val_path)  # Validation environment
    total_reward = 0
    for _ in range(num_episodes):
        state = validation_env.observation()
        terminated = False
        episode_reward = 0
        while not terminated:
            action_idx = np.argmax(agent.Q_table[agent.discretize_state(state)])
            action = agent.action_space[action_idx]
            next_state, reward, terminated = validation_env.step(action)
            episode_reward += reward
            state = next_state
        total_reward += episode_reward
        validation_env.day = 1  # Reset environment
    return total_reward / num_episodes


# Train the agent
aggregate_reward = 0

for episode in range(episodes):
    state = environment.observation()
    terminated = False
    episode_reward = 0

    while not terminated:
        # Choose an action
        action_idx = agent.choose_action(state)
        action = agent.action_space[action_idx]

        # Take the action in the environment
        next_state, reward, terminated = environment.step(action)
        episode_reward += reward

        # Update the Q-table
        agent.update_Q(state, action_idx, reward, next_state, terminated)

        # Move to the next state
        state = next_state

    # Decay epsilon after each episode
    agent.decay_epsilon()

    # Track training metrics
    aggregate_reward += episode_reward
    training_rewards.append(episode_reward)

    # Debug print for training
    print(
        f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Epsilon = {agent.epsilon:.4f}"
    )

    # Validation
    if (episode + 1) % validate_every == 0:
        avg_validation_reward = validate_agent(agent, val_path)
        validation_rewards.append((episode + 1, avg_validation_reward))
        print(
            f"Validation at Episode {episode + 1}: Avg Reward = {avg_validation_reward:.2f}"
        )

    # Reset environment for the next episode
    environment.day = 1

# Save Q-table at the end
agent.save_q_table()

# Plot training and validation metrics if the flag is set
if make_graphs:
    plt.figure()
    plt.plot(range(1, episodes + 1), training_rewards, label="Training Rewards")
    if validation_rewards:
        validation_episodes, val_rewards = zip(*validation_rewards)
        plt.plot(
            validation_episodes, val_rewards, label="Validation Rewards", marker="o"
        )
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training and Validation Rewards")
    plt.legend()
    plt.grid()
    plt.savefig("training_validation_rewards.png")
    print("Graph saved as 'training_validation_rewards.png'")

print(f"Total reward after {episodes} episodes: {aggregate_reward:.2f}")
