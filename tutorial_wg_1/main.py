from env import DataCenterEnv
from q_learning_tabular import QLearningAgent
import numpy as np
import argparse

args = argparse.ArgumentParser()
args.add_argument('--path', type=str, default='train.xlsx')
args.add_argument('--episodes', type=int, default=1000)
args = args.parse_args()

np.set_printoptions(suppress=True, precision=2)
path_to_dataset = args.path
episodes = args.episodes

# Initialize environment and agent
environment = DataCenterEnv(path_to_dataset)
agent = QLearningAgent(environment)

# Train the agent
aggregate_reward = 0

agent.save_q_table_to_csv("q_table.csv")

for episode in range(episodes):
    state = environment.observation()
    terminated = False
    episode_reward = 0

    while not terminated:
        # Choose an action using the Q-learning agent
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

    aggregate_reward += episode_reward
    print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Epsilon = {agent.epsilon:.4f}")

    # Save Q-table at the end of the episode
    if (episode + 1) % 100 == 0:
        agent.save_q_table()

    environment.day = 1
    print(f"Resetting day to 1 at episode {episode + 1}")


print(f'Total reward after {episodes} episodes: {aggregate_reward:.2f}')
