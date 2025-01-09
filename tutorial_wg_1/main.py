from env import DataCenterEnv
from q_learning_tabular import QLearningAgent
import numpy as np
import argparse
from training_monitor import TrainingMonitor
from validation_utils import validate_agent

def main():
    args = argparse.ArgumentParser()
    args.add_argument('--train_path', type=str, default='train.xlsx')
    args.add_argument('--val_path', type=str, default='validate.xlsx')
    args.add_argument('--episodes', type=int, default=1000)
    args.add_argument('--validate_every', type=int, default=50)
    args = args.parse_args()

    np.set_printoptions(suppress=True, precision=2)
    train_path = args.train_path
    val_path = args.val_path
    episodes = args.episodes

    # Initialize environments, agent, and training monitor
    train_environment = DataCenterEnv(train_path)
    val_environment = DataCenterEnv(val_path)
    agent = QLearningAgent(train_environment)
    monitor = TrainingMonitor()

    # Train the agent
    aggregate_reward = 0
    for episode in range(episodes):
        state = train_environment.observation()
        terminated = False
        episode_reward = 0

        while not terminated:
            # Choose an action using the Q-learning agent
            action_idx = agent.choose_action(state)
            action = agent.action_space[action_idx]

            # Take the action in the environment
            next_state, reward, terminated = train_environment.step(action)
            episode_reward += reward

            # Update the Q-table
            agent.update_Q(state, action_idx, reward, next_state, terminated)

            # Move to the next state
            state = next_state

        # Decay epsilon after each episode
        agent.decay_epsilon()

        # Update training monitor
        monitor.update(episode_reward, agent.epsilon)
        aggregate_reward += episode_reward

        # Run validation episodes periodically
        if (episode + 1) % args.validate_every == 0:
            validation_reward = validate_agent(val_environment, agent)
            monitor.add_validation_reward(validation_reward)
            print(f"Validation reward at episode {episode + 1}: {validation_reward:.2f}")
            
            # Plot progress
            monitor.plot_training_progress()

        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Epsilon = {agent.epsilon:.4f}")

        # Save Q-table at regular intervals
        if (episode + 1) % 10 == 0:
            agent.save_q_table()
            print(f"Q-table saved at episode {episode + 1}")
            print("--------------------------------------------------")
            print(agent.Q_table[:5, :5, :5, :5, :5])
            print("--------------------------------------------------")

        train_environment.day = 1
        val_environment.day = 1
        print(f"Resetting day to 1 at episode {episode + 1}")

    print(f'Total reward after {episodes} episodes: {aggregate_reward:.2f}')
    
    # Final plots
    monitor.plot_training_progress()

if __name__ == "__main__":
    main()
