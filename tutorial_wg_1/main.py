from env import DataCenterEnv
from q_learning_agent import QLearningAgent
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
from datetime import datetime


def create_output_directory():
    """Create and return path to timestamped output directory"""
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"run_{run_timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def plot_metrics(training_rewards, validation_rewards, output_dir):
    """Plot and save training/validation metrics"""
    plt.figure()
    plt.plot(
        range(1, len(training_rewards) + 1), training_rewards, label="Training Rewards"
    )
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
    graph_path = os.path.join(output_dir, "training_validation_rewards.png")
    plt.savefig(graph_path)
    print(f"Graph saved at {graph_path}")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=str, default="./data/train.xlsx")
    parser.add_argument("--val-path", type=str, default="./data/validate.xlsx")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--validate-every", type=int, default=50)
    parser.add_argument("--make-graphs", action="store_true")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--load-model", type=str, default=None)
    args = parser.parse_args()

    # Create output directory
    output_dir = create_output_directory()

    # Initialize environments
    train_env = DataCenterEnv(args.train_path)
    val_env = DataCenterEnv(args.val_path) if args.validate_every else None

    # Initialize agent
    agent = QLearningAgent(train_env, model_path=args.load_model)

    # Train agent
    training_rewards, validation_rewards = agent.train(
        train_env,
        episodes=args.episodes,
        validate_every=args.validate_every,
        val_env=val_env,
    )

    # Save model if requested
    if args.save_model:
        model_path = os.path.join(output_dir, "model.h5")
        agent.save(model_path)
        print(f"Model saved at {model_path}")

    # Plot metrics if requested
    if args.make_graphs:
        plot_metrics(training_rewards, validation_rewards, output_dir)

    # Print final metrics
    print(f"Total reward after {args.episodes} episodes: {sum(training_rewards):.2f}")


if __name__ == "__main__":
    main()
