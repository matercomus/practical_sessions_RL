from env import DataCenterEnv
from DQN_testing import DeepQLearningAgent
import argparse
import os
import numpy as np
from datetime import datetime
from plotting_utils import plot_metrics


def create_output_directory(output_dir=".", run_name=None):
    """Create and return path to timestamped output directory"""
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_name = f"run_{run_timestamp}"
    if run_name:
        dir_name += f"_{run_name}"
    output_dir = os.path.join(output_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=str, default="./data/train.xlsx")
    parser.add_argument("--val-path", type=str, default="./data/validate.xlsx")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--validate-every", type=int, default=50)
    parser.add_argument("--load-model", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    # Create output directory
    output_dir = create_output_directory(
        output_dir=args.output_dir, run_name=args.run_name
    )

    # Initialize environments
    train_env = DataCenterEnv(args.train_path)
    val_env = DataCenterEnv(args.val_path) if args.validate_every else None

    # Initialize agent
    agent = DeepQLearningAgent(
        train_env,
        model_path=args.load_model,
        total_episodes=args.episodes,
        output_dir=output_dir,
    )

    # Print and save q table stats if agent has this method
    if hasattr(agent, "print_and_save_q_table_stats"):
        agent.print_and_save_q_table_stats(output_dir)

    # Train agent
    training_rewards, validation_rewards, state_action_history, _ = agent.train(
        train_env,
        episodes=args.episodes,
        validate_every=args.validate_every,
        val_env=val_env,
    )

    agent.save_state_action_history(state_action_history)
    plot_metrics(training_rewards, validation_rewards, output_dir)

    # Print final metrics
    print(
        f"Mean reward after {args.episodes} episodes: {np.mean(training_rewards):.2f}"
    )


if __name__ == "__main__":
    main()
