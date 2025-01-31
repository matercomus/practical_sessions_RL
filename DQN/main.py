from env import DataCenterEnv
from DQN_testing import DeepQLearningAgent
import argparse
import os
import numpy as np
from datetime import datetime
from plotting_utils import plot_metrics
import json


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=str, default="./data/train.xlsx")
    parser.add_argument("--val-path", type=str, default="./data/validate.xlsx")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--validate-every", type=int, default=50)
    parser.add_argument("--load-model", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--run-dir", type=str, default=None, help="Directory for validation results"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run validation only on a saved model",
    )
    args = parser.parse_args()

    if args.validate_only:
        if not args.run_dir:
            raise ValueError("--run-dir required for validation-only mode")

        episodes = 1
        output_dir = args.run_dir
        os.makedirs(output_dir, exist_ok=True)

        model_path = args.load_model
        if not model_path:
            model_path = os.path.join(output_dir, "model_checkpoint.pth")
            if not os.path.exists(model_path):
                raise ValueError(f"Model checkpoint not found at {model_path}")

        val_env = DataCenterEnv(args.val_path)
        agent = DeepQLearningAgent(
            val_env,  # Environment needed for initialization
            total_episodes=episodes,
            model_path=model_path,
            output_dir=output_dir,
        )

        val_original, val_shaped, val_history = agent.validate(
            val_env,
            num_episodes=episodes,
        )

        # Save validation results and history
        val_results = {
            "original_reward": val_original,
            "shaped_reward": val_shaped,
            "num_episodes": episodes,
        }
        val_path = os.path.join(output_dir, "validation_results.json")
        with open(val_path, "w") as f:
            json.dump(val_results, f, indent=2)

        agent.save_state_action_history(
            val_history, "state_action_history_validation.json"
        )

        # Plot validation results
        validation_episodes = [episodes]
        validation_original = [val_original]
        validation_shaped = [val_shaped]

        plot_metrics(
            [],  # Empty training data
            [],
            validation_original,
            validation_shaped,
            validation_episodes,
            output_dir,
        )
        print(f"Validation Results (Average over {episodes} episodes):")
        print(f"Original Reward: {val_original:.2f}")
        print(f"Shaped Reward: {val_shaped:.2f}")
        return

    # Normal training flow
    output_dir = create_output_directory(
        output_dir=args.output_dir, run_name=args.run_name
    )

    train_env = DataCenterEnv(args.train_path)
    val_env = DataCenterEnv(args.val_path) if args.validate_every else None

    agent = DeepQLearningAgent(
        train_env,
        model_path=args.load_model,
        total_episodes=args.episodes,
        output_dir=output_dir,
    )

    if hasattr(agent, "print_and_save_q_table_stats"):
        agent.print_and_save_q_table_stats(output_dir)

    # Modified to receive both reward types
    training_original, training_shaped, validation_scores, state_action_history, _ = (
        agent.train(
            train_env,
            episodes=args.episodes,
            validate_every=args.validate_every,
            val_env=val_env,
        )
    )

    agent.save_state_action_history(state_action_history)

    # Prepare validation data for plotting
    validation_episodes = [score[0] for score in validation_scores]
    validation_original = [score[1] for score in validation_scores]
    validation_shaped = [score[2] for score in validation_scores]

    plot_metrics(
        training_original,
        training_shaped,
        validation_original,
        validation_shaped,
        validation_episodes,
        output_dir,
    )

    print(f"Mean original reward: {np.mean(training_original):.2f}")
    print(f"Mean shaped reward: {np.mean(training_shaped):.2f}")


if __name__ == "__main__":
    main()
