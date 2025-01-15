from env import DataCenterEnv
from q_learning_agent import QLearningAgent
import argparse
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta


def create_output_directory():
    """Create and return path to timestamped output directory"""
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"run_{run_timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def plot_metrics(training_rewards, validation_rewards, output_dir):
    """Plot and save training/validation metrics"""
    # Plot training rewards
    plt.figure()
    plt.plot(
        range(1, len(training_rewards) + 1), training_rewards, label="Training Rewards"
    )
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Rewards")
    plt.legend()
    plt.grid()
    training_graph_path = os.path.join(output_dir, "training_rewards.png")
    plt.savefig(training_graph_path)
    print(f"Training graph saved at {training_graph_path}")

    # Plot validation rewards if available
    if validation_rewards:
        plt.figure()
        validation_episodes, val_rewards = zip(*validation_rewards)
        plt.plot(
            validation_episodes, val_rewards, label="Validation Rewards", marker="o"
        )
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Validation Rewards")
        plt.legend()
        plt.grid()
        validation_graph_path = os.path.join(output_dir, "validation_rewards.png")
        plt.savefig(validation_graph_path)
        print(f"Validation graph saved at {validation_graph_path}")


def plot_agent_behavior(
    state_action_history,
    output_dir,
    n_days=7,
    episodes_to_plot=None,
):
    """Plot the agent's behavior over episodes"""
    if episodes_to_plot is None:
        episodes_to_plot = [len(state_action_history) - 1]  # Default to last episode

    hours_per_day = 24
    steps_per_interval = n_days * hours_per_day

    for episode_idx in episodes_to_plot:
        episode_history = state_action_history[episode_idx]

        for interval_start in range(0, len(episode_history), steps_per_interval):
            interval_end = min(
                interval_start + steps_per_interval, len(episode_history)
            )
            interval_history = episode_history[interval_start:interval_end]

            if not interval_history:
                continue

            states, actions = zip(*interval_history)
            storage, price, hour = zip(*[state[:3] for state in states])

            time_labels = [
                f"Day {i // hours_per_day + 1}, {str(timedelta(hours=i % hours_per_day))[:-3]}"
                for i in range(len(storage))
            ]

            # Adjust the step size for x-axis labels to avoid clutter
            step_size = max(1, len(storage) // 10)

            plt.figure(figsize=(12, 8))

            plt.subplot(3, 1, 1)
            plt.plot(range(len(storage)), storage, "bo-", label="Storage")
            plt.xticks(
                range(0, len(storage), step_size),
                time_labels[::step_size],
                rotation=45,
                ha="right",
            )
            plt.xlabel(f"Time (days and hours)")
            plt.ylabel("Storage")
            plt.title(
                f"Episode {episode_idx + 1}, Interval {interval_start // steps_per_interval + 1} - Storage"
            )
            plt.legend()
            plt.ylim(min(storage) - 1, max(storage) + 1)

            plt.subplot(3, 1, 2)
            plt.plot(range(len(price)), price, "ro-", label="Price")
            plt.xticks(
                range(0, len(price), step_size),
                time_labels[::step_size],
                rotation=45,
                ha="right",
            )
            plt.xlabel(f"Time (days and hours)")
            plt.ylabel("Price")
            plt.title(
                f"Episode {episode_idx + 1}, Interval {interval_start // steps_per_interval + 1} - Price"
            )
            plt.legend()
            plt.ylim(min(price) - 1, max(price) + 1)

            plt.subplot(3, 1, 3)
            plt.plot(range(len(actions)), actions, "go-", label="Action")
            plt.xticks(
                range(0, len(actions), step_size),
                time_labels[::step_size],
                rotation=45,
                ha="right",
            )
            plt.xlabel(f"Time (days and hours)")
            plt.ylabel("Action")
            plt.title(
                f"Episode {episode_idx + 1}, Interval {interval_start // steps_per_interval + 1} - Action"
            )
            plt.legend()
            plt.ylim(min(actions) - 1, max(actions) + 1)

            behavior_graph_path = os.path.join(
                output_dir,
                f"episode_{episode_idx + 1}_interval_{interval_start // steps_per_interval + 1}_behavior.png",
            )
            plt.tight_layout()
            plt.savefig(behavior_graph_path)
            plt.close()
            print(
                f"Behavior graph for Episode {episode_idx + 1}, Interval {interval_start // steps_per_interval + 1} saved at {behavior_graph_path}"
            )


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
    parser.add_argument("--plot-interval-days", type=int, default=7)
    parser.add_argument("--save-state-action-history", action="store_true")
    args = parser.parse_args()

    # Create output directory
    output_dir = create_output_directory()

    # Initialize environments
    train_env = DataCenterEnv(args.train_path)
    val_env = DataCenterEnv(args.val_path) if args.validate_every else None

    # Initialize agent
    agent = QLearningAgent(train_env, model_path=args.load_model)

    # Train agent
    training_rewards, validation_rewards, state_action_history = agent.train(
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

    # Save state-action history if requested
    if args.save_state_action_history:
        filename = "state_action_history.json"
        save_path = os.path.join(output_dir, filename)
        agent.save_state_action_history(state_action_history, save_path)
        print(f"State-action history saved at {save_path}")

    # Plot metrics if requested
    if args.make_graphs:
        plot_metrics(training_rewards, validation_rewards, output_dir)
        plot_agent_behavior(
            state_action_history,
            output_dir,
            n_days=args.plot_interval_days,
        )

    # Print final metrics
    print(f"Total reward after {args.episodes} episodes: {sum(training_rewards):.2f}")


if __name__ == "__main__":
    main()
