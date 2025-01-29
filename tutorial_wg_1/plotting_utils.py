import matplotlib.pyplot as plt
import os
from datetime import timedelta


def plot_metrics(training_rewards, validation_rewards, output_dir):
    """Plot and save training/validation metrics"""
    # Plot training rewards
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(training_rewards) + 1), training_rewards, label="Training Rewards"
    )
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Rewards")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    training_graph_path = os.path.join(output_dir, "training_rewards.png")
    plt.savefig(training_graph_path)
    print(f"Training graph saved at {training_graph_path}")

    # Plot validation rewards if available
    if validation_rewards:
        plt.figure(figsize=(10, 6))
        validation_episodes, val_rewards = zip(*validation_rewards)
        plt.plot(
            validation_episodes, val_rewards, label="Validation Rewards", marker="o"
        )
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Validation Rewards")
        plt.legend()
        plt.grid()
        plt.tight_layout()
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

            states, actions, original_rewards, reshaped_rewards = zip(*interval_history)
            storage, price, _ = zip(*[state[:3] for state in states])

            time_labels = [
                f"Day {i // hours_per_day + 1}, {str(timedelta(hours=i % hours_per_day))[:-3]}"
                for i in range(len(storage))
            ]

            # Adjust the step size for x-axis labels to avoid clutter
            step_size = max(1, len(storage) // 10)

            plt.figure(figsize=(14, 10))

            plt.subplot(4, 1, 1)
            plt.plot(range(len(storage)), storage, "bo-", label="Storage")
            plt.xticks(
                range(0, len(storage), step_size),
                time_labels[::step_size],
                rotation=45,
                ha="right",
            )
            plt.xlabel("Time (days and hours)")
            plt.ylabel("Storage")
            plt.title(
                f"Episode {episode_idx + 1}, Interval {interval_start // steps_per_interval + 1} - Storage"
            )
            plt.legend()
            plt.ylim(min(storage) - 1, max(storage) + 1)

            plt.subplot(4, 1, 2)
            plt.plot(range(len(price)), price, "ro-", label="Price")
            plt.xticks(
                range(0, len(price), step_size),
                time_labels[::step_size],
                rotation=45,
                ha="right",
            )
            plt.xlabel("Time (days and hours)")
            plt.ylabel("Price")
            plt.title(
                f"Episode {episode_idx + 1}, Interval {interval_start // steps_per_interval + 1} - Price"
            )
            plt.legend()
            plt.ylim(min(price) - 1, max(price) + 1)

            plt.subplot(4, 1, 3)
            plt.plot(range(len(actions)), actions, "go-", label="Action")
            plt.xticks(
                range(0, len(actions), step_size),
                time_labels[::step_size],
                rotation=45,
                ha="right",
            )
            plt.xlabel("Time (days and hours)")
            plt.ylabel("Action")
            plt.title(
                f"Episode {episode_idx + 1}, Interval {interval_start // steps_per_interval + 1} - Action"
            )
            plt.legend()
            plt.ylim(min(actions) - 1, max(actions) + 1)

            plt.subplot(4, 1, 4)
            plt.plot(
                range(len(original_rewards)),
                original_rewards,
                "mo-",
                label="Original Reward",
            )
            plt.plot(
                range(len(reshaped_rewards)),
                reshaped_rewards,
                "co-",
                label="Reshaped Reward",
            )
            plt.xticks(
                range(0, len(original_rewards), step_size),
                time_labels[::step_size],
                rotation=45,
                ha="right",
            )
            plt.xlabel("Time (days and hours)")
            plt.ylabel("Reward")
            plt.title(
                f"Episode {episode_idx + 1}, Interval {interval_start // steps_per_interval + 1} - Rewards"
            )
            plt.legend()
            plt.ylim(
                min(min(original_rewards), min(reshaped_rewards)) - 1,
                max(max(original_rewards), max(reshaped_rewards)) + 1,
            )

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
