import matplotlib.pyplot as plt
import json
import os
import argparse
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

            # Extract values from dictionaries using proper keys
            states = [step["state"] for step in interval_history]
            chosen_actions = [step["chosen_action"] for step in interval_history]
            executed_action = [step["executed_action"] for step in interval_history]
            original_rewards = [step["original_reward"] for step in interval_history]
            reshaped_rewards = [
                step["shaped_reward"] for step in interval_history
            ]  # Corrected key name
            was_forced = [step["was_forced"] for step in interval_history]
            reason = [step["force_reason"] for step in interval_history]

            storage, price, _ = zip(*[state[:3] for state in states])

            time_labels = [
                f"Day {i // hours_per_day + 1}, {str(timedelta(hours=i % hours_per_day))[:-3]}"
                for i in range(len(storage))
            ]

            # Prepare action visualization parameters
            executed_actions = list(executed_action)
            chosen_actions = list(chosen_actions)
            was_forced_list = list(was_forced)

            executed_colors = []
            edge_colors = []
            discrepancy_indices = []
            forced_indices = []

            for i in range(len(executed_actions)):
                ea = executed_actions[i]
                ca = chosen_actions[i]
                wf = was_forced_list[i]

                # Executed action color
                if ea > 0:
                    color = "green"
                elif ea < 0:
                    color = "red"
                else:
                    color = "yellow"
                executed_colors.append(color)

                # Edge color for action mismatch
                ea_dir = 1 if ea > 0 else (-1 if ea < 0 else 0)
                ca_dir = ca
                if ca_dir != ea_dir:
                    edge_colors.append("black")
                    discrepancy_indices.append(i)
                else:
                    edge_colors.append("none")

                # Track forced actions
                if wf:
                    forced_indices.append(i)

            step_size = max(1, len(storage) // 10)

            plt.figure(figsize=(14, 12))

            # Storage plot
            plt.subplot(4, 1, 1)
            plt.plot(range(len(storage)), storage, "bo-", label="Storage")
            plt.xticks(
                range(0, len(storage), step_size),
                time_labels[::step_size],
                rotation=45,
                ha="right",
            )
            plt.ylabel("Storage")
            plt.title(
                f"Episode {episode_idx + 1}, Interval {interval_start // steps_per_interval + 1}"
            )
            plt.legend()
            plt.ylim(min(storage) - 1, max(storage) + 1)

            # Price plot with action visualization
            plt.subplot(4, 1, 2)
            plt.plot(range(len(price)), price, "k-", label="Price")
            # Plot executed actions
            plt.scatter(
                range(len(price)),
                price,
                c=executed_colors,
                s=40,
                edgecolors=edge_colors,
                linewidths=0.5,
                zorder=2,
                marker="o",
            )
            # Overlay forced actions
            if forced_indices:
                forced_x = [x for x in forced_indices]
                forced_y = [price[x] for x in forced_indices]
                plt.scatter(
                    forced_x,
                    forced_y,
                    marker="X",
                    s=60,
                    edgecolors="black",
                    facecolors="none",
                    linewidths=1,
                    zorder=3,
                    label="Forced Action",
                )

            plt.xticks(
                range(0, len(price), step_size),
                time_labels[::step_size],
                rotation=45,
                ha="right",
            )
            plt.ylabel("Price (Actions)")
            # Legend
            legend_handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="green",
                    markersize=10,
                    label="Buy (Executed)",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="red",
                    markersize=10,
                    label="Sell (Executed)",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="yellow",
                    markersize=10,
                    label="Hold (Executed)",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="black",
                    linestyle="None",
                    markersize=10,
                    label="Action Override",
                    markerfacecolor="none",
                    markeredgewidth=1,
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="X",
                    color="black",
                    linestyle="None",
                    markersize=10,
                    label="Forced Action",
                    markeredgewidth=1,
                ),
            ]
            plt.legend(handles=legend_handles, loc="upper right")

            # Original Reward plot
            plt.subplot(4, 1, 3)
            plt.plot(
                range(len(original_rewards)),
                original_rewards,
                "mo-",
                label="Original Reward",
            )
            plt.xticks(
                range(0, len(original_rewards), step_size),
                time_labels[::step_size],
                rotation=45,
                ha="right",
            )
            plt.ylabel("Original Reward")
            plt.legend()
            plt.ylim(min(original_rewards) - 1, max(original_rewards) + 1)

            # Reshaped Reward plot
            plt.subplot(4, 1, 4)
            plt.plot(
                range(len(reshaped_rewards)),
                reshaped_rewards,
                "co-",
                label="Reshaped Reward",
            )
            plt.xticks(
                range(0, len(reshaped_rewards), step_size),
                time_labels[::step_size],
                rotation=45,
                ha="right",
            )
            plt.xlabel("Time (days and hours)")
            plt.ylabel("Reshaped Reward")
            plt.legend()
            plt.ylim(min(reshaped_rewards) - 1, max(reshaped_rewards) + 1)

            # Save the figure
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-run-dir", type=str, default=".")
    parser.add_argument("--n-days", type=int, default=7)
    args = parser.parse_args()

    output_dir_bp = os.path.join(args.model_run_dir, "agent_behavior_plots")
    os.makedirs(output_dir_bp, exist_ok=True)

    state_action_history_path = os.path.join(
        args.model_run_dir, "state_action_history.json"
    )
    with open(state_action_history_path, "r") as f:
        state_action_history = json.load(f)

    plot_agent_behavior(
        state_action_history,
        output_dir=output_dir_bp,
        n_days=args.n_days,
    )

    print(f"Agent behavior plots saved at {output_dir_bp}")


if __name__ == "__main__":
    main()
