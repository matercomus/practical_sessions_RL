import matplotlib.pyplot as plt
import json
import os
import argparse
from datetime import timedelta


def plot_metrics(
    train_orig, train_shaped, val_orig, val_shaped, val_episodes, output_dir
):
    plt.figure(figsize=(12, 6))

    # Training data handling
    episodes = list(range(1, len(train_orig) + 1)) if train_orig else []

    # Training curves (only if data exists)
    if train_orig:
        plt.plot(
            episodes,
            train_orig,
            label="Training Original Reward",
            color="blue",
            alpha=0.6,
        )
        plt.plot(
            episodes,
            train_shaped,
            label="Training Shaped Reward",
            color="green",
            alpha=0.6,
        )

    # Validation markers if available
    if val_episodes:
        plt.scatter(
            val_episodes,
            val_orig,
            marker="X",
            s=100,
            label="Validation Original",
            color="darkblue",
        )
        plt.scatter(
            val_episodes,
            val_shaped,
            marker="X",
            s=100,
            label="Validation Shaped",
            color="darkgreen",
        )

    plt.title("Reward Progression During Training")
    plt.xlabel("Training Episodes")
    plt.ylabel("Reward Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save and close
    plt.savefig(os.path.join(output_dir, "reward_progression.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "reward_progression.pdf"))
    plt.close()


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
            executed_actions = [step["executed_action"] for step in interval_history]
            original_rewards = [step["original_reward"] for step in interval_history]
            reshaped_rewards = [step["shaped_reward"] for step in interval_history]
            was_forced = [step["was_forced"] for step in interval_history]
            reasons = [step["force_reason"] for step in interval_history]

            storage, price, _ = zip(*[state[:3] for state in states])

            time_labels = [
                f"Day {i // hours_per_day + 1}, {str(timedelta(hours=i % hours_per_day))[:-3]}"
                for i in range(len(storage))
            ]

            # Prepare visualization parameters
            executed_colors = []
            edge_colors = []
            forced_indices = []
            reason_markers = []

            for i, (ea, ca, wf, reason) in enumerate(
                zip(executed_actions, chosen_actions, was_forced, reasons)
            ):
                # Executed action color
                if ea > 0:
                    fill_color = "green"
                elif ea < 0:
                    fill_color = "red"
                else:
                    fill_color = "yellow"
                executed_colors.append(fill_color)

                # Edge color (chosen action)
                if ca > 0:
                    edge_color = "darkgreen"
                elif ca < 0:
                    edge_color = "darkred"
                else:
                    edge_color = "gold"
                edge_colors.append(edge_color if ca != ea else "none")

                # Track forced actions with reasons
                if wf:
                    forced_indices.append(i)
                    reason_markers.append({"x": i, "y": price[i], "reason": reason})

            plt.figure(figsize=(14, 14))

            # Storage plot
            plt.subplot(5, 1, 1)
            plt.plot(storage, "bo-", label="Storage Level")
            plt.xticks(range(0, len(storage), len(storage) // 8), [], rotation=45)
            plt.ylabel("Storage Level")
            plt.title(f"Episode {episode_idx+1} Behavior Analysis")
            plt.legend()

            # Price plot with action visualization
            plt.subplot(5, 1, 2)
            plt.plot(price, "k-", label="Price", alpha=0.5)

            # Plot executed actions with edge colors showing chosen actions
            scatter = plt.scatter(
                range(len(price)),
                price,
                c=executed_colors,
                s=60,
                edgecolors=edge_colors,
                linewidths=2,
                zorder=3,
                marker="o",
            )

            # Add forced action markers
            marker_map = {
                "forced_buy": ("^", "green"),
                "prevented_sell": ("v", "yellow"),
                "no_storage": ("s", "blue"),
            }
            for rm in reason_markers:
                marker, color = marker_map.get(rm["reason"], ("x", "black"))
                plt.scatter(
                    rm["x"],
                    rm["y"] + 5,
                    marker=marker,
                    s=100,
                    color=color,
                    edgecolors="black",
                    linewidths=1,
                    zorder=4,
                )

            plt.ylabel("Price & Actions")
            plt.xticks(range(0, len(price), len(price) // 8), [], rotation=45)

            # Action discrepancy plot
            plt.subplot(5, 1, 3)
            discrepancies = [
                1 if ca != ea else 0 for ca, ea in zip(chosen_actions, executed_actions)
            ]
            plt.plot(discrepancies, "m-", label="Action Discrepancy (1 = mismatch)")
            plt.ylabel("Action Mismatch")
            plt.ylim(-0.1, 1.1)
            plt.xticks(
                range(0, len(discrepancies), len(discrepancies) // 8), [], rotation=45
            )

            # Original Reward plot
            plt.subplot(5, 1, 4)
            plt.plot(original_rewards, "go-", label="Original Reward")
            plt.ylabel("Original Reward")
            plt.xticks(
                range(0, len(original_rewards), len(original_rewards) // 8),
                [],
                rotation=45,
            )

            # Reshaped Reward plot
            plt.subplot(5, 1, 5)
            plt.plot(reshaped_rewards, "co-", label="Reshaped Reward")
            plt.ylabel("Reshaped Reward")
            plt.xticks(
                range(0, len(reshaped_rewards), len(reshaped_rewards) // 8),
                [
                    time_labels[i]
                    for i in range(0, len(reshaped_rewards), len(reshaped_rewards) // 8)
                ],
                rotation=45,
                ha="right",
            )

            # Create unified legend
            legend_elements = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="green",
                    markersize=10,
                    label="Executed Buy",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="red",
                    markersize=10,
                    label="Executed Sell",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="yellow",
                    markersize=10,
                    label="Executed Hold",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="darkgreen",
                    markerfacecolor="none",
                    markersize=10,
                    label="Chosen Buy",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="darkred",
                    markerfacecolor="none",
                    markersize=10,
                    label="Chosen Sell",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="gold",
                    markerfacecolor="none",
                    markersize=10,
                    label="Chosen Hold",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="^",
                    color="green",
                    linestyle="None",
                    markersize=10,
                    label="Forced Buy",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="v",
                    color="yellow",
                    linestyle="None",
                    markersize=10,
                    label="Forced hold",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="blue",
                    linestyle="None",
                    markersize=10,
                    label="No Storage",
                ),
            ]

            plt.figlegend(
                handles=legend_elements,
                loc="lower center",
                ncol=3,
                bbox_to_anchor=(0.5, -0.05),
            )

            # Save the figure
            behavior_graph_path = os.path.join(
                output_dir,
                f"episode_{episode_idx + 1}_interval_{interval_start // steps_per_interval + 1}_behavior.png",
            )
            plt.tight_layout()
            plt.savefig(behavior_graph_path, bbox_inches="tight")
            plt.close()
            print(f"Behavior graph saved at {behavior_graph_path}")


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
