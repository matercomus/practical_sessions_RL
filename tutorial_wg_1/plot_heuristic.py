import matplotlib.pyplot as plt
import json
import os
from env import DataCenterEnv
import numpy as np
from datetime import timedelta
from heuristic_good import heuristic_action
import argparse

def plot_heuristic_behavior(state_action_history, output_dir, n_days=7, episodes_to_plot=None):
    """Plot the heuristic agent's behavior over episodes"""
    if episodes_to_plot is None:
        episodes_to_plot = [len(state_action_history) - 1]  # Default to last episode

    hours_per_day = 24
    steps_per_interval = n_days * hours_per_day

    for episode_idx in episodes_to_plot:
        episode_history = state_action_history[episode_idx]

        for interval_start in range(0, len(episode_history), steps_per_interval):
            interval_end = min(interval_start + steps_per_interval, len(episode_history))
            interval_history = episode_history[interval_start:interval_end]

            if not interval_history:
                continue

            # Extract values from dictionaries using proper keys
            states = [step["state"] for step in interval_history]
            actions = [step["chosen_action"] for step in interval_history]
            storage, price, _ = zip(*[state[:3] for state in states])

            time_labels = [
                f"Day {i // hours_per_day + 1}, {str(timedelta(hours=i % hours_per_day))[:-3]}"
                for i in range(len(storage))
            ]

            # Prepare action colors
            action_colors = []
            for action in actions:
                if action > 0:
                    color = "green"
                elif action < 0:
                    color = "red"
                else:
                    color = "yellow"
                action_colors.append(color)

            plt.figure(figsize=(14, 8))

            # Storage plot
            plt.subplot(2, 1, 1)
            plt.plot(storage, "bo-", label="Storage Level")
            plt.xticks(range(0, len(storage), len(storage) // 8), [], rotation=45)
            plt.ylabel("Storage Level (MWh)")
            plt.title(f"Heuristic Strategy Analysis - Episode {episode_idx+1}")
            plt.legend()
            plt.grid(True)

            # Price plot with action visualization
            plt.subplot(2, 1, 2)
            plt.plot(price, "k-", label="Price", alpha=0.5)
            
            # Plot actions
            scatter = plt.scatter(
                range(len(price)),
                price,
                c=action_colors,
                s=60,
                zorder=3,
                marker="o",
                label="Actions"
            )
            
            plt.ylabel("Price ($/MWh) & Actions")
            plt.xticks(
                range(0, len(price), len(price) // 8),
                [time_labels[i] for i in range(0, len(price), len(price) // 8)],
                rotation=45,
                ha="right"
            )
            plt.grid(True)

            # Create unified legend
            legend_elements = [
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="green", 
                          markersize=10, label="Buy"),
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="red", 
                          markersize=10, label="Sell"),
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="yellow", 
                          markersize=10, label="Hold"),
            ]

            plt.figlegend(
                handles=legend_elements,
                loc="lower center",
                ncol=3,
                bbox_to_anchor=(0.5, -0.15)
            )

            # Save the figure
            behavior_graph_path = os.path.join(
                output_dir,
                f"heuristic_episode_{episode_idx + 1}_interval_{interval_start // steps_per_interval + 1}_behavior.png"
            )
            plt.tight_layout()
            plt.savefig(behavior_graph_path, bbox_inches="tight")
            plt.close()
            print(f"Heuristic behavior graph saved at {behavior_graph_path}")

def run_heuristic(env_path, n_episodes=1, threshold_buy=70, threshold_sell=140):
    """Run heuristic strategy and collect state-action history"""
    environment = DataCenterEnv(env_path)
    state_action_history = []
    
    for episode in range(n_episodes):
        state = environment.observation()
        terminated = False
        episode_history = []
        
        while not terminated:
            storage, price, hour, date = state
            
            # Get heuristic action
            action = heuristic_action(price, storage, hour, 
                                    threshold_sell=threshold_sell, 
                                    threshold_buy=threshold_buy)
            
            # Take action and get next state
            next_state, reward, terminated = environment.step(action)
            
            step_info = {
                "state": list(state),
                "chosen_action": action
            }
            
            episode_history.append(step_info)
            state = next_state
            
        state_action_history.append(episode_history)
        environment.reset()
        
    return state_action_history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-path", type=str, default="train.xlsx")
    parser.add_argument("--output-dir", type=str, default="heuristic_plots")
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--n-days", type=int, default=7)
    parser.add_argument("--threshold-buy", type=float, default=70)
    parser.add_argument("--threshold-sell", type=float, default=140)
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run heuristic and collect history
    state_action_history = run_heuristic(
        args.env_path,
        args.n_episodes,
        args.threshold_buy,
        args.threshold_sell
    )
    
    # Save history
    history_path = os.path.join(args.output_dir, "heuristic_history.json")
    with open(history_path, "w") as f:
        json.dump(state_action_history, f)
    
    # Create plots
    plot_heuristic_behavior(
        state_action_history,
        output_dir=args.output_dir,
        n_days=args.n_days,
        episodes_to_plot=list(range(args.n_episodes))
    )
    
    print(f"Heuristic behavior plots saved at {args.output_dir}")

if __name__ == "__main__":
    main()