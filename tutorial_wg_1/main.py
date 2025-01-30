from env import DataCenterEnv
import argparse
import os
from datetime import datetime
from plotting_utils import plot_metrics, plot_agent_behavior
from q_learning_agent import QLearningAgent


def create_output_directory():
    """Create and return path to timestamped output directory"""
    # Get project directory path
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create timestamp
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create output directory path in project directory
    output_dir = os.path.join(project_dir, f"run_{run_timestamp}")
    
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directory: {e}")
        # Fallback to user's home directory
        home_dir = os.path.expanduser("~")
        output_dir = os.path.join(home_dir, "rl_outputs", f"run_{run_timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
    print(f"Output directory created at: {output_dir}")
    return output_dir


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=str, default="train.xlsx")
    parser.add_argument("--val-path", type=str, default="validate.xlsx")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--validate-every", type=int, default=50)
    parser.add_argument("--make-graphs", action="store_true")
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

    # Print and save q table stats if agent has this mehtod
    if hasattr(agent, "print_and_save_q_table_stats"):
        agent.print_and_save_q_table_stats(output_dir)

    # Train agent
    training_rewards, validation_rewards, state_action_history = agent.train(
        train_env,
        episodes=args.episodes,
        validate_every=args.validate_every,
        val_env=val_env,
    )

    # Save model
    model_path = os.path.join(output_dir, "model.h5")
    agent.save(model_path)
    print(f"Model saved at {model_path}")

    # Save state-action history if requested
   
    filename = "state_action_history.json"
    save_path = os.path.join(output_dir, filename)
    # agent.save_state_action_history(state_action_history, save_path)
    # print(f"State-action history saved at {save_path}")

# Plot metrics if requested

    # plot_metrics(training_rewards, validation_rewards, output_dir)
    # plot_agent_behavior(
    #     state_action_history,
    #     output_dir,
    #     n_days=args.plot_interval_days,
    # )

    # Print final metrics
    print(f"Total reward after {args.episodes} episodes: {sum(training_rewards):.2f}")


if __name__ == "__main__":
    main()
