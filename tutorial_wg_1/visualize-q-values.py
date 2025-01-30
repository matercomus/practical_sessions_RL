import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import argparse
import logging
import json

from DQN_testing import DQN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_and_params(run_folder):
    """
    Load the model checkpoint and parameters from a run folder.

    Args:
        run_folder: Path to the run folder containing model checkpoint and hyperparameters

    Returns:
        tuple: (model, hyperparameters)
    """
    # Set up paths for model checkpoint and hyperparameters
    checkpoint_path = os.path.join(run_folder, "model_checkpoint.pth")
    hparams_path = os.path.join(run_folder, "hparams.json")

    # Load hyperparameters
    if os.path.exists(hparams_path):
        with open(hparams_path, "r") as f:
            hparams = json.load(f)
        logger.info("Loaded hyperparameters from hparams.json")
    else:
        hparams = {}
        logger.warning("No hyperparameters file found, using defaults")

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load the checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No model checkpoint found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Create and load the model
    model = DQN(input_dim=4, output_dim=3).to(device)
    model.load_state_dict(checkpoint["policy_net_state_dict"])
    model.eval()

    logger.info(f"Successfully loaded model from {checkpoint_path}")

    return model, hparams


def create_qvalue_plots(
    model,
    output_dir,
    storage_scale=170.0,
    price_min=0.0,
    price_max=100.0,
    storage_levels=10,
    price_levels=10,
):
    """
    Creates heatmaps visualizing Q-values across different state dimensions.

    Args:
        model: The loaded DQN model
        output_dir: Directory to save the plots
        storage_scale: Maximum storage capacity
        price_min/max: Price range for visualization
        storage_levels/price_levels: Resolution of the visualization grid
    """
    logger.info("Starting Q-value visualization creation")

    # Create a grid of states to evaluate
    storage_values = np.linspace(0, storage_scale, storage_levels)
    price_values = np.linspace(price_min, price_max, price_levels)

    # Key time points to visualize
    hours_to_plot = [0, 6, 12, 18, 23]  # Different times of day
    days_to_plot = [1, 4, 7]  # Start, middle, end of week

    # Create custom colormap for intuitive visualization
    colors = ["red", "yellow", "green"]
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=100)

    for hour in hours_to_plot:
        for day in days_to_plot:
            logger.info(f"Processing visualization for hour {hour}, day {day}")

            # Create state value grids
            X, Y = np.meshgrid(storage_values, price_values)
            q_values = {
                "sell": np.zeros_like(X),
                "hold": np.zeros_like(X),
                "buy": np.zeros_like(X),
            }

            # Calculate Q-values across the grid
            for i in range(storage_levels):
                for j in range(price_levels):
                    # Normalize state components to [0,1] range
                    norm_state = np.array(
                        [
                            storage_values[i] / storage_scale,
                            (price_values[j] - price_min) / (price_max - price_min),
                            hour / 23.0,
                            (day - 1) / 6.0,
                        ],
                        dtype=np.float32,
                    )

                    # Get device from model and move tensor to the same device
                    device = next(model.parameters()).device
                    state_tensor = (
                        torch.tensor(norm_state, dtype=torch.float32)
                        .unsqueeze(0)
                        .to(device)
                    )

                    with torch.no_grad():
                        q_vals = model(state_tensor).cpu().numpy()[0]
                        q_values["sell"][j, i] = q_vals[0]
                        q_values["hold"][j, i] = q_vals[1]
                        q_values["buy"][j, i] = q_vals[2]
            # Create visualization with three subplots
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(f"Q-Values at Hour {hour:02d}, Day {day}")

            # Find global value range for consistent color scaling
            vmin = min(q_values[action].min() for action in q_values)
            vmax = max(q_values[action].max() for action in q_values)

            # Create heatmaps for each action
            for ax, (action, values) in zip(axes, q_values.items()):
                im = ax.pcolormesh(X, Y, values, cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_title(f"{action.capitalize()} Action Q-Values")
                ax.set_xlabel("Storage Level (MWh)")
                ax.set_ylabel("Price ($/MWh)")
                plt.colorbar(im, ax=ax)

            # Save the visualization
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f"qvalues_hour{hour:02d}_day{day}.png")
            plt.savefig(plot_path, bbox_inches="tight", dpi=300)
            plt.close()

            logger.info(f"Saved visualization to {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Q-values from a DQN run folder"
    )
    parser.add_argument(
        "--run-folder",
        type=str,
        required=True,
        help="Path to the run folder containing model checkpoint and hyperparameters",
    )
    parser.add_argument(
        "--storage-levels",
        type=int,
        default=10,
        help="Number of storage levels to sample",
    )
    parser.add_argument(
        "--price-levels", type=int, default=10, help="Number of price levels to sample"
    )

    args = parser.parse_args()

    # Create output directory within the run folder
    output_dir = os.path.join(args.run_folder, "qvalue_plots")
    os.makedirs(output_dir, exist_ok=True)

    # Load model and parameters
    try:
        model, hparams = load_model_and_params(args.run_folder)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return

    # Try to load price range from state_action_history if available
    history_path = os.path.join(args.run_folder, "state_action_history.json")
    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                history = json.load(f)
            # Extract price values from history
            prices = [step["state"][1] for episode in history for step in episode]
            price_min, price_max = min(prices), max(prices)
            logger.info(f"Using price range from history: [{price_min}, {price_max}]")
        except Exception as e:
            logger.warning(f"Could not load price range from history: {e}")
            price_min, price_max = 0.0, 100.0
    else:
        price_min, price_max = 0.0, 100.0

    # Create visualizations
    create_qvalue_plots(
        model,
        output_dir,
        price_min=price_min,
        price_max=price_max,
        storage_levels=args.storage_levels,
        price_levels=args.price_levels,
    )

    logger.info(f"Q-value visualization complete! Results saved in {output_dir}")


if __name__ == "__main__":
    main()
