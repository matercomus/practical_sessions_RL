import os
import argparse
import logging
import optuna
from DQN_testing import DeepQLearningAgent  # or however your code is organized
from env import DataCenterEnv  # same note as above

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def optimize_hyperparameters(
    train_path: str,
    val_path: str,
    n_trials: int = 10,
    train_episodes: int = 50,
    validate_every: int = 10,
    output_dir: str = ".",
    n_jobs: int = 1,
):
    """
    Runs Optuna hyperparameter optimization for DeepQLearningAgent on the DataCenterEnv.
    """

    # Make sure top-level output directory exists
    os.makedirs(output_dir, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        """
        Optuna objective function that returns the validation score (higher is better).
        Each trial should create its own environment and agent.
        """
        # Create subdirectory for this trial to store logs/models separately
        trial_dir = os.path.join(output_dir, f"trial_{trial.number}")
        os.makedirs(trial_dir, exist_ok=True)

        # Suggest hyperparameters
        discount_rate = trial.suggest_float("discount_rate", 0.90, 0.999, step=0.01)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        tau = trial.suggest_float("tau", 0.001, 0.05)
        batch_size = trial.suggest_int("batch_size", 32, 256, step=32)

        # Instantiate fresh environments for this trial
        train_env = DataCenterEnv(train_path)
        val_env = DataCenterEnv(val_path)

        # Create agent using the suggested hyperparams
        agent = DeepQLearningAgent(
            env=train_env,
            discount_rate=discount_rate,
            learning_rate=learning_rate,
            tau=tau,
            batch_size=batch_size,
            output_dir=trial_dir,  # So logs/model are saved per-trial
            total_episodes=train_episodes,
        )

        # Train agent
        agent.train(
            env=train_env,
            episodes=train_episodes,
            validate_every=validate_every,
            val_env=val_env,
        )

        # Evaluate agent on validation set
        avg_val_reward = agent.validate(val_env, num_episodes=5)
        return avg_val_reward

    # Create Optuna study, maximize reward
    study = optuna.create_study(direction="maximize")

    # Run the optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,  # This parameter allows parallel trial execution
    )

    logger.info("Optuna search complete.")
    logger.info(f"Best trial number: {study.best_trial.number}")
    logger.info(f"Best value (avg reward): {study.best_trial.value}")
    logger.info(f"Best hyperparams: {study.best_trial.params}")

    return study


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter optimization for DeepQLearningAgent."
    )
    parser.add_argument("--train-path", type=str, default="./data/train.xlsx")
    parser.add_argument("--val-path", type=str, default="./data/validate.xlsx")
    parser.add_argument(
        "--hp-opt-trials", type=int, default=10, help="Number of Optuna trials."
    )
    parser.add_argument(
        "--hp-opt-train-episodes",
        type=int,
        default=50,
        help="Number of episodes per trial.",
    )
    parser.add_argument(
        "--validate-every", type=int, default=10, help="Validate every N episodes."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="optuna_results",
        help="Directory where logs/results are stored.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for Optuna optimization.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Run hyperparameter optimization
    study = optimize_hyperparameters(
        train_path=args.train_path,
        val_path=args.val_path,
        n_trials=args.hp_opt_trials,
        train_episodes=args.hp_opt_train_episodes,
        validate_every=args.validate_every,
        output_dir=args.output_dir,
        n_jobs=args.n_jobs,
    )

    # Save best params to a file
    best_params_path = os.path.join(args.output_dir, "best_params.txt")
    with open(best_params_path, "w") as f:
        f.write(str(study.best_trial.params))
    logger.info(f"Best hyperparameters saved to {best_params_path}")


if __name__ == "__main__":
    main()
