# DQN Agent

## Setup
Set up the virtual environment with the requirements specified in `DQN/PRL-environment.yml`.

## Data
All commands below assume a `DQN/data` folder with the train and validation sets, but these can be overridden with:
- `--train-path TRAIN_PATH`
- `--val-path VAL_PATH`


## If you want to just run the best model on the test data
First, unzip the models in DQN/DQN-models.zip

Best model: `run_2025-01-30_21-14-26_DQN-rs5-buy-under-70-sell-over-180-a89f9b88eb43`

```bash
python main.py --validate-only --run-dir DQN/DQN-models/run_2025-01-30_21-14-26_DQN-rs5-buy-under-70-sell-over-180-a89f9b88eb43 --val-path /path/to/test/data
```

---

## Run Training with Optional Periodical Validation
```bash
python main.py --episodes 100 --validate-every 10 --output-dir /path/to/output --run-name example-run-name
```
Note: `--run-name` is a string added to the run directory created in the output directory (default is `run_timestamp`).

## Validate / Test
```bash
python main.py --validate-only --run-dir /path/to/output/run_timestamp_example-run-name
```

## Plot Behavior Analysis
```bash
python plotting_utils.py --n-days 3 --model-run-dir /path/to/output/run_timestamp_example-run-name
```

## Hyperparameter optimization

Do the following to get all options and setup as needed.

```bash
python hyperparam_opt.py -h
```
