from env import DataCenterEnv
from q_learning_tabular import QAgent
import numpy as np
import argparse

args = argparse.ArgumentParser()
args.add_argument('--path', type=str, default='train.xlsx')
args = args.parse_args()

np.set_printoptions(suppress=True, precision=2)
path_to_dataset = args.path

environment = DataCenterEnv(path_to_dataset)
agent = QAgent(environment)

# Train agent
agent.train(episodes=1000)

# Test agent
state = environment.observation()
aggregate_reward = 0
terminated = False

while not terminated:
    action = agent.act(state)
    next_state, reward, terminated = environment.step(action)
    state = next_state
    aggregate_reward += reward
    print("Action:", action)
    print("Next state:", next_state)
    print("Reward:", reward)

print('Total reward:', aggregate_reward)