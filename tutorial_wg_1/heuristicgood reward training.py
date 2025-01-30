from env import DataCenterEnv
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
from datetime import timedelta
import matplotlib.dates as mdates  # Add this line
import pandas as pd

def heuristic_action(price, storage, hour, threshold_sell=150, threshold_buy=70):
    """
    Smarter heuristic:
    - Must ensure we get 120 MWh by end of day
    - Only sell if we have excess storage
    - Buy more aggressively as day progresses if we're behind
    """
    hours_left = 24 - hour
    required_energy = 120 - storage  # How much we still need today
    
    # If we're behind schedule, force buying
    if hours_left > 0:  # Prevent division by zero
        required_rate = required_energy / hours_left
        if required_rate > 10:  # If we need more than max rate, must buy now
            return 1.0
    
    # If we have excess energy and price is high, sell
    excess_energy = max(0, storage - required_energy)
    if excess_energy > 0 and price > threshold_sell:
        return -1.0
        
    # If price is good and we still need energy, buy
    if price < threshold_buy and storage < 170:
        return 1.0
    
    # Do nothing if no clear action needed
    return 0.0

def collect_episode_history(environment, threshold_buy=70, threshold_sell=140):
    state = environment.observation()
    history = []
    terminated = False
    
    while not terminated:
        storage, price, hour, _ = state
        action = heuristic_action(price, storage, hour, threshold_buy, threshold_sell)
        print_state_contents(state)
        next_state, reward, terminated = environment.step(action)
        history.append((state, action, reward))
        state = next_state
    
    return history

def print_state_contents(state):
    storage, price, hour, date = state
    print("\nState Contents:")
    print(f"Storage: {storage} MWh")
    print(f"Price: {price} $/MWh")
    print(f"Hour: {hour}")
    print(f"Date: {date}")  # This should contain the actual date from the Excel file

def plot_heuristic_behavior(history, output_dir, episode=1):
    states, actions, rewards = zip(*history)
    storage, price, hours, _ = zip(*states)
    
    # Convert hours to proper datetime sequence
    start_date = pd.to_datetime('2010-01-01')
    dates = []
    day_counter = 0
    
    for h in hours:
        # Calculate current date: add days and then hours
        current_date = start_date + pd.Timedelta(days=day_counter) + pd.Timedelta(hours=h)
        dates.append(current_date)
        if h == 24:  # End of day
            day_counter += 1
    
    hours_per_week = 24 * 7
    num_weeks = len(storage) // hours_per_week + (1 if len(storage) % hours_per_week else 0)
    
    action_colors = ['red' if a == -1 else 'yellow' if a == 0 else 'green' for a in actions]
    
    for week in range(num_weeks):
        start_idx = week * hours_per_week
        end_idx = min((week + 1) * hours_per_week, len(storage))
        week_dates = dates[start_idx:end_idx]
        
        plt.figure(figsize=(15, 12))
        
        # Storage plot
        plt.subplot(4,1,1)
        plt.plot(week_dates, storage[start_idx:end_idx], 'b-', label='Storage')
        plt.scatter(week_dates, storage[start_idx:end_idx], color='blue')
        plt.title(f'Storage Level - Week {week+1}')
        plt.ylabel('MWh')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        
        # Price plot
        plt.subplot(4,1,2)
        plt.plot(week_dates, price[start_idx:end_idx], 'g-', label='Price')
        plt.scatter(week_dates, price[start_idx:end_idx], 
                   c=action_colors[start_idx:end_idx], label='_nolegend_')
        plt.scatter([], [], c='red', label='Sell')
        plt.scatter([], [], c='yellow', label='Hold')
        plt.scatter([], [], c='green', label='Buy')
        plt.title(f'Electricity Price - Week {week+1}')
        plt.ylabel('$/MWh')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        
        # Actions plot
        plt.subplot(4,1,3)
        plt.plot(week_dates, actions[start_idx:end_idx], 'g-', label='Action')
        plt.scatter(week_dates, actions[start_idx:end_idx], color='green')
        plt.title(f'Actions Taken - Week {week+1}')
        plt.ylabel('Action (-1,0,1)')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        
        # Rewards plot
        plt.subplot(4,1,4)
        plt.plot(week_dates, rewards[start_idx:end_idx], 'm-', label='Reward')
        plt.scatter(week_dates, rewards[start_idx:end_idx], color='magenta')
        plt.title(f'Rewards - Week {week+1}')
        plt.ylabel('Reward')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'episode_{episode}_week_{week+1}_behavior.png'))
        plt.close()

def print_history(history):
    print("\nComplete Episode History:")
    print("Day | Hour | Storage | Price  | Action | Reward")
    print("-" * 50)
    
    current_day = 1
    prev_hour = 0
    
    for state, action, reward in history:
        storage, price, hour, _ = state
        hour_of_day = int(hour % 24) + 1
        
        # Increment day when hour resets
        if hour_of_day < prev_hour:
            current_day += 1
        prev_hour = hour_of_day
            
        print(f"{current_day:3d} | {hour_of_day:4d} | {storage:7.2f} | {price:6.2f} | {action:6.1f} | {reward:6.2f}")

def main():
    args = argparse.ArgumentParser()
    args.add_argument('--path', type=str, default='train.xlsx')
    args = args.parse_args()

    # plot_dir = 'validate_plots' if 'validate' in args.path else 'train_plots'
    # os.makedirs(plot_dir, exist_ok=True)
    
    environment = DataCenterEnv(args.path)
    
    # Collect history once
    history = collect_episode_history(environment, threshold_buy=70, threshold_sell=140)
    
    # Verify history length
    print(f"\nTotal timesteps in history: {len(history)}")
    
    # Print full history
    # print_history(history)
    state = environment.observation()
    print_state_contents(state)
    
    # Calculate total reward
    total_reward = sum(reward for _, _, reward in history)
    print(f"\nTotal Reward: {total_reward:.2f}")

if __name__ == "__main__":
    main()


    
if __name__ == "__main__":
    main()
