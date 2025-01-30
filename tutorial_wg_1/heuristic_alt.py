from env import DataCenterEnv
import numpy as np
import argparse

def heuristic_action(price, storage, hour, day_max_price, threshold_buy=70):
    """
    Adaptive heuristic:
    - Must ensure we get 120 MWh by end of day
    - Sell when price is highest of the day so far and storage > 50
    - Buy more aggressively as day progresses if we're behind
    """
    hours_left = 24 - hour % 24  # Changed to use modulo for multi-day episodes
    required_energy = 120 - storage  # How much we still need today
    
    # If we're behind schedule, force buying
    if hours_left > 0:  # Prevent division by zero
        required_rate = required_energy / hours_left
        if required_rate > 10:  # If we need more than max rate, must buy now
            return 1.0
    
    # If we have enough storage and current price is highest of the day, sell
    if storage > 30 and price > 50 and price >= 0.9*day_max_price:  # Added minimum threshold
        return -1.0
        
    # If price is good and we still need energy, buy
    if price < threshold_buy and storage < 170:
        return 1.0
    
    # Do nothing if no clear action needed
    return 0.0

def main():
    args = argparse.ArgumentParser()
    args.add_argument('--path', type=str, default='train.xlsx')
    args.add_argument('--episodes', type=int, default=1)
    args = args.parse_args()

    np.set_printoptions(suppress=True, precision=2)
    
    environment = DataCenterEnv(args.path)
    aggregate_reward = 0
    
    for episode in range(args.episodes):
        state = environment.observation()
        terminated = False
        episode_reward = 0
        day_max_price = 0  # Track maximum price for current day
        last_hour = -1  # Track hour changes
        
        while not terminated:
            # Get current state information
            storage, price, hour, _ = state
            
            # Reset max price at the start of each day
            if hour < last_hour:  # Hour decreased means new day
                day_max_price = price
            else:
                day_max_price = max(day_max_price, price)
            
            last_hour = hour
            
            # Determine action based on state
            action = heuristic_action(price, storage, hour, day_max_price)
            
            # Take action in environment
            next_state, reward, terminated = environment.step(action)
            episode_reward += reward
            
            # Move to next state
            state = next_state
        
        aggregate_reward += episode_reward
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
        
        # Reset environment for next episode
        state = environment.reset()
    
    print(f'Total reward after {args.episodes} episodes: {aggregate_reward:.2f}')
    print(f'Average reward per episode: {aggregate_reward/args.episodes:.2f}')

if __name__ == "__main__":
    main()