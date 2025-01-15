from env import DataCenterEnv
import numpy as np
import argparse

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

def main():
    args = argparse.ArgumentParser()
    args.add_argument('--path', type=str, default='train.xlsx')
    args.add_argument('--episodes', type=int, default=1)
    #args.add_argument('--threshold', type=float, default=75.0)
    args = args.parse_args()

    np.set_printoptions(suppress=True, precision=2)
    
    environment = DataCenterEnv(args.path)
    aggregate_reward = 0

    threshold_b = 50
    threshold_s = 100
    
    for episode in range(args.episodes):
        state = environment.observation()
        terminated = False
        episode_reward = 0
        
        threshold_b += 1
        threshold_s += 1
        
        while not terminated:
            # Get current state information
            storage, price, hour, _ = state
            
            # Determine action based on state
            action = heuristic_action(price, storage, hour, threshold_buy=70, threshold_sell=140)
            
            # Take action in environment
            next_state, reward, terminated = environment.step(action)
            episode_reward += reward
            
            # Move to next state
            state = next_state
        
        aggregate_reward += episode_reward
        #print("threshold_s = ", threshold_s)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
        
        # Reset environment for next episode
        state = environment.reset()
    
    print(f'Total reward after {args.episodes} episodes: {aggregate_reward:.2f}')
    print(f'Average reward per episode: {aggregate_reward/args.episodes:.2f}')

if __name__ == "__main__":
    main()