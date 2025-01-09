import numpy as np
def validate_agent(env, agent, num_validation_episodes=5):
    """
    Run validation episodes without updating the Q-table
    """
    validation_rewards = []
    
    for episode in range(num_validation_episodes):
        state = env.observation()
        terminated = False
        episode_reward = 0
        
        while not terminated:
            # Use greedy policy during validation (no exploration)
            storage_idx, price_idx, hour_idx, day_idx = agent.discretize_state(state)
            action_idx = np.argmax(agent.Q_table[storage_idx, price_idx, hour_idx, day_idx])
            action = agent.action_space[action_idx]
            
            # Take action in environment
            next_state, reward, terminated = env.step(action)
            episode_reward += reward
            state = next_state
            
        validation_rewards.append(episode_reward)
        env.day = 1  # Reset environment
        
    return np.mean(validation_rewards)
