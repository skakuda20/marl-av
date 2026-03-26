"""
Simple example showing how to use the intersection environment.
"""

import numpy as np
from env.intersection_env import IntersectionEnv
from agents.heuristic import create_heuristic_pair


def run_simple_example():
    """Run a simple example episode."""
    
    # Create environment
    env = IntersectionEnv(dt=0.1, max_steps=200)
    
    # Create two cautious agents
    agent_1, agent_2 = create_heuristic_pair("yield", "aggressive")
    
    print("Running intersection scenario...")
    print("Agent 1: Yield policy")
    print("Agent 2: Aggressive policy")
    print("-" * 50)
    
    # Reset environment
    obs, info = env.reset()
    
    total_reward_1 = 0
    total_reward_2 = 0
    step = 0
    
    # Run episode
    done = False
    while not done:
        # Get actions from both agents
        action_1 = agent_1.get_action(obs["agent_1"])
        action_2 = agent_2.get_action(obs["agent_2"])
        action = np.array([action_1, action_2])
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Accumulate rewards
        total_reward_1 += reward["agent_1"]
        total_reward_2 += reward["agent_2"]
        step += 1
        
        # Print every 20 steps
        if step % 20 == 0:
            print(f"Step {step}:")
            print(f"  V1: pos=({info['state_1'][0]:.1f}, {info['state_1'][1]:.1f})")
            print(f"  V2: pos=({info['state_2'][0]:.1f}, {info['state_2'][1]:.1f})")
        
        done = terminated or truncated
    
    # Print final results
    print("\n" + "=" * 50)
    print("Episode Complete!")
    print("=" * 50)
    print(f"Total steps: {step}")
    print(f"Collision occurred: {info.get('collision', False)}")
    print(f"Vehicle 1 reached goal: {info.get('done_1', False)}")
    print(f"Vehicle 2 reached goal: {info.get('done_2', False)}")
    print(f"Total reward - V1: {total_reward_1:.2f}, V2: {total_reward_2:.2f}")


if __name__ == "__main__":
    run_simple_example()
