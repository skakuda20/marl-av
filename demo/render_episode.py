"""
Visualization demo for the intersection environment.
Runs episodes with heuristic agents and displays results.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from typing import List, Tuple

from env.intersection_env import IntersectionEnv
from agents.heuristic import create_heuristic_pair


def run_episode(
    env: IntersectionEnv,
    agent_1,
    agent_2,
    render: bool = False
) -> Tuple[List, List, List, dict]:
    """
    Run one episode with given agents.
    
    Args:
        env: The intersection environment
        agent_1: First agent (heuristic or learned)
        agent_2: Second agent (heuristic or learned)
        render: Whether to print state during episode
    
    Returns:
        Tuple of (states_1, states_2, rewards, info)
    """
    obs, info = env.reset()
    
    states_1 = [info["state_1"].copy()]
    states_2 = [info["state_2"].copy()]
    rewards_1 = []
    rewards_2 = []
    
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        # Get actions from agents
        action_1 = agent_1.get_action(obs["agent_1"])
        action_2 = agent_2.get_action(obs["agent_2"])
        action = np.array([action_1, action_2])
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Store data
        states_1.append(info["state_1"].copy())
        states_2.append(info["state_2"].copy())
        rewards_1.append(reward["agent_1"])
        rewards_2.append(reward["agent_2"])
        
        if render:
            env.render()
    
    episode_info = {
        "collision": info.get("collision", False),
        "done_1": info.get("done_1", False),
        "done_2": info.get("done_2", False),
        "total_reward_1": sum(rewards_1),
        "total_reward_2": sum(rewards_2),
        "steps": info["steps"]
    }
    
    return states_1, states_2, (rewards_1, rewards_2), episode_info


def plot_trajectory(
    states_1: List[np.ndarray],
    states_2: List[np.ndarray],
    env: IntersectionEnv,
    episode_info: dict,
    save_path: str = None
):
    """
    Plot the trajectories of both vehicles.
    
    Args:
        states_1: List of states for vehicle 1
        states_2: List of states for vehicle 2
        env: Environment instance for parameters
        episode_info: Information about the episode
        save_path: Optional path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Extract positions
    pos_1 = np.array([[s[0], s[1]] for s in states_1])
    pos_2 = np.array([[s[0], s[1]] for s in states_2])
    
    # Plot 1: Trajectories
    ax1.plot(pos_1[:, 0], pos_1[:, 1], 'b-', linewidth=2, label='Vehicle 1', alpha=0.7)
    ax1.plot(pos_2[:, 0], pos_2[:, 1], 'r-', linewidth=2, label='Vehicle 2', alpha=0.7)
    
    # Mark start and end positions
    ax1.plot(pos_1[0, 0], pos_1[0, 1], 'bo', markersize=10, label='V1 Start')
    ax1.plot(pos_2[0, 0], pos_2[0, 1], 'ro', markersize=10, label='V2 Start')
    ax1.plot(pos_1[-1, 0], pos_1[-1, 1], 'b*', markersize=15, label='V1 End')
    ax1.plot(pos_2[-1, 0], pos_2[-1, 1], 'r*', markersize=15, label='V2 End')
    
    # Mark goals
    ax1.plot(env.goal_pos_1[0], env.goal_pos_1[1], 'b^', markersize=12, 
             label='V1 Goal', alpha=0.5)
    ax1.plot(env.goal_pos_2[0], env.goal_pos_2[1], 'r^', markersize=12, 
             label='V2 Goal', alpha=0.5)
    
    # Draw intersection zone
    intersection_circle = patches.Circle((0, 0), 5.0, fill=False, 
                                        edgecolor='gray', linestyle='--', 
                                        linewidth=2, label='Intersection')
    ax1.add_patch(intersection_circle)
    
    ax1.set_xlabel('X Position (m)', fontsize=12)
    ax1.set_ylabel('Y Position (m)', fontsize=12)
    ax1.set_title('Vehicle Trajectories', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    ax1.set_xlim(-25, 25)
    ax1.set_ylim(-25, 25)
    
    # Plot 2: Speed over time
    speeds_1 = [np.sqrt(s[2]**2 + s[3]**2) for s in states_1]
    speeds_2 = [np.sqrt(s[2]**2 + s[3]**2) for s in states_2]
    time = np.arange(len(speeds_1)) * env.dt
    
    ax2.plot(time, speeds_1, 'b-', linewidth=2, label='Vehicle 1')
    ax2.plot(time, speeds_2, 'r-', linewidth=2, label='Vehicle 2')
    ax2.axhline(y=env.dynamics.max_velocity, color='gray', linestyle='--', 
                alpha=0.5, label='Max Speed')
    
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Speed (m/s)', fontsize=12)
    ax2.set_title('Vehicle Speeds', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # Add episode information
    info_text = f"Steps: {episode_info['steps']} | "
    info_text += f"Collision: {episode_info['collision']} | "
    info_text += f"V1 Goal: {episode_info['done_1']} | "
    info_text += f"V2 Goal: {episode_info['done_2']}\n"
    info_text += f"Total Rewards - V1: {episode_info['total_reward_1']:.1f}, "
    info_text += f"V2: {episode_info['total_reward_2']:.1f}"
    
    fig.suptitle(info_text, fontsize=11, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()


def main():
    """Run demonstration episodes with different agent pairs."""
    
    # Create environment
    env = IntersectionEnv(dt=0.1, max_steps=200)
    
    # Define test scenarios
    scenarios = [
        ("constant", "constant", "Both Constant Velocity"),
        ("cautious", "cautious", "Both Cautious"),
        ("aggressive", "cautious", "Aggressive vs Cautious"),
        ("yield", "constant", "Yield vs Constant"),
    ]
    
    print("=" * 70)
    print("INTERSECTION ENVIRONMENT DEMO")
    print("=" * 70)
    
    for policy_1, policy_2, description in scenarios:
        print(f"\nScenario: {description}")
        print(f"  Agent 1: {policy_1.capitalize()}")
        print(f"  Agent 2: {policy_2.capitalize()}")
        print("-" * 70)
        
        # Create agents
        agent_1, agent_2 = create_heuristic_pair(policy_1, policy_2)
        
        # Run episode
        states_1, states_2, rewards, episode_info = run_episode(
            env, agent_1, agent_2, render=False
        )
        
        # Print results
        print(f"  Result:")
        print(f"    Steps: {episode_info['steps']}")
        print(f"    Collision: {episode_info['collision']}")
        print(f"    Vehicle 1 reached goal: {episode_info['done_1']}")
        print(f"    Vehicle 2 reached goal: {episode_info['done_2']}")
        print(f"    Total Reward V1: {episode_info['total_reward_1']:.2f}")
        print(f"    Total Reward V2: {episode_info['total_reward_2']:.2f}")
        
        # Visualize first scenario
        if policy_1 == "cautious" and policy_2 == "cautious":
            print(f"  Generating visualization...")
            plot_trajectory(states_1, states_2, env, episode_info)
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
