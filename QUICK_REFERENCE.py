"""
Quick Reference Guide for marl-av Project
=========================================

RUNNING DEMOS
-------------
python test_basic.py              # Run validation tests
python example_usage.py           # Simple usage example
python demo/render_episode.py     # Full demo with visualization

CREATING ENVIRONMENT
-------------------
from env import IntersectionEnv

env = IntersectionEnv(
    dt=0.1,              # Time step (seconds)
    max_steps=200,       # Episode length
    goal_threshold=2.0,  # Distance to goal (meters)
    render_mode=None     # 'human' or 'rgb_array'
)

CREATING AGENTS
--------------
from agents import create_heuristic_pair

agent_1, agent_2 = create_heuristic_pair(
    policy_1="cautious",  # Options: constant, cautious, aggressive, yield
    policy_2="yield"
)

RUNNING EPISODE
--------------
import numpy as np

obs, info = env.reset()
done = False

while not done:
    # Get actions
    action_1 = agent_1.get_action(obs["agent_1"])
    action_2 = agent_2.get_action(obs["agent_2"])
    action = np.array([action_1, action_2])
    
    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

# Check results
print(f"Collision: {info['collision']}")
print(f"V1 reached goal: {info['done_1']}")
print(f"V2 reached goal: {info['done_2']}")

OBSERVATION SPACE
----------------
Each agent observes 10D vector:
[own_x, own_y, own_vx, own_vy,     # Own state (4D)
 other_x, other_y, other_vx, other_vy,  # Other state (4D)
 goal_distance,                     # Distance to own goal (1D)
 other_distance]                    # Distance to other vehicle (1D)

ACTION SPACE
-----------
Each agent controls acceleration: [-1, 1]
 -1: Full braking (5 m/s²)
  0: Maintain speed
  1: Full acceleration (3 m/s²)

REWARD STRUCTURE
---------------
- Collision: -100 (terminal)
- Goal reached: +50
- Time penalty: -0.1 per step
- Efficiency bonus: +0.5 * (speed / max_speed)
- Proximity penalty: if distance < 5m

HEURISTIC AGENTS
---------------
ConstantVelocityAgent  - Maintains constant speed (8 m/s)
CautiousAgent         - Slows down when other nearby
AggressiveAgent       - High speed, brakes only if critical
YieldAgent            - Yields to other at intersection

CUSTOMIZATION
------------
# Custom dynamics
from env.dynamics import VehicleDynamics
dynamics = VehicleDynamics(dt=0.1)
dynamics.max_velocity = 20.0  # Change max speed
dynamics.max_acceleration = 4.0

# Custom rewards
from env.rewards import RewardFunction
reward_fn = RewardFunction(
    collision_penalty=-200.0,  # Bigger penalty
    goal_reward=100.0,
    time_penalty=-0.2
)

# Custom heuristic agent
from agents.heuristic import HeuristicAgent
class MyAgent(HeuristicAgent):
    def get_action(self, observation):
        # Your custom logic here
        return 0.0  # Return action in [-1, 1]

VISUALIZATION
------------
from demo.render_episode import plot_trajectory, run_episode

states_1, states_2, rewards, info = run_episode(env, agent_1, agent_2)
plot_trajectory(states_1, states_2, env, info, save_path="results.png")

KEY PARAMETERS
-------------
Vehicle:
- Length: 4.5m
- Width: 2.0m
- Max speed: 15 m/s (~54 km/h)
- Max accel: 3 m/s²
- Max brake: 5 m/s²

Environment:
- Intersection at origin (0, 0)
- V1: starts at (-20, 0), goal at (20, 0)
- V2: starts at (0, -20), goal at (0, 20)
- Initial speed: 8 m/s both
- Goal threshold: 2.0m
- Max episode: 200 steps

FILE STRUCTURE
-------------
env/
  dynamics.py         - Vehicle physics
  rewards.py          - Reward calculation
  intersection_env.py - Main environment
agents/
  heuristic.py        - Baseline agents
demo/
  render_episode.py   - Visualization
"""

if __name__ == "__main__":
    print(__doc__)
