# marl-av

Multi-Agent Reinforcement Learning for Autonomous Vehicle Interaction

## Overview

This project compares game-theoretic and reinforcement learning approaches to autonomous vehicle interaction at intersections. The implementation provides a clean, modular simulation environment using Python and Gymnasium.

## Features

- **Two-Vehicle Intersection Environment**: Vehicles approach from perpendicular directions
- **Deterministic Vehicle Dynamics**: Simple kinematic model with acceleration control
- **Reward Structure**: Balances collision avoidance, efficiency, and goal achievement
- **Heuristic Agents**: Multiple baseline policies for testing and comparison
- **Visualization Tools**: Plot trajectories and analyze episode outcomes

## Project Structure

```
marl-av/
├── env/
│   ├── intersection_env.py   # Gymnasium environment
│   ├── dynamics.py            # Vehicle dynamics model
│   └── rewards.py             # Reward function
├── agents/
│   ├── heuristic.py           # Heuristic baseline agents
│   ├── rl_agent.py            # RL agent (future work)
│   └── gradient_solver.py     # Game-theoretic solver (future work)
├── demo/
│   └── render_episode.py      # Visualization demo
├── training/
│   └── train_rl.py            # Training script (future work)
├── evaluation/
│   ├── compare_methods.py     # Comparison framework (future work)
│   └── metrics.py             # Evaluation metrics (future work)
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd marl-av

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Run Demo with Heuristic Agents

```bash
python demo/render_episode.py
```

This runs multiple scenarios with different agent pairs:
- Both constant velocity
- Both cautious
- Aggressive vs cautious
- Yield vs constant

### Use the Environment Programmatically

```python
from env.intersection_env import IntersectionEnv
from agents.heuristic import create_heuristic_pair
import numpy as np

# Create environment
env = IntersectionEnv(dt=0.1, max_steps=200)

# Create agents
agent_1, agent_2 = create_heuristic_pair("cautious", "cautious")

# Run episode
obs, info = env.reset()
done = False

while not done:
    action_1 = agent_1.get_action(obs["agent_1"])
    action_2 = agent_2.get_action(obs["agent_2"])
    action = np.array([action_1, action_2])
    
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

print(f"Episode finished: Collision={info['collision']}")
```

## Environment Details

### Observation Space

Each agent observes (10-dimensional vector):
- Own position (x, y)
- Own velocity (vx, vy)
- Other vehicle position (x, y)
- Other vehicle velocity (vx, vy)
- Distance to own goal
- Distance to other vehicle

### Action Space

Continuous acceleration control in [-1, 1]:
- -1: Maximum braking (5 m/s²)
- 0: Maintain speed
- 1: Maximum acceleration (3 m/s²)

### Reward Structure

- **Collision**: -100 (terminal)
- **Goal reached**: +50
- **Time penalty**: -0.1 per step
- **Efficiency bonus**: +0.5 * (speed / max_speed)
- **Proximity penalty**: Scaled by distance when < 5m

### Vehicle Dynamics

- Maximum speed: 15 m/s (~54 km/h)
- Maximum acceleration: 3 m/s²
- Maximum braking: 5 m/s²
- Vehicle dimensions: 4.5m × 2.0m
- Time step: 0.1s (configurable)

## Heuristic Agent Types

1. **ConstantVelocityAgent**: Maintains constant speed
2. **CautiousAgent**: Slows down when other vehicle is nearby
3. **AggressiveAgent**: Maintains high speed, only brakes if collision imminent
4. **YieldAgent**: Yields to other vehicle at intersection

## Future Work

- [ ] Implement RL training pipeline
- [ ] Add game-theoretic solver
- [ ] Comprehensive evaluation metrics
- [ ] Multi-scenario testing
- [ ] More complex intersection geometries

## License

MIT License

## Contributing

Contributions welcome! Please feel free to submit issues or pull requests.
