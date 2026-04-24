"""
Quick reference for the current marl-av workflow.

Commands
--------
python3 test_basic.py
python3 test_suite.py
python3 run_experiment.py

Environment
-----------
from env.intersection_env import IntersectionEnv
env = IntersectionEnv(scenario_split="train")
obs, info = env.reset(seed=0)

Observation layout (11D)
------------------------
[own_x, own_y, own_vx, own_vy,
 other_x, other_y, other_vx, other_vy,
 goal_distance, other_distance, own_priority]

Scenario randomization
----------------------
- spawn distance
- initial speed
- arrival offset
- intersection center
- priority hint

Useful modules
--------------
env/scenarios.py            scenario distributions
training/train_rl.py        seeded Q-learning
evaluation/metrics.py       safety/smoothness/fairness/generalization metrics
evaluation/compare_methods.py  self-play and cross-play evaluation
agents/gradient_solver.py   empirical best response and approximate Nash
"""

if __name__ == "__main__":
    print(__doc__)
