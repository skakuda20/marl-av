# marl-av

Multi-agent RL and empirical game-theoretic evaluation for two-vehicle intersection negotiation.

## Changes

- Goal rewards are now transition-based: each agent gets the goal reward once, when it first reaches the goal.
- `IntersectionEnv.reset()` now randomizes scenario parameters with deterministic seeds:
  spawn distance, initial speed, arrival offset, intersection center, and a per-agent priority hint.
- Train and test scenario distributions are split (`train` vs `test`) so generalization gaps are measurable.
- Evaluation now reports cross-play, minimum TTC, safety margin, jerk, reward fairness, seen-vs-unseen gap, and regret against fixed opponents.
- `run_experiment.py` runs repeated seeds, SVO ablations, symmetric/asymmetric pairings, empirical best responses, and an approximate Nash baseline over the heuristic policy set.

## Quick Start

```bash
python3 test_basic.py
python3 test_suite.py
python3 run_experiment.py
```

## Main Components

- [env/intersection_env.py](/home/kakudas/GitHub/marl-av/env/intersection_env.py)
  Randomized Gymnasium environment with seeded train/test splits.
- [env/scenarios.py](/home/kakudas/GitHub/marl-av/env/scenarios.py)
  Scenario distribution definitions and reset-time sampling.
- [training/train_rl.py](/home/kakudas/GitHub/marl-av/training/train_rl.py)
  Seeded tabular Q-learning with optional SVO shaping.
- [evaluation/compare_methods.py](/home/kakudas/GitHub/marl-av/evaluation/compare_methods.py)
  Self-play and cross-play evaluation with trace collection.
- [evaluation/metrics.py](/home/kakudas/GitHub/marl-av/evaluation/metrics.py)
  Efficiency, safety, smoothness, fairness, and generalization metrics.
- [agents/gradient_solver.py](/home/kakudas/GitHub/marl-av/agents/gradient_solver.py)
  Empirical best-response and approximate Nash search over the heuristic policy set.

## Scenarios

The observation is now 11-dimensional:

1. own position `(x, y)`
2. own velocity `(vx, vy)`
3. other position `(x, y)`
4. other velocity `(vx, vy)`
5. distance to own goal
6. distance to the other vehicle
7. own priority hint in `{-1, 0, 1}`

## Baselines

Supported heuristic/game baselines include:

- `constant`
- `cautious`
- `aggressive`
- `yield`
- `priority`
- empirical best response over the heuristic set
- approximate Nash over the heuristic set

## Outputs

`run_experiment.py` writes [results/experiment_summary.json](/home/kakudas/GitHub/marl-av/results/experiment_summary.json) containing:

- per-seed training/evaluation results
- mean/std aggregates across seeds
- seen vs unseen self-play metrics
- seen vs unseen cross-play against heuristics, mixtures, and RL ablations
- regret against fixed opponents
- empirical game-theoretic baseline summaries
