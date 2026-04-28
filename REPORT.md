# Social Value Orientation for Empirical Game-Theoretic Autonomous Driving

## Abstract

This project studies two-vehicle autonomous intersection negotiation as a multi-agent strategic decision problem. The central question is whether Social Value Orientation (SVO), used as a reward-shaping mechanism for learned driving agents, improves safety and coordination relative to selfish learned agents and empirical game-theoretic baselines. The system combines a randomized intersection environment, tabular Q-learning, SVO-shaped updates, parameterized heuristic policies, empirical best-response search, and approximate pure-strategy equilibrium analysis over a finite policy universe.

The main result is that small asymmetric SVO improves learned policy robustness on unseen scenarios. The best learned self-play policy, `asymmetric_svo_5`, achieves 98.7% unseen success with a 1.2% collision rate, compared with 94.8% unseen success and a 5.0% collision rate for both the selfish and non-SVO learned baselines. In the empirical game, adding SVO policies shifts the equilibrium from `yield_early` vs `aggressive_fast` to `yield_early` vs `asymmetric_svo_main`, increasing equilibrium joint utility from 96.91 to 112.36 while maintaining zero measured exploitability in the finite empirical game.

## 1. Introduction

Autonomous intersection negotiation is not simply a control problem. Each vehicle's best action depends on what the other vehicle is likely to do, how aggressively it intends to enter the conflict zone, and whether it will yield under uncertainty. This makes the setting naturally game-theoretic: each agent's behavior is strategically coupled to the other agent's behavior.

Classical heuristic and game-theoretic driving policies can encode useful interaction patterns such as yielding, priority taking, and cautious braking. However, fixed policies can be brittle when scenario distributions shift or when the other vehicle comes from a heterogeneous population. Purely selfish learned agents can also struggle because safe interaction often requires implicitly valuing the other agent's outcome.

This project evaluates SVO as an inductive bias for learning interaction-aware autonomous driving policies. SVO changes the reward used during Q-learning by blending an agent's own reward with the other agent's reward. The goal is not to make agents uniformly altruistic, but to test whether small asymmetric social preferences can produce safer and more robust strategic behavior.

The project makes four main contributions:

- It builds a randomized two-vehicle intersection environment with distinct seen and unseen scenario distributions.
- It trains selfish, non-SVO, and asymmetric SVO Q-learning agents under a shared training budget.
- It constructs an empirical game over parameterized heuristic policies and learned RL/SVO policies.
- It compares empirical games with and without SVO candidates using the same payoff matrix, regret computation, and exploitability analysis.

## 2. Problem Setting

The task is two-vehicle intersection negotiation. Vehicle 1 approaches along one road and Vehicle 2 approaches along a perpendicular road. Each vehicle controls longitudinal acceleration and must reach its goal while avoiding collision. The scenario captures a common autonomous driving conflict: two agents must coordinate access to a shared intersection without centralized communication.

Each agent observes its own state, the other vehicle's state, distance-to-goal, distance-to-other, and a priority hint. The environment randomizes spawn distance, initial speed, arrival timing, intersection center, lane offsets, priority assignment, observation noise, and partial occlusion. The train and test splits are intentionally different so that generalization can be measured rather than assumed.

The core evaluation criteria are:

- Success: both vehicles reach their goals without collision.
- Collision: any overlap between vehicle bounding boxes.
- Efficiency: average time-to-clear and delay relative to an unimpeded trajectory.
- Safety: minimum time-to-collision and safety margin.
- Smoothness: average jerk from acceleration changes.
- Fairness: similarity between the two agents' episode returns.
- Generalization: performance gap between seen training-distribution scenarios and unseen test-distribution scenarios.

## 3. Method

### 3.1 Environment

The environment is a two-agent intersection simulator with deterministic vehicle dynamics and randomized reset-time scenarios. The unseen split is intentionally harder than the seen split. It includes denser arrival overlap, broader speed variation, stronger priority asymmetry, geometry variation, observation noise, speed noise, and partial occlusion.

This design is important because a policy that works only on the seen split may have learned a narrow timing pattern rather than a robust negotiation strategy. The harder unseen split creates a more meaningful test of strategic behavior.

### 3.2 Learning

Both agents use tabular Q-learning. The observation is discretized into a compact state representation based on distance to goal, distance to the other vehicle, own speed, intersection phase, and agent identity. The action set contains discrete acceleration commands in the continuous acceleration range.

The non-SVO baseline updates each agent using only its own environment reward. The selfish SVO setting uses an SVO angle of 0 degrees, which is equivalent to caring only about the agent's own reward. Asymmetric SVO policies assign Agent 1 a positive SVO angle while Agent 2 remains selfish.

### 3.3 SVO Reward Shaping

SVO transforms the reward used for learning:

```text
r_svo = r_self * cos(phi) + r_other * sin(phi)
```

Here `phi` is the SVO angle. A 0 degree angle is selfish, while larger positive angles place increasing weight on the other agent's reward. This project focuses on asymmetric SVO with Agent 1 using angles `5, 10, 15, 20, 22.5, 30, 45` degrees and Agent 2 remaining selfish.

This asymmetry is deliberate. Symmetric prosociality can cause both agents to over-yield or create unstable coordination. Asymmetric SVO tests whether a modest amount of other-regarding behavior in one agent can improve interaction without eliminating decisive action.

### 3.4 Baselines

The experiments compare against:

- Selfish learned agents.
- A non-SVO learned RL baseline with the same model capacity and training budget.
- Parameterized heuristic policies, including aggressive, cautious, yielding, priority-based, and constant-speed variants.
- Approximate Nash profiles over the empirical policy universe.
- Empirical best-response references.

## 4. Empirical Game Construction

The empirical game is built from a shared policy universe. Each policy can be selected by either player, and the payoff for each ordered policy pair is estimated through cross-play in the environment.

The policy universe includes:

- Parameterized heuristic variants such as `aggressive_fast`, `yield_early`, and `priority_strict`.
- Learned selfish and non-SVO RL policies.
- Learned asymmetric SVO policies.

For each policy pair, the evaluator records driving metrics and converts them into a multi-objective utility. This utility prioritizes success and collision avoidance, then includes efficiency, safety, smoothness, fairness, and reward. The same payoff matrix is used for equilibrium search, best-response computation, regret tables, and exploitability.

Two empirical games are compared:

- Without SVO: the policy universe excludes SVO learned policies.
- With SVO: the policy universe includes asymmetric SVO learned policies.

This comparison isolates the strategic impact of allowing SVO policies into the game.

## 5. Experimental Design

The experiment uses three random seeds: `0, 1, 2`. Each trained configuration is evaluated on both seen and unseen scenario splits. The main learned configurations are:

- `selfish_vs_selfish`
- `non_svo_rl_baseline`
- `asymmetric_svo_5`
- `asymmetric_svo_10`
- `asymmetric_svo_main` / `asymmetric_svo_15`
- `asymmetric_svo_20`
- `asymmetric_svo_22.5`
- `asymmetric_svo_30`
- `asymmetric_svo_45`

The primary metrics are unseen success and unseen collision. These are the most direct indicators of whether the policy can negotiate the intersection safely under distribution shift. Secondary metrics include fairness, time-to-clear, joint utility, regret, and exploitability.

Generated analysis assets are available under `results/summary_plots/`, including:

- `overview_table.png`
- `svo_sweep_summary.png`
- `cross_play_heatmap_unseen.png`
- `empirical_game_comparison.png`
- `with_svo_payoff_heatmap.png`

## 6. Main Results

### 6.1 Learned Policy Performance

The best learned self-play policy on the unseen split is `asymmetric_svo_5`:

| Policy | Unseen Success | Unseen Collision | Fairness | Joint Utility |
| --- | ---: | ---: | ---: | ---: |
| `asymmetric_svo_5` | 98.7% | 1.2% | 0.958 | 95.77 |
| `asymmetric_svo_10` | 97.3% | 2.3% | 0.962 | 97.99 |
| `asymmetric_svo_main` / `asymmetric_svo_15` | 96.2% | 3.2% | 0.955 | 99.38 |
| `selfish_vs_selfish` | 94.8% | 5.0% | 0.963 | 99.44 |
| `non_svo_rl_baseline` | 94.8% | 5.0% | 0.963 | 99.44 |
| `asymmetric_svo_45` | 92.2% | 6.8% | 0.953 | 83.76 |

Small asymmetric SVO improves the most important safety and completion metrics. The 5 degree policy reduces the collision rate from 5.0% to 1.2% relative to the selfish and non-SVO learned baselines. The 10 degree policy is also strong and achieves the best learned-method joint utility in the direct unseen comparison.

The results also show that more SVO is not automatically better. At 45 degrees, success falls to 92.2% and collision rises to 6.8%. This supports the interpretation that mild asymmetric social preference is useful, while stronger prosocial weighting can destabilize the learned driving behavior.

### 6.2 Direct Method Ranking

In the direct unseen method comparison, `empirical_best_response` has the highest joint utility at 104.66. This is an oracle-style reference rather than a deployable learned policy, so the learned-only ranking is more relevant for assessing trained agents.

The top learned methods in direct unseen comparison are:

| Learned Method | Success | Collision | Fairness | Joint Utility |
| --- | ---: | ---: | ---: | ---: |
| `asymmetric_svo_10` | 97.8% | 1.7% | 0.957 | 102.47 |
| `asymmetric_svo_5` | 99.0% | 1.0% | 0.961 | 101.32 |
| `selfish_vs_selfish` | 94.8% | 5.0% | 0.963 | 99.44 |
| `asymmetric_svo_main` | 96.0% | 3.7% | 0.960 | 98.93 |
| `non_svo_rl_baseline` | 93.2% | 6.3% | 0.962 | 92.96 |

This table supports the central result: small asymmetric SVO is competitive with or better than selfish learned policies under the multi-objective driving utility.

### 6.3 Cross-Play Robustness

Cross-play is important because real traffic is heterogeneous. A policy that only succeeds against itself may fail against a different style of driver.

For the strongest learned self-play policy, `asymmetric_svo_5`, unseen cross-play remains strong against several key opponents:

| Opponent | Success | Collision |
| --- | ---: | ---: |
| `asymmetric_svo_10` | 99.0% | 0.7% |
| `empirical_best_response` | 98.8% | 1.2% |
| `approx_nash` | 98.7% | 1.2% |
| `priority_vs_yield` | 93.2% | 6.8% |
| `asymmetric_svo_main` | 89.3% | 10.7% |

The strong cross-play against `approx_nash` and `empirical_best_response` is particularly relevant to the project goal. It indicates that the best SVO policy is not merely overfit to SVO self-play; it remains robust against game-theoretic reference behavior.

## 7. Game-Theoretic Findings

The clearest game-theoretic result comes from comparing empirical games with and without SVO policies in the candidate universe.

| Game | Empirical Equilibrium | Joint Utility | Exploitability |
| --- | --- | ---: | ---: |
| Without SVO | `yield_early` vs `aggressive_fast` | 96.91 | 0.00 |
| With SVO | `yield_early` vs `asymmetric_svo_main` | 112.36 | 0.00 |

Adding SVO policies changes the empirical equilibrium and increases joint utility by 15.45. Both equilibria have zero exploitability within their finite empirical games, meaning no candidate policy in the same universe has a profitable unilateral deviation under the measured payoff matrix.

This is the strongest evidence that SVO is not only improving isolated learned policies. It changes the strategic structure of the empirical game. Once SVO policies are available, the equilibrium selects an SVO learned policy as one side of the interaction and reaches a substantially higher joint utility.

## 8. Interpretation

The results support a specific, bounded claim: small asymmetric SVO improves learned coordination and improves the empirical game equilibrium. The data does not support a broader claim that stronger prosociality always improves driving behavior.

The useful regime appears to be mild SVO, especially around 5 to 10 degrees. These settings reduce collisions and improve unseen success while preserving decisive behavior. Larger angles can make the agent overweight the other vehicle's reward, which may produce inefficient or unstable negotiation.

The best way to frame the findings is:

> Small asymmetric SVO acts as a useful social inductive bias for autonomous intersection negotiation. It improves learned safety under unseen scenarios and shifts the empirical game toward a higher-joint-utility equilibrium.

This framing is stronger than claiming that SVO dominates every baseline. It accounts for the empirical best-response reference scoring highest overall while still identifying SVO as the best deployable learned-policy improvement.

## 9. Implications

The project suggests that social preference can be a useful design element in multi-agent autonomous driving. A small amount of other-regarding reward shaping can reduce collision risk without requiring hand-coded yielding rules.

The empirical game analysis also demonstrates why self-play alone is insufficient. The most important evidence comes from cross-play, best-response comparison, and equilibrium analysis. These tools reveal whether learned policies behave well in a broader strategic population.

For autonomous driving, the practical implication is that learned policies should be evaluated as participants in a traffic population, not only as isolated controllers. SVO provides one mechanism for improving population-level outcomes.

## 10. Limitations

The study remains limited in several ways.

First, the environment contains two vehicles and a single primary conflict point. This is useful for controlled analysis, but real traffic includes more agents, lane changes, signaling, pedestrians, and richer intent uncertainty.

Second, the learned agents use tabular Q-learning. This makes the behavior interpretable and easy to compare, but it limits representational capacity. Stronger deep RL or multi-agent RL baselines may change the relative ranking.

Third, the empirical game is finite. Equilibrium and exploitability are measured only relative to the policies included in the candidate universe. A zero exploitability value means no profitable deviation was found among the available candidates, not that the profile is analytically unexploitable.

Fourth, empirical best response is an oracle-style reference. It is useful for analysis but should not be presented as equivalent to a deployable learned policy.

Finally, the current results include duplicate configurations: `asymmetric_svo_main` and `asymmetric_svo_15` represent the same SVO angle. Future reporting should collapse these for clarity.

## 11. Conclusion

This project demonstrates a working empirical game-theoretic framework for autonomous intersection negotiation and uses it to evaluate SVO-shaped learning. The results show that small asymmetric SVO improves learned policy robustness under harder unseen scenarios. The best learned self-play policy, `asymmetric_svo_5`, improves unseen success from 94.8% to 98.7% and reduces collision from 5.0% to 1.2% relative to selfish and non-SVO learned baselines.

The game-theoretic result is even more important. Adding SVO policies to the empirical policy universe changes the equilibrium from `yield_early` vs `aggressive_fast` to `yield_early` vs `asymmetric_svo_main`, increasing equilibrium joint utility by 15.45. This supports the claim that SVO affects strategic interaction, not just isolated reward optimization.

The main conclusion is that mild asymmetric SVO is a promising social inductive bias for multi-agent autonomous driving. It improves safety and coordination when applied carefully, but stronger prosociality can degrade performance.

## 12. Future Work

Future work should expand both the learning method and the empirical game.

The most direct extension is to include richer policy candidates: intermediate RL checkpoints, more parameterized heuristic variants, and policies trained under additional SVO schedules. This would make the empirical game more representative and reduce dependence on a small finite candidate set.

The equilibrium analysis should also move beyond pure strategies. Mixed-strategy equilibria would better reflect heterogeneous traffic, where drivers do not all follow the same deterministic policy.

The learning side should be upgraded from tabular Q-learning to function approximation or deep multi-agent RL. This would allow more expressive policies, richer observations, and more realistic scenario distributions.

The environment should be extended to more than two vehicles, multi-lane geometry, pedestrian conflicts, explicit signaling, and stronger partial observability. These additions would test whether SVO remains useful under more realistic traffic complexity.

Finally, future systems could learn or adapt SVO online. A fixed angle is a useful starting point, but real driving may require context-dependent social preference: more assertive behavior in low-risk settings and more cooperative behavior under dense or uncertain traffic.

## Figures and Artifacts

The following generated artifacts support the report:

- `results/summary_plots/overview_table.png`
- `results/summary_plots/svo_sweep_summary.png`
- `results/summary_plots/cross_play_heatmap_unseen.png`
- `results/summary_plots/empirical_game_comparison.png`
- `results/summary_plots/with_svo_payoff_heatmap.png`
- `results/summary_plots/experiment_summary_report.md`
- `results/summary_plots/presentation_summary.md`
