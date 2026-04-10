"""
Comparison framework for evaluating and contrasting agent approaches.

Usage
-----
Pass a dict of {label: (agent_1, agent_2)} to compare_methods().
Agents can be any object with a get_action(obs) method —
HeuristicAgent instances from agents.heuristic or trained RLAgent
instances from agents.rl_agent both work.

Example
-------
    from agents.heuristic import create_heuristic_pair
    from training.train_rl import train_rl_agent
    from env.rewards import SVO_PROSOCIAL, SVO_SELFISH
    from evaluation.compare_methods import compare_methods

    rl_1, rl_2, _ = train_rl_agent(num_episodes=2000, verbose=False)

    results = compare_methods(
        num_eval_episodes=200,
        method_dict={
            "cautious vs aggressive": create_heuristic_pair("cautious", "aggressive"),
            "yield vs constant":      create_heuristic_pair("yield", "constant"),
            "RL (prosocial vs selfish)": (rl_1, rl_2),
        },
    )
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from typing import Dict, List, Tuple, Any

from env.intersection_env import IntersectionEnv
from agents.heuristic import create_heuristic_pair
from evaluation.metrics import (
    compute_success_rate,
    compute_collision_rate,
    compute_efficiency,
)


def run_eval_episodes(
    agent_1: Any,
    agent_2: Any,
    env: IntersectionEnv,
    n_episodes: int = 100,
) -> List[Dict]:
    """
    Run n_episodes greedy evaluation episodes and return per-episode results.

    Both HeuristicAgent (get_action(obs)) and RLAgent
    (get_action(obs, training=False)) are supported; the call always uses
    training=False where applicable.

    Args:
        agent_1: Agent for vehicle 1.
        agent_2: Agent for vehicle 2.
        env: Intersection environment instance.
        n_episodes: Number of episodes to run.

    Returns:
        List of dicts with keys:
            collision, done_1, done_2, steps,
            total_reward_1, total_reward_2.
    """
    results = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_r1 = 0.0
        total_r2 = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            # Support both HeuristicAgent (1-arg) and RLAgent (training kwarg)
            try:
                a1 = agent_1.get_action(obs["agent_1"], training=False)
            except TypeError:
                a1 = agent_1.get_action(obs["agent_1"])

            try:
                a2 = agent_2.get_action(obs["agent_2"], training=False)
            except TypeError:
                a2 = agent_2.get_action(obs["agent_2"])

            obs, reward, terminated, truncated, info = env.step(
                np.array([a1, a2])
            )
            total_r1 += reward["agent_1"]
            total_r2 += reward["agent_2"]

        results.append({
            "collision": bool(info.get("collision", False)),
            "done_1":    bool(info.get("done_1", False)),
            "done_2":    bool(info.get("done_2", False)),
            "steps":     int(info["steps"]),
            "total_reward_1": float(total_r1),
            "total_reward_2": float(total_r2),
        })

    return results


def compare_methods(
    num_eval_episodes: int = 200,
    method_dict: Dict[str, Tuple[Any, Any]] = None,
    env: IntersectionEnv = None,
) -> Dict[str, Dict]:
    """
    Evaluate multiple agent pairs and print a comparison table.

    Args:
        num_eval_episodes: Episodes per method.
        method_dict: {label: (agent_1, agent_2)}. Defaults to four
                     heuristic baseline pairs if omitted.
        env: Optional pre-configured IntersectionEnv. If None, a default
             IntersectionEnv() is created. Pass an env when training used
             non-default params (e.g. max_steps, reward weights).

    Returns:
        Dict mapping each label to its metrics dict:
            success_rate, collision_rate, avg_time_to_clear,
            avg_delay, avg_steps.
    """
    if method_dict is None:
        method_dict = {
            "constant vs cautious":   create_heuristic_pair("constant", "cautious"),
            "cautious vs aggressive": create_heuristic_pair("cautious", "aggressive"),
            "yield vs aggressive":    create_heuristic_pair("yield", "aggressive"),
            "yield vs cautious":      create_heuristic_pair("yield", "cautious"),
        }

    if env is None:
        env = IntersectionEnv()
    all_results = {}

    for label, (agent_1, agent_2) in method_dict.items():
        episodes = run_eval_episodes(agent_1, agent_2, env, n_episodes=num_eval_episodes)

        efficiency = compute_efficiency(episodes)
        all_results[label] = {
            "success_rate":      compute_success_rate(episodes),
            "collision_rate":    compute_collision_rate(episodes),
            "avg_time_to_clear": efficiency["avg_time_to_clear"],
            "avg_delay":         efficiency["avg_delay"],
            "avg_steps":         efficiency["avg_steps"],
        }

    _print_table(all_results)
    return all_results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _print_table(results: Dict[str, Dict]) -> None:
    col_w = max(len(label) for label in results) + 2

    header = (
        f"{'Method':<{col_w}} "
        f"{'Success':>8} "
        f"{'Collision':>10} "
        f"{'T-clear(s)':>11} "
        f"{'Delay(s)':>9} "
        f"{'AvgSteps':>9}"
    )
    sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)

    for label, m in results.items():
        t_clear = f"{m['avg_time_to_clear']:.2f}" if m["avg_time_to_clear"] is not None else "  N/A"
        print(
            f"{label:<{col_w}} "
            f"{m['success_rate']:>8.2%} "
            f"{m['collision_rate']:>10.2%} "
            f"{t_clear:>11} "
            f"{m['avg_delay']:>9.2f} "
            f"{m['avg_steps']:>9.1f}"
        )

    print(sep)


if __name__ == "__main__":
    print(f"Evaluating heuristic baselines (200 episodes each)...\n")
    compare_methods(num_eval_episodes=200)
