"""
Comparison framework for evaluating and contrasting agent approaches.
"""

from __future__ import annotations

import os
import sys
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.heuristic import create_heuristic_pair
from env.intersection_env import IntersectionEnv
from evaluation.metrics import summarize_episodes


def _clone_env(env: IntersectionEnv) -> IntersectionEnv:
    return deepcopy(env)


def _get_action(agent: Any, obs: np.ndarray) -> float:
    try:
        return float(agent.get_action(obs, training=False))
    except TypeError:
        return float(agent.get_action(obs))


def run_eval_episodes(
    agent_1: Any,
    agent_2: Any,
    env: IntersectionEnv,
    n_episodes: int = 100,
    seed: int = 0,
    scenario_split: str | None = None,
) -> List[Dict]:
    """
    Run evaluation episodes and return rich per-episode traces.
    """
    results = []

    for episode_idx in range(n_episodes):
        reset_options = {}
        if scenario_split is not None:
            reset_options["scenario_split"] = scenario_split

        if episode_idx == 0:
            obs, info = env.reset(seed=seed, options=reset_options or None)
        else:
            obs, info = env.reset(options=reset_options or None)

        if hasattr(agent_1, "start_episode"):
            agent_1.start_episode(env.np_random)
        if hasattr(agent_2, "start_episode"):
            agent_2.start_episode(env.np_random)

        total_r1 = 0.0
        total_r2 = 0.0
        terminated = False
        truncated = False
        states_1 = [info["state_1"].copy().tolist()]
        states_2 = [info["state_2"].copy().tolist()]
        actions_1: List[float] = []
        actions_2: List[float] = []

        while not (terminated or truncated):
            a1 = _get_action(agent_1, obs["agent_1"])
            a2 = _get_action(agent_2, obs["agent_2"])
            actions_1.append(a1)
            actions_2.append(a2)

            obs, reward, terminated, truncated, info = env.step(np.array([a1, a2]))
            total_r1 += reward["agent_1"]
            total_r2 += reward["agent_2"]
            states_1.append(info["state_1"].copy().tolist())
            states_2.append(info["state_2"].copy().tolist())

        results.append(
            {
                "collision": bool(info.get("collision", False)),
                "done_1": bool(info.get("done_1", False)),
                "done_2": bool(info.get("done_2", False)),
                "steps": int(info["steps"]),
                "total_reward_1": float(total_r1),
                "total_reward_2": float(total_r2),
                "actions_1": actions_1,
                "actions_2": actions_2,
                "states_1": states_1,
                "states_2": states_2,
                "scenario": info.get("scenario"),
            }
        )

    return results


def compare_methods(
    num_eval_episodes: int = 200,
    method_dict: Dict[str, Tuple[Any, Any]] | None = None,
    env: IntersectionEnv | None = None,
    seed: int = 0,
    scenario_split: str | None = None,
    utility_weights: Dict[str, float] | None = None,
) -> Dict[str, Dict]:
    """
    Evaluate multiple agent pairs and return a metrics table.
    """
    if method_dict is None:
        method_dict = {
            "constant vs cautious": create_heuristic_pair("constant", "cautious"),
            "cautious vs aggressive": create_heuristic_pair("cautious", "aggressive"),
            "yield vs aggressive": create_heuristic_pair("yield", "aggressive"),
            "priority vs yield": create_heuristic_pair("priority", "yield"),
        }

    env = env or IntersectionEnv()
    all_results: Dict[str, Dict] = {}

    for idx, (label, (agent_1, agent_2)) in enumerate(method_dict.items()):
        eval_env = _clone_env(env)
        episodes = run_eval_episodes(
            agent_1,
            agent_2,
            eval_env,
            n_episodes=num_eval_episodes,
            seed=seed + idx * 1000,
            scenario_split=scenario_split,
        )
        all_results[label] = summarize_episodes(episodes, dt=env.dt, utility_weights=utility_weights)

    _print_table(all_results)
    return all_results


def evaluate_cross_play(
    row_agents: Dict[str, Tuple[Any, Any]],
    col_agents: Dict[str, Tuple[Any, Any]],
    env: IntersectionEnv,
    n_episodes: int = 100,
    seed: int = 0,
    scenario_split: str | None = None,
    utility_weights: Dict[str, float] | None = None,
) -> Dict[str, Dict[str, Dict]]:
    """
    Cross-play agent_1 from each row pair against agent_2 from each column pair.
    """
    matrix: Dict[str, Dict[str, Dict]] = {}
    for row_idx, (row_label, (row_agent_1, _)) in enumerate(row_agents.items()):
        matrix[row_label] = {}
        for col_idx, (col_label, (_, col_agent_2)) in enumerate(col_agents.items()):
            eval_env = _clone_env(env)
            episodes = run_eval_episodes(
                row_agent_1,
                col_agent_2,
                eval_env,
                n_episodes=n_episodes,
                seed=seed + row_idx * 1000 + col_idx * 17,
                scenario_split=scenario_split,
            )
            matrix[row_label][col_label] = summarize_episodes(
                episodes,
                dt=env.dt,
                utility_weights=utility_weights,
            )
    return matrix


def _print_table(results: Dict[str, Dict]) -> None:
    col_w = max(len(label) for label in results) + 2
    header = (
        f"{'Method':<{col_w}} "
        f"{'Success':>8} "
        f"{'Collision':>10} "
        f"{'MinTTC':>8} "
        f"{'Margin':>8} "
        f"{'Jerk':>8}"
        f" {'MO-U':>8}"
    )
    sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)

    for label, m in results.items():
        print(
            f"{label:<{col_w}} "
            f"{m['success_rate']:>8.2%} "
            f"{m['collision_rate']:>10.2%} "
            f"{m['min_time_to_collision']:>8.2f} "
            f"{m['safety_margin']:>8.2f} "
            f"{m['avg_jerk']:>8.2f} "
            f"{m['multi_objective_utility_joint']:>8.2f}"
        )

    print(sep)
