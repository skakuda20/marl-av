"""
Evaluation metrics for comparing agent approaches.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

_UNIMPEDED_STEPS = 50


def _safe_mean(values: List[float], default: float = 0.0) -> float:
    if not values:
        return float(default)
    return float(np.mean(values))


def compute_success_rate(episode_results: List[Dict]) -> float:
    if not episode_results:
        return 0.0
    successes = sum(
        1
        for r in episode_results
        if r["done_1"] and r["done_2"] and not r["collision"]
    )
    return successes / len(episode_results)


def compute_collision_rate(episode_results: List[Dict]) -> float:
    if not episode_results:
        return 0.0
    return sum(1 for r in episode_results if r["collision"]) / len(episode_results)


def compute_efficiency(episode_results: List[Dict], dt: float = 0.1) -> Dict:
    if not episode_results:
        return {"avg_time_to_clear": None, "avg_delay": 0.0, "avg_steps": 0.0}

    all_steps = [r["steps"] for r in episode_results]
    successful = [
        r for r in episode_results
        if r["done_1"] and r["done_2"] and not r["collision"]
    ]

    avg_time_to_clear = (
        float(np.mean([r["steps"] for r in successful]) * dt) if successful else None
    )
    avg_delay = float(np.mean(all_steps) - _UNIMPEDED_STEPS) * dt
    avg_steps = float(np.mean(all_steps))
    return {
        "avg_time_to_clear": avg_time_to_clear,
        "avg_delay": avg_delay,
        "avg_steps": avg_steps,
    }


def _episode_min_distance(states_1: List[List[float]], states_2: List[List[float]]) -> float:
    min_dist = np.inf
    for s1, s2 in zip(states_1, states_2):
        p1 = np.array(s1[:2], dtype=float)
        p2 = np.array(s2[:2], dtype=float)
        min_dist = min(min_dist, float(np.linalg.norm(p2 - p1)))
    return float(min_dist)


def _episode_min_ttc(states_1: List[List[float]], states_2: List[List[float]]) -> float:
    min_ttc = np.inf
    for s1, s2 in zip(states_1, states_2):
        p1 = np.array(s1[:2], dtype=float)
        p2 = np.array(s2[:2], dtype=float)
        v1 = np.array(s1[2:4], dtype=float)
        v2 = np.array(s2[2:4], dtype=float)
        rel_pos = p2 - p1
        rel_vel = v2 - v1
        dist = np.linalg.norm(rel_pos)
        if dist < 1e-6:
            return 0.0
        closing_speed = -float(np.dot(rel_pos, rel_vel) / dist)
        if closing_speed > 1e-6:
            min_ttc = min(min_ttc, dist / closing_speed)
    return float(min_ttc if np.isfinite(min_ttc) else 1e6)


def compute_safety_metrics(episode_results: List[Dict]) -> Dict[str, float]:
    if not episode_results:
        return {"min_time_to_collision": 0.0, "safety_margin": 0.0}

    min_ttc = min(
        _episode_min_ttc(r["states_1"], r["states_2"]) for r in episode_results
    )
    min_distance = min(
        _episode_min_distance(r["states_1"], r["states_2"]) for r in episode_results
    )
    return {
        "min_time_to_collision": float(min_ttc),
        "safety_margin": float(min_distance),
    }


def _mean_abs_jerk(actions: List[float], dt: float) -> float:
    if len(actions) < 3:
        return 0.0
    accelerations = np.array(actions, dtype=float)
    jerks = np.diff(accelerations) / max(dt, 1e-6)
    return float(np.mean(np.abs(jerks)))


def compute_smoothness_metrics(episode_results: List[Dict], dt: float = 0.1) -> Dict[str, float]:
    if not episode_results:
        return {"avg_jerk": 0.0}
    per_episode = []
    for r in episode_results:
        jerk_1 = _mean_abs_jerk(r["actions_1"], dt)
        jerk_2 = _mean_abs_jerk(r["actions_2"], dt)
        per_episode.append((jerk_1 + jerk_2) / 2.0)
    return {"avg_jerk": _safe_mean(per_episode)}


def compute_fairness_metrics(episode_results: List[Dict]) -> Dict[str, float]:
    if not episode_results:
        return {"reward_fairness": 0.0}
    scores = []
    for r in episode_results:
        r1 = float(r["total_reward_1"])
        r2 = float(r["total_reward_2"])
        scores.append(1.0 - abs(r1 - r2) / (abs(r1) + abs(r2) + 1e-6))
    return {"reward_fairness": _safe_mean(scores)}


def compute_return_stats(episode_results: List[Dict]) -> Dict[str, float]:
    if not episode_results:
        return {
            "avg_reward_1": 0.0,
            "avg_reward_2": 0.0,
            "std_reward_1": 0.0,
            "std_reward_2": 0.0,
        }
    rewards_1 = [float(r["total_reward_1"]) for r in episode_results]
    rewards_2 = [float(r["total_reward_2"]) for r in episode_results]
    return {
        "avg_reward_1": float(np.mean(rewards_1)),
        "avg_reward_2": float(np.mean(rewards_2)),
        "std_reward_1": float(np.std(rewards_1)),
        "std_reward_2": float(np.std(rewards_2)),
    }


def compute_generalization_gap(
    seen_metrics: Dict[str, float],
    unseen_metrics: Dict[str, float],
    key: str = "success_rate",
) -> float:
    return float(seen_metrics.get(key, 0.0) - unseen_metrics.get(key, 0.0))


def summarize_episodes(episode_results: List[Dict], dt: float = 0.1) -> Dict[str, Optional[float]]:
    efficiency = compute_efficiency(episode_results, dt=dt)
    safety = compute_safety_metrics(episode_results)
    smoothness = compute_smoothness_metrics(episode_results, dt=dt)
    fairness = compute_fairness_metrics(episode_results)
    returns = compute_return_stats(episode_results)
    return {
        "success_rate": compute_success_rate(episode_results),
        "collision_rate": compute_collision_rate(episode_results),
        **efficiency,
        **safety,
        **smoothness,
        **fairness,
        **returns,
    }
