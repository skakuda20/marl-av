"""
Evaluation metrics for comparing agent approaches.

Each function accepts a list of episode-result dicts. The expected keys per
dict match those produced by compare_methods.run_eval_episodes (and the
existing demo/render_episode.py:run_episode):

    collision      : bool   — whether a collision occurred
    done_1         : bool   — whether agent_1 reached its goal
    done_2         : bool   — whether agent_2 reached its goal
    steps          : int    — number of steps in the episode
    total_reward_1 : float  — cumulative raw reward for agent_1
    total_reward_2 : float  — cumulative raw reward for agent_2
"""

import numpy as np
from typing import List, Dict

# Unimpeded travel time baseline: 40 m gap / 8 m/s initial speed = 5 s → 50 steps at dt=0.1
_UNIMPEDED_STEPS = 50


def compute_success_rate(episode_results: List[Dict]) -> float:
    """
    Fraction of episodes where both vehicles reached their goals without collision.

    Args:
        episode_results: List of episode result dicts.

    Returns:
        Success rate in [0, 1].
    """
    if not episode_results:
        return 0.0
    successes = sum(
        1 for r in episode_results
        if r["done_1"] and r["done_2"] and not r["collision"]
    )
    return successes / len(episode_results)


def compute_collision_rate(episode_results: List[Dict]) -> float:
    """
    Fraction of episodes that ended in a collision.

    Args:
        episode_results: List of episode result dicts.

    Returns:
        Collision rate in [0, 1].
    """
    if not episode_results:
        return 0.0
    return sum(1 for r in episode_results if r["collision"]) / len(episode_results)


def compute_efficiency(episode_results: List[Dict], dt: float = 0.1) -> Dict:
    """
    Compute time-based efficiency metrics across episodes.

    Args:
        episode_results: List of episode result dicts.
        dt: Environment time step in seconds (default 0.1).

    Returns:
        Dictionary with:
            avg_time_to_clear : mean seconds to complete the episode for
                                successful (non-collision, both done) episodes.
                                None if no successful episodes.
            avg_delay         : mean extra seconds beyond the unimpeded
                                baseline (_UNIMPEDED_STEPS * dt) across all
                                episodes. Positive means slower than baseline.
            avg_steps         : mean episode length in steps across all
                                episodes.
    """
    if not episode_results:
        return {"avg_time_to_clear": None, "avg_delay": 0.0, "avg_steps": 0.0}

    all_steps = [r["steps"] for r in episode_results]
    successful = [r for r in episode_results if r["done_1"] and r["done_2"] and not r["collision"]]

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


if __name__ == "__main__":
    # Quick self-test with synthetic data
    mock = [
        {"collision": False, "done_1": True,  "done_2": True,  "steps": 52, "total_reward_1": 40.0, "total_reward_2": 40.0},
        {"collision": False, "done_1": True,  "done_2": True,  "steps": 60, "total_reward_1": 35.0, "total_reward_2": 35.0},
        {"collision": True,  "done_1": False, "done_2": False, "steps": 30, "total_reward_1": -80.0, "total_reward_2": -80.0},
        {"collision": False, "done_1": True,  "done_2": False, "steps": 200, "total_reward_1": 10.0, "total_reward_2": -10.0},
    ]
    print(f"Success rate : {compute_success_rate(mock):.2f}")   # 0.50
    print(f"Collision rate: {compute_collision_rate(mock):.2f}")  # 0.25
    print(f"Efficiency   : {compute_efficiency(mock)}")
