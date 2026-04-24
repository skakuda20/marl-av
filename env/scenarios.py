"""
Scenario sampling utilities for the intersection environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class ScenarioDistribution:
    """Parameterized distribution over reset-time scenario variations."""

    spawn_distance_range: Tuple[float, float]
    initial_speed_range: Tuple[float, float]
    arrival_offset_range: Tuple[float, float]
    center_offset_range: Tuple[float, float]
    priority_probs: Dict[str, float]


DEFAULT_SCENARIO_DISTRIBUTIONS: Dict[str, ScenarioDistribution] = {
    "train": ScenarioDistribution(
        spawn_distance_range=(18.0, 24.0),
        initial_speed_range=(6.5, 9.5),
        arrival_offset_range=(-1.25, 1.25),
        center_offset_range=(-1.5, 1.5),
        priority_probs={"balanced": 0.4, "agent_1": 0.3, "agent_2": 0.3},
    ),
    "test": ScenarioDistribution(
        spawn_distance_range=(24.0, 32.0),
        initial_speed_range=(4.0, 12.0),
        arrival_offset_range=(1.5, 3.5),
        center_offset_range=(-3.0, 3.0),
        priority_probs={"balanced": 0.2, "agent_1": 0.4, "agent_2": 0.4},
    ),
}


def _sample_priority(rng: np.random.Generator, probs: Dict[str, float]) -> str:
    labels = list(probs.keys())
    weights = np.array([probs[label] for label in labels], dtype=float)
    weights = weights / weights.sum()
    return str(rng.choice(labels, p=weights))


def priority_value(priority: str, agent_id: str) -> float:
    """Return a per-agent priority hint in {-1, 0, 1}."""
    if priority == "balanced":
        return 0.0
    if priority == agent_id:
        return 1.0
    return -1.0


def sample_scenario(
    rng: np.random.Generator,
    distribution: ScenarioDistribution,
) -> Dict[str, object]:
    """
    Sample one scenario configuration.

    The vehicles still travel along perpendicular roads, but reset-time
    variation changes their spawn distance, initial speed, arrival offset,
    and the location of the intersection center.
    """
    spawn_min, spawn_max = distribution.spawn_distance_range
    speed_min, speed_max = distribution.initial_speed_range
    offset_min, offset_max = distribution.arrival_offset_range
    center_min, center_max = distribution.center_offset_range

    priority = _sample_priority(rng, distribution.priority_probs)

    center_x = float(rng.uniform(center_min, center_max))
    center_y = float(rng.uniform(center_min, center_max))

    speed_1 = float(rng.uniform(speed_min, speed_max))
    speed_2 = float(rng.uniform(speed_min, speed_max))

    spawn_1 = float(rng.uniform(spawn_min, spawn_max))
    arrival_offset = float(rng.uniform(offset_min, offset_max))

    target_tti_1 = spawn_1 / max(speed_1, 1e-6)
    target_tti_2 = max(0.25, target_tti_1 + arrival_offset)
    spawn_2 = float(np.clip(target_tti_2 * speed_2, spawn_min, spawn_max))

    return {
        "priority": priority,
        "center": (center_x, center_y),
        "spawn_distance_1": spawn_1,
        "spawn_distance_2": spawn_2,
        "initial_speed_1": speed_1,
        "initial_speed_2": speed_2,
        "arrival_offset": arrival_offset,
        "start_pos_1": np.array([center_x - spawn_1, center_y], dtype=np.float32),
        "goal_pos_1": (center_x + spawn_1, center_y),
        "start_vel_1": np.array([speed_1, 0.0], dtype=np.float32),
        "start_pos_2": np.array([center_x, center_y - spawn_2], dtype=np.float32),
        "goal_pos_2": (center_x, center_y + spawn_2),
        "start_vel_2": np.array([0.0, speed_2], dtype=np.float32),
    }
