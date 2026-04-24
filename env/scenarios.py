"""
Scenario sampling utilities for the intersection environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import numpy as np


@dataclass(frozen=True)
class ScenarioDistribution:
    """Parameterized distribution over reset-time scenario variations."""

    spawn_distance_range: Tuple[float, float]
    initial_speed_range: Tuple[float, float]
    arrival_offset_range: Tuple[float, float]
    center_offset_range: Tuple[float, float]
    priority_probs: Dict[str, float]
    geometry_probs: Dict[str, float]
    lane_offset_range: Tuple[float, float] = (0.0, 0.0)
    observation_noise_std: float = 0.0
    occlusion_probability: float = 0.0
    speed_noise_std: float = 0.0


DEFAULT_SCENARIO_DISTRIBUTIONS: Dict[str, ScenarioDistribution] = {
    "train": ScenarioDistribution(
        spawn_distance_range=(18.0, 24.0),
        initial_speed_range=(6.5, 9.5),
        arrival_offset_range=(-1.25, 1.25),
        center_offset_range=(-1.5, 1.5),
        priority_probs={"balanced": 0.4, "agent_1": 0.3, "agent_2": 0.3},
        geometry_probs={"standard": 0.6, "offset_narrow": 0.2, "offset_wide": 0.2},
        lane_offset_range=(-1.0, 1.0),
        observation_noise_std=0.02,
        occlusion_probability=0.0,
        speed_noise_std=0.15,
    ),
    "test": ScenarioDistribution(
        spawn_distance_range=(18.0, 34.0),
        initial_speed_range=(3.0, 13.5),
        arrival_offset_range=(-0.4, 0.4),
        center_offset_range=(-4.0, 4.0),
        priority_probs={"balanced": 0.1, "agent_1": 0.45, "agent_2": 0.45},
        geometry_probs={"standard": 0.35, "offset_narrow": 0.3, "offset_wide": 0.35},
        lane_offset_range=(-3.0, 3.0),
        observation_noise_std=0.2,
        occlusion_probability=0.18,
        speed_noise_std=0.65,
    ),
}


def _sample_priority(rng: np.random.Generator, probs: Dict[str, float]) -> str:
    labels = list(probs.keys())
    weights = np.array([probs[label] for label in labels], dtype=float)
    weights = weights / weights.sum()
    return str(rng.choice(labels, p=weights))


def _sample_label(rng: np.random.Generator, probs: Dict[str, float]) -> str:
    labels = list(probs.keys())
    weights = np.array([probs[label] for label in labels], dtype=float)
    weights = weights / weights.sum()
    return str(rng.choice(labels, p=weights))


def scenario_distribution_from_dict(raw: Mapping[str, object]) -> ScenarioDistribution:
    return ScenarioDistribution(
        spawn_distance_range=tuple(raw["spawn_distance_range"]),
        initial_speed_range=tuple(raw["initial_speed_range"]),
        arrival_offset_range=tuple(raw["arrival_offset_range"]),
        center_offset_range=tuple(raw["center_offset_range"]),
        priority_probs=dict(raw["priority_probs"]),
        geometry_probs=dict(raw.get("geometry_probs", {"standard": 1.0})),
        lane_offset_range=tuple(raw.get("lane_offset_range", (0.0, 0.0))),
        observation_noise_std=float(raw.get("observation_noise_std", 0.0)),
        occlusion_probability=float(raw.get("occlusion_probability", 0.0)),
        speed_noise_std=float(raw.get("speed_noise_std", 0.0)),
    )


def build_scenario_distributions(config: Mapping[str, object] | None) -> Dict[str, ScenarioDistribution]:
    if not config:
        return DEFAULT_SCENARIO_DISTRIBUTIONS
    return {
        split: scenario_distribution_from_dict(raw)
        for split, raw in config.items()
    }


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
    geometry = _sample_label(rng, distribution.geometry_probs)

    center_x = float(rng.uniform(center_min, center_max))
    center_y = float(rng.uniform(center_min, center_max))

    speed_1 = float(rng.uniform(speed_min, speed_max))
    speed_2 = float(rng.uniform(speed_min, speed_max))
    if distribution.speed_noise_std > 0.0:
        speed_1 += float(rng.normal(0.0, distribution.speed_noise_std))
        speed_2 += float(rng.normal(0.0, distribution.speed_noise_std))
    speed_1 = float(np.clip(speed_1, speed_min, speed_max))
    speed_2 = float(np.clip(speed_2, speed_min, speed_max))

    spawn_1 = float(rng.uniform(spawn_min, spawn_max))
    arrival_offset = float(rng.uniform(offset_min, offset_max))

    target_tti_1 = spawn_1 / max(speed_1, 1e-6)
    target_tti_2 = max(0.25, target_tti_1 + arrival_offset)
    spawn_2 = float(np.clip(target_tti_2 * speed_2, spawn_min, spawn_max))

    lane_min, lane_max = distribution.lane_offset_range
    lane_shift_1 = float(rng.uniform(lane_min, lane_max))
    lane_shift_2 = float(rng.uniform(lane_min, lane_max))
    if geometry == "offset_narrow":
        lane_shift_1 *= 0.5
        lane_shift_2 *= 0.5
    elif geometry == "offset_wide":
        lane_shift_1 *= 1.25
        lane_shift_2 *= 1.25

    return {
        "priority": priority,
        "geometry": geometry,
        "center": (center_x, center_y),
        "spawn_distance_1": spawn_1,
        "spawn_distance_2": spawn_2,
        "initial_speed_1": speed_1,
        "initial_speed_2": speed_2,
        "arrival_offset": arrival_offset,
        "lane_shift_1": lane_shift_1,
        "lane_shift_2": lane_shift_2,
        "observation_noise_std": float(distribution.observation_noise_std),
        "occlusion_probability": float(distribution.occlusion_probability),
        "start_pos_1": np.array([center_x - spawn_1, center_y + lane_shift_1], dtype=np.float32),
        "goal_pos_1": (center_x + spawn_1, center_y + lane_shift_1),
        "start_vel_1": np.array([speed_1, 0.0], dtype=np.float32),
        "start_pos_2": np.array([center_x + lane_shift_2, center_y - spawn_2], dtype=np.float32),
        "goal_pos_2": (center_x + lane_shift_2, center_y + spawn_2),
        "start_vel_2": np.array([0.0, speed_2], dtype=np.float32),
    }
