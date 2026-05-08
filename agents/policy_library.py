"""
Policy registry for empirical game construction.

The registry unifies:
- parameterized heuristic variants
- fixed baseline heuristics
- trained RL policy candidates
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Tuple

from agents.heuristic import (
    AggressiveAgent,
    CautiousAgent,
    ConstantVelocityAgent,
    HeuristicAgent,
    PriorityAgent,
    YieldAgent,
    create_heuristic_agent,
)


AgentFactory = Callable[[str], Any]


@dataclass(frozen=True)
class PolicySpec:
    policy_id: str
    family: str
    source: str = "heuristic"
    params: Dict[str, Any] = field(default_factory=dict)


def _make_constant(agent_id: str, params: Dict[str, Any]) -> HeuristicAgent:
    return ConstantVelocityAgent(agent_id, target_speed=float(params.get("target_speed", 8.0)))


def _make_cautious(agent_id: str, params: Dict[str, Any]) -> HeuristicAgent:
    return CautiousAgent(
        agent_id,
        target_speed=float(params.get("target_speed", 8.0)),
        safe_distance=float(params.get("safe_distance", 8.0)),
        critical_distance=float(params.get("critical_distance", 5.0)),
    )


def _make_aggressive(agent_id: str, params: Dict[str, Any]) -> HeuristicAgent:
    return AggressiveAgent(
        agent_id,
        target_speed=float(params.get("target_speed", 12.0)),
        critical_distance=float(params.get("critical_distance", 3.0)),
    )


def _make_yield(agent_id: str, params: Dict[str, Any]) -> HeuristicAgent:
    return YieldAgent(
        agent_id,
        target_speed=float(params.get("target_speed", 8.0)),
        intersection_zone=float(params.get("intersection_zone", 5.0)),
        safe_distance=float(params.get("safe_distance", 10.0)),
    )


def _make_priority(agent_id: str, params: Dict[str, Any]) -> HeuristicAgent:
    return PriorityAgent(
        agent_id,
        priority_speed=float(params.get("priority_speed", 10.0)),
        yield_speed=float(params.get("yield_speed", 4.0)),
        approach_zone=float(params.get("approach_zone", 16.0)),
    )


HEURISTIC_FAMILY_BUILDERS: Dict[str, Callable[[str, Dict[str, Any]], HeuristicAgent]] = {
    "constant": _make_constant,
    "cautious": _make_cautious,
    "aggressive": _make_aggressive,
    "yield": _make_yield,
    "priority": _make_priority,
}


DEFAULT_HEURISTIC_LIBRARY: Tuple[PolicySpec, ...] = (
    PolicySpec("constant", "constant"),
    PolicySpec("constant_cruise_10", "constant", params={"target_speed": 10.0}),
    PolicySpec("cautious", "cautious"),
    PolicySpec("cautious_buffered", "cautious", params={"target_speed": 7.5, "safe_distance": 10.0, "critical_distance": 5.5}),
    PolicySpec("cautious_late_brake", "cautious", params={"target_speed": 8.5, "safe_distance": 7.0, "critical_distance": 4.0}),
    PolicySpec("aggressive", "aggressive"),
    PolicySpec("aggressive_fast", "aggressive", params={"target_speed": 13.0, "critical_distance": 2.5}),
    PolicySpec("aggressive_brake_late", "aggressive", params={"target_speed": 12.0, "critical_distance": 2.0}),
    PolicySpec("yield", "yield"),
    PolicySpec("yield_early", "yield", params={"target_speed": 7.0, "intersection_zone": 7.0, "safe_distance": 12.0}),
    PolicySpec("yield_late", "yield", params={"target_speed": 8.5, "intersection_zone": 4.0, "safe_distance": 8.5}),
    PolicySpec("priority", "priority"),
    PolicySpec("priority_strict", "priority", params={"priority_speed": 11.0, "yield_speed": 2.5, "approach_zone": 18.0}),
    PolicySpec("priority_soft", "priority", params={"priority_speed": 9.0, "yield_speed": 5.0, "approach_zone": 14.0}),
)


def create_policy_agent(agent_id: str, spec: PolicySpec) -> Any:
    if spec.source == "heuristic":
        builder = HEURISTIC_FAMILY_BUILDERS[spec.family]
        return builder(agent_id, spec.params)
    raise ValueError(f"Unsupported policy source for direct instantiation: {spec.source}")


def build_policy_pair_map(
    policy_specs: Iterable[PolicySpec],
    rl_policy_pairs: Dict[str, Tuple[Any, Any]] | None = None,
) -> Dict[str, Tuple[Any, Any]]:
    pair_map: Dict[str, Tuple[Any, Any]] = {}
    for spec in policy_specs:
        pair_map[spec.policy_id] = (
            create_policy_agent("agent_1", spec),
            create_policy_agent("agent_2", spec),
        )

    for label, pair in (rl_policy_pairs or {}).items():
        pair_map[label] = pair

    return pair_map


def policy_specs_without_svo(policy_specs: Iterable[PolicySpec], rl_policy_pairs: Dict[str, Tuple[Any, Any]] | None = None) -> Dict[str, Tuple[Any, Any]]:
    rl_pairs = {
        label: pair
        for label, pair in (rl_policy_pairs or {}).items()
        if "svo" not in label or label == "non_svo_rl_baseline"
    }
    return build_policy_pair_map(policy_specs, rl_pairs)


def policy_specs_with_svo(policy_specs: Iterable[PolicySpec], rl_policy_pairs: Dict[str, Tuple[Any, Any]] | None = None) -> Dict[str, Tuple[Any, Any]]:
    return build_policy_pair_map(policy_specs, rl_policy_pairs)
