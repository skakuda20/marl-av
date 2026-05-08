"""
Empirical game construction and analysis over a shared policy universe.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

from env.intersection_env import IntersectionEnv
from evaluation.compare_methods import evaluate_cross_play


@dataclass
class EmpiricalGameResult:
    payoff_matrix: Dict[str, Dict[str, Dict[str, float]]]
    best_response_1: Dict[str, Dict[str, float]]
    best_response_2: Dict[str, Dict[str, float]]
    equilibrium: Dict[str, float | str]
    regret_table_1: Dict[str, Dict[str, float]]
    regret_table_2: Dict[str, Dict[str, float]]


def _metric_value(metrics: Dict[str, Any], key: str) -> float:
    value = metrics[key]
    if isinstance(value, dict) and "mean" in value:
        return float(value["mean"])
    return float(value)


def build_empirical_payoff_matrix(
    policy_pairs: Dict[str, Tuple[Any, Any]],
    env: IntersectionEnv,
    n_episodes: int = 100,
    seed: int = 0,
    scenario_split: str | None = None,
    utility_weights: Dict[str, float] | None = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    cross_play = evaluate_cross_play(
        row_agents=policy_pairs,
        col_agents=policy_pairs,
        env=deepcopy(env),
        n_episodes=n_episodes,
        seed=seed,
        scenario_split=scenario_split,
        utility_weights=utility_weights,
    )

    matrix: Dict[str, Dict[str, Dict[str, float]]] = {}
    for row_label, row_values in cross_play.items():
        matrix[row_label] = {}
        for col_label, metrics in row_values.items():
            matrix[row_label][col_label] = {
                "utility_1": _metric_value(metrics, "multi_objective_utility_1"),
                "utility_2": _metric_value(metrics, "multi_objective_utility_2"),
                "utility_joint": _metric_value(metrics, "multi_objective_utility_joint"),
                "success_rate": _metric_value(metrics, "success_rate"),
                "collision_rate": _metric_value(metrics, "collision_rate"),
                "reward_fairness": _metric_value(metrics, "reward_fairness"),
            }
    return matrix


def _best_response_maps(
    payoff_matrix: Dict[str, Dict[str, Dict[str, float]]]
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    row_labels = list(payoff_matrix.keys())
    col_labels = list(next(iter(payoff_matrix.values())).keys())

    br_1: Dict[str, Dict[str, float]] = {}
    for col in col_labels:
        best_policy = max(row_labels, key=lambda row: payoff_matrix[row][col]["utility_1"])
        br_1[col] = {
            "policy": best_policy,
            "value": payoff_matrix[best_policy][col]["utility_1"],
        }

    br_2: Dict[str, Dict[str, float]] = {}
    for row in row_labels:
        best_policy = max(col_labels, key=lambda col: payoff_matrix[row][col]["utility_2"])
        br_2[row] = {
            "policy": best_policy,
            "value": payoff_matrix[row][best_policy]["utility_2"],
        }

    return br_1, br_2


def _equilibrium_from_payoff_matrix(
    payoff_matrix: Dict[str, Dict[str, Dict[str, float]]]
) -> Dict[str, float | str]:
    row_labels = list(payoff_matrix.keys())
    col_labels = list(next(iter(payoff_matrix.values())).keys())
    best_profile: Dict[str, float | str] | None = None

    for row in row_labels:
        for col in col_labels:
            utility_1 = payoff_matrix[row][col]["utility_1"]
            utility_2 = payoff_matrix[row][col]["utility_2"]
            best_dev_1 = max(payoff_matrix[alt][col]["utility_1"] for alt in row_labels)
            best_dev_2 = max(payoff_matrix[row][alt]["utility_2"] for alt in col_labels)
            exploitability = max(best_dev_1 - utility_1, best_dev_2 - utility_2)
            profile = {
                "policy_1": row,
                "policy_2": col,
                "payoff_1": utility_1,
                "payoff_2": utility_2,
                "payoff_joint": payoff_matrix[row][col]["utility_joint"],
                "exploitability": float(exploitability),
            }
            if best_profile is None or float(profile["exploitability"]) < float(best_profile["exploitability"]):
                best_profile = profile

    assert best_profile is not None
    return best_profile


def _regret_tables(
    payoff_matrix: Dict[str, Dict[str, Dict[str, float]]],
    best_response_1: Dict[str, Dict[str, float]],
    best_response_2: Dict[str, Dict[str, float]],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    row_labels = list(payoff_matrix.keys())
    col_labels = list(next(iter(payoff_matrix.values())).keys())

    regret_1: Dict[str, Dict[str, float]] = {}
    for row in row_labels:
        regret_1[row] = {}
        for col in col_labels:
            regret_1[row][col] = float(best_response_1[col]["value"] - payoff_matrix[row][col]["utility_1"])

    regret_2: Dict[str, Dict[str, float]] = {}
    for col in col_labels:
        regret_2[col] = {}
        for row in row_labels:
            regret_2[col][row] = float(best_response_2[row]["value"] - payoff_matrix[row][col]["utility_2"])

    return regret_1, regret_2


def analyze_empirical_game(
    policy_pairs: Dict[str, Tuple[Any, Any]],
    env: IntersectionEnv,
    n_episodes: int = 100,
    seed: int = 0,
    scenario_split: str | None = None,
    utility_weights: Dict[str, float] | None = None,
) -> EmpiricalGameResult:
    payoff_matrix = build_empirical_payoff_matrix(
        policy_pairs=policy_pairs,
        env=env,
        n_episodes=n_episodes,
        seed=seed,
        scenario_split=scenario_split,
        utility_weights=utility_weights,
    )
    br_1, br_2 = _best_response_maps(payoff_matrix)
    equilibrium = _equilibrium_from_payoff_matrix(payoff_matrix)
    regret_1, regret_2 = _regret_tables(payoff_matrix, br_1, br_2)
    return EmpiricalGameResult(
        payoff_matrix=payoff_matrix,
        best_response_1=br_1,
        best_response_2=br_2,
        equilibrium=equilibrium,
        regret_table_1=regret_1,
        regret_table_2=regret_2,
    )
