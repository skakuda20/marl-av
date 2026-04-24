"""
Empirical game-theoretic baselines over the heuristic policy set.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from agents.heuristic import BestResponseAgent, create_heuristic_agent
from evaluation.compare_methods import run_eval_episodes


DEFAULT_POLICY_SET = ["constant", "cautious", "aggressive", "yield", "priority"]


@dataclass
class EmpiricalNashResult:
    policy_1: str
    policy_2: str
    payoff_1: float
    payoff_2: float
    exploitability: float


def _clone_env(env):
    env_copy = deepcopy(env)
    return env_copy


def _mean_payoffs(episodes: List[Dict]) -> Tuple[float, float]:
    return (
        float(np.mean([ep["total_reward_1"] for ep in episodes])),
        float(np.mean([ep["total_reward_2"] for ep in episodes])),
    )


def empirical_best_response(
    env,
    opponent_policy: str,
    agent_id: str = "agent_1",
    candidate_policies: Iterable[str] = DEFAULT_POLICY_SET,
    n_episodes: int = 100,
) -> Tuple[BestResponseAgent, Dict[str, float]]:
    """
    Search a heuristic policy set for the highest-payoff empirical best response.
    """
    candidate_scores: Dict[str, float] = {}
    for policy in candidate_policies:
        eval_env = _clone_env(env)
        if agent_id == "agent_1":
            agent_1 = create_heuristic_agent("agent_1", policy)
            agent_2 = create_heuristic_agent("agent_2", opponent_policy)
            episodes = run_eval_episodes(agent_1, agent_2, eval_env, n_episodes=n_episodes)
            candidate_scores[policy] = float(np.mean([ep["total_reward_1"] for ep in episodes]))
        else:
            agent_1 = create_heuristic_agent("agent_1", opponent_policy)
            agent_2 = create_heuristic_agent("agent_2", policy)
            episodes = run_eval_episodes(agent_1, agent_2, eval_env, n_episodes=n_episodes)
            candidate_scores[policy] = float(np.mean([ep["total_reward_2"] for ep in episodes]))

    best_policy = max(candidate_scores, key=candidate_scores.get)
    return (
        BestResponseAgent(agent_id, create_heuristic_agent(agent_id, best_policy), best_policy),
        candidate_scores,
    )


def approximate_nash_equilibrium(
    env,
    candidate_policies: Iterable[str] = DEFAULT_POLICY_SET,
    n_episodes: int = 100,
) -> EmpiricalNashResult:
    """
    Find the lowest-exploitability pure-strategy profile in the heuristic policy set.
    """
    policies = list(candidate_policies)
    payoff_matrix: Dict[Tuple[str, str], Tuple[float, float]] = {}

    for policy_1 in policies:
        for policy_2 in policies:
            eval_env = _clone_env(env)
            agent_1 = create_heuristic_agent("agent_1", policy_1)
            agent_2 = create_heuristic_agent("agent_2", policy_2)
            episodes = run_eval_episodes(agent_1, agent_2, eval_env, n_episodes=n_episodes)
            payoff_matrix[(policy_1, policy_2)] = _mean_payoffs(episodes)

    best_result: EmpiricalNashResult | None = None
    for policy_1 in policies:
        for policy_2 in policies:
            payoff_1, payoff_2 = payoff_matrix[(policy_1, policy_2)]
            best_dev_1 = max(payoff_matrix[(dev_1, policy_2)][0] for dev_1 in policies)
            best_dev_2 = max(payoff_matrix[(policy_1, dev_2)][1] for dev_2 in policies)
            exploitability = max(best_dev_1 - payoff_1, best_dev_2 - payoff_2)
            result = EmpiricalNashResult(
                policy_1=policy_1,
                policy_2=policy_2,
                payoff_1=payoff_1,
                payoff_2=payoff_2,
                exploitability=float(exploitability),
            )
            if best_result is None or result.exploitability < best_result.exploitability:
                best_result = result

    assert best_result is not None
    return best_result
