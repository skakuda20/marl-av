"""
Experiment runner for repeated seeded training, ablations, and cross-play.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

from agents.heuristic import MixtureAgent, create_heuristic_agent, create_heuristic_pair
from agents.policy_library import (
    DEFAULT_HEURISTIC_LIBRARY,
    policy_specs_with_svo,
    policy_specs_without_svo,
)
from env.intersection_env import IntersectionEnv
from env.rewards import RewardFunction
from evaluation.compare_methods import compare_methods, evaluate_cross_play, run_eval_episodes
from evaluation.empirical_game import analyze_empirical_game
from evaluation.metrics import compute_generalization_gap, summarize_episodes
from training.train_rl import train_rl_agent


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _svo_radians(degrees: float) -> float:
    return math.radians(float(degrees))


def _env_kwargs(params: dict, scenario_split: str) -> Dict[str, Any]:
    env_cfg = params["env"]
    return {
        "dt": env_cfg["dt"],
        "max_steps": env_cfg["max_steps"],
        "goal_threshold": env_cfg["goal_threshold"],
        "scenario_split": scenario_split,
        "scenario_distributions": env_cfg.get("scenario_distributions"),
        "localize_observations": env_cfg.get("localize_observations", True),
    }


def _reward_kwargs(params: dict) -> Dict[str, Any]:
    rew_cfg = params["rewards"]
    return {
        "collision_penalty": rew_cfg["collision_penalty"],
        "goal_reward": rew_cfg["goal_reward"],
        "time_penalty": rew_cfg["time_penalty"],
        "efficiency_weight": rew_cfg["efficiency_weight"],
        "min_safe_distance": rew_cfg["min_safe_distance"],
        "progress_weight": rew_cfg.get("progress_weight", 0.1),
    }


def _make_env(params: dict, scenario_split: str) -> IntersectionEnv:
    env = IntersectionEnv(**_env_kwargs(params, scenario_split))
    env.reward_fn = RewardFunction(**_reward_kwargs(params))
    return env


def _aggregate_numeric(values: List[Any]) -> Any:
    first = values[0]
    if isinstance(first, dict):
        return {key: _aggregate_numeric([v[key] for v in values]) for key in first}
    if isinstance(first, list):
        if not first:
            return {"mean": [], "std": []}
        return {
            "mean": np.mean(np.array(values, dtype=float), axis=0).tolist(),
            "std": np.std(np.array(values, dtype=float), axis=0).tolist(),
        }
    if isinstance(first, (int, float, np.integer, np.floating)) or first is None:
        arr = np.array([0.0 if v is None else float(v) for v in values], dtype=float)
        return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}
    return first


def _utility_weights(params: dict) -> Dict[str, float]:
    return params.get("evaluation", {}).get("multi_objective_weights", {})


def _ablation_configs(params: dict) -> List[Dict[str, Any]]:
    base_svo = params["svo"]
    ab_cfg = params.get("ablations", {})
    angle_list = ab_cfg.get("asymmetric_svo_angles_degrees", [5.0, 10.0, 15.0, 20.0, 22.5, 30.0, 45.0])
    configs = [
        {
            "label": "selfish_vs_selfish",
            "agent_1_degrees": 0.0,
            "agent_2_degrees": 0.0,
            "use_svo": True,
        },
        {
            "label": "non_svo_rl_baseline",
            "agent_1_degrees": 0.0,
            "agent_2_degrees": 0.0,
            "use_svo": False,
        },
    ]

    main_angle = float(base_svo["agent_1_degrees"])
    agent_2_angle = float(base_svo.get("agent_2_degrees", 0.0))
    if main_angle > 0.0:
        configs.append(
            {
                "label": "asymmetric_svo_main",
                "agent_1_degrees": main_angle,
                "agent_2_degrees": agent_2_angle,
                "use_svo": True,
            }
        )

    seen = {cfg["label"] for cfg in configs}
    for angle in angle_list:
        label = f"asymmetric_svo_{angle:g}"
        if angle == 0.0 or label in seen:
            continue
        configs.append(
            {
                "label": label,
                "agent_1_degrees": float(angle),
                "agent_2_degrees": 0.0,
                "use_svo": True,
            }
        )
        seen.add(label)
    return configs


def _opponent_pairs(params: dict) -> Dict[str, Tuple[Any, Any]]:
    eval_cfg = params["evaluation"]
    pairs: Dict[str, Tuple[Any, Any]] = {}
    for p1, p2 in eval_cfg.get("baselines", []):
        pairs[f"{p1}_vs_{p2}"] = create_heuristic_pair(p1, p2)

    mixture_policies = eval_cfg.get(
        "unseen_mixture_policies",
        ["aggressive", "cautious", "yield", "priority"],
    )
    pairs["unseen_mixture"] = (
        create_heuristic_agent("agent_1", "priority"),
        MixtureAgent("agent_2", mixture_policies),
    )
    pairs["mixed_population_pair"] = (
        MixtureAgent("agent_1", ["priority", "cautious"]),
        MixtureAgent("agent_2", mixture_policies),
    )
    return pairs


def train_one_seed(params: dict, config: Dict[str, Any], seed: int):
    tr_cfg = params["training"]
    ql_cfg = params["q_learning"]
    agent_1, agent_2, train_metrics = train_rl_agent(
        num_episodes=tr_cfg["num_episodes"],
        agent_1_svo=_svo_radians(config["agent_1_degrees"]),
        agent_2_svo=_svo_radians(config["agent_2_degrees"]),
        use_svo=config["use_svo"],
        seed=seed,
        env_kwargs=_env_kwargs(params, "train"),
        reward_kwargs=_reward_kwargs(params),
        agent_kwargs={
            "learning_rate": ql_cfg["learning_rate"],
            "discount": ql_cfg["discount"],
            "epsilon_start": ql_cfg["epsilon_start"],
            "epsilon_min": ql_cfg["epsilon_min"],
            "epsilon_decay": ql_cfg["epsilon_decay"],
        },
        log_interval=tr_cfg["log_interval"],
        verbose=tr_cfg.get("verbose", True),
    )
    return agent_1, agent_2, train_metrics


def _baseline_pairs_from_game(
    empirical_game: Dict[str, Any],
    policy_pairs: Dict[str, Tuple[Any, Any]],
) -> Tuple[Dict[str, Tuple[Any, Any]], Dict[str, Any]]:
    equilibrium = empirical_game["equilibrium"]
    eq_policy_1 = str(equilibrium["policy_1"])
    eq_policy_2 = str(equilibrium["policy_2"])
    br_policy_1 = str(empirical_game["best_response_1"][eq_policy_2]["policy"])
    br_policy_2 = str(empirical_game["best_response_2"][eq_policy_1]["policy"])

    pairs = {
        "approx_nash": (policy_pairs[eq_policy_1][0], policy_pairs[eq_policy_2][1]),
        "empirical_best_response": (policy_pairs[br_policy_1][0], policy_pairs[br_policy_2][1]),
    }
    metadata = {
        "approx_nash": equilibrium,
        "empirical_best_response": {
            "policy_1": br_policy_1,
            "policy_2": br_policy_2,
            "candidate_scores_1": empirical_game["best_response_1"],
            "candidate_scores_2": empirical_game["best_response_2"],
        },
    }
    return pairs, metadata


def _empirical_game_universes(
    trained_agents: Dict[str, Tuple[Any, Any]],
) -> Dict[str, Dict[str, Tuple[Any, Any]]]:
    return {
        "without_svo": policy_specs_without_svo(DEFAULT_HEURISTIC_LIBRARY, trained_agents),
        "with_svo": policy_specs_with_svo(DEFAULT_HEURISTIC_LIBRARY, trained_agents),
    }


def _analyze_empirical_games_for_seed(
    params: dict,
    seed: int,
    trained_agents: Dict[str, Tuple[Any, Any]],
) -> Dict[str, Any]:
    utility_weights = _utility_weights(params)
    n_eval = params["evaluation"]["num_eval_episodes"]
    unseen_env = _make_env(params, "test")
    universes = _empirical_game_universes(trained_agents)
    results: Dict[str, Any] = {}

    for universe_name, policy_pairs in universes.items():
        game_result = analyze_empirical_game(
            policy_pairs=policy_pairs,
            env=deepcopy(unseen_env),
            n_episodes=n_eval,
            seed=seed,
            scenario_split="test",
            utility_weights=utility_weights,
        )
        results[universe_name] = {
            "equilibrium": game_result.equilibrium,
            "best_response_1": game_result.best_response_1,
            "best_response_2": game_result.best_response_2,
            "regret_table_1": game_result.regret_table_1,
            "regret_table_2": game_result.regret_table_2,
            "payoff_matrix": game_result.payoff_matrix,
        }

    return results


def evaluate_one_seed(
    params: dict,
    config: Dict[str, Any],
    seed: int,
    agent_1,
    agent_2,
    rl_opponents: Dict[str, Tuple[Any, Any]],
    baseline_pairs: Dict[str, Tuple[Any, Any]],
    baseline_metadata: Dict[str, Any],
    empirical_game_reference: Dict[str, Any],
) -> Dict[str, Any]:
    eval_cfg = params["evaluation"]
    n_eval = eval_cfg["num_eval_episodes"]
    utility_weights = _utility_weights(params)
    seen_env = _make_env(params, "train")
    unseen_env = _make_env(params, "test")

    seen_episodes = run_eval_episodes(
        agent_1, agent_2, deepcopy(seen_env), n_episodes=n_eval, seed=seed, scenario_split="train"
    )
    unseen_episodes = run_eval_episodes(
        agent_1, agent_2, deepcopy(unseen_env), n_episodes=n_eval, seed=seed, scenario_split="test"
    )
    seen_metrics = summarize_episodes(seen_episodes, dt=seen_env.dt, utility_weights=utility_weights)
    unseen_metrics = summarize_episodes(unseen_episodes, dt=unseen_env.dt, utility_weights=utility_weights)

    heuristic_opponents = _opponent_pairs(params)
    row_agents = {config["label"]: (agent_1, agent_2)}
    col_agents = {**heuristic_opponents, **baseline_pairs, **rl_opponents}
    cross_seen = evaluate_cross_play(
        row_agents=row_agents,
        col_agents=col_agents,
        env=deepcopy(seen_env),
        n_episodes=n_eval,
        seed=seed,
        scenario_split="train",
        utility_weights=utility_weights,
    )
    cross_unseen = evaluate_cross_play(
        row_agents=row_agents,
        col_agents=col_agents,
        env=deepcopy(unseen_env),
        n_episodes=n_eval,
        seed=seed,
        scenario_split="test",
        utility_weights=utility_weights,
    )

    regret = {}
    for opponent_name, opponent_pair in heuristic_opponents.items():
        if "_vs_" not in opponent_name:
            continue
        if config["label"] not in empirical_game_reference["with_svo"]["regret_table_1"]:
            continue
        best_response_1 = empirical_game_reference["with_svo"]["best_response_1"]
        mapped_policy = baseline_metadata.get("opponent_policy_map", {}).get(opponent_name)
        if mapped_policy is None:
            continue
        best_response_value = float(best_response_1[mapped_policy]["value"])
        trained_payoff = cross_unseen[config["label"]][opponent_name]["multi_objective_utility_1"]
        regret[opponent_name] = {
            "best_response_policy": best_response_1[mapped_policy]["policy"],
            "best_response_value": best_response_value,
            "trained_value": trained_payoff,
            "regret": best_response_value - trained_payoff,
        }

    heuristic_results = compare_methods(
        num_eval_episodes=n_eval,
        method_dict={**heuristic_opponents, **baseline_pairs},
        env=deepcopy(unseen_env),
        seed=seed,
        scenario_split="test",
        utility_weights=utility_weights,
    )

    return {
        "config": config,
        "seed": seed,
        "seen_self_play": seen_metrics,
        "unseen_self_play": unseen_metrics,
        "generalization_gap": {
            "success_rate": compute_generalization_gap(seen_metrics, unseen_metrics, "success_rate"),
            "collision_rate": compute_generalization_gap(seen_metrics, unseen_metrics, "collision_rate"),
            "avg_reward_1": compute_generalization_gap(seen_metrics, unseen_metrics, "avg_reward_1"),
        },
        "cross_play_seen": cross_seen,
        "cross_play_unseen": cross_unseen,
        "regret_against_fixed_opponents": regret,
        "game_theoretic_baselines": {
            **baseline_metadata,
            "heuristic_baselines": heuristic_results,
        },
        "empirical_game_reference": empirical_game_reference["with_svo"],
    }


def _evaluate_method_tables_for_seed(
    params: dict,
    seed: int,
    trained_agents: Dict[str, Tuple[Any, Any]],
    baseline_pairs: Dict[str, Tuple[Any, Any]],
    game_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    utility_weights = _utility_weights(params)
    n_eval = params["evaluation"]["num_eval_episodes"]
    unseen_env = _make_env(params, "test")
    seen_env = _make_env(params, "train")
    heuristic_pairs = _opponent_pairs(params)
    method_pairs = {**trained_agents, **heuristic_pairs, **baseline_pairs}

    return {
        "method_comparison_seen": compare_methods(
            num_eval_episodes=n_eval,
            method_dict=method_pairs,
            env=deepcopy(seen_env),
            seed=seed,
            scenario_split="train",
            utility_weights=utility_weights,
        ),
        "method_comparison_unseen": compare_methods(
            num_eval_episodes=n_eval,
            method_dict=method_pairs,
            env=deepcopy(unseen_env),
            seed=seed,
            scenario_split="test",
            utility_weights=utility_weights,
        ),
        "cross_play_seen_full": evaluate_cross_play(
            row_agents=method_pairs,
            col_agents=method_pairs,
            env=deepcopy(seen_env),
            n_episodes=n_eval,
            seed=seed,
            scenario_split="train",
            utility_weights=utility_weights,
        ),
        "cross_play_unseen_full": evaluate_cross_play(
            row_agents=method_pairs,
            col_agents=method_pairs,
            env=deepcopy(unseen_env),
            n_episodes=n_eval,
            seed=seed,
            scenario_split="test",
            utility_weights=utility_weights,
        ),
        "game_theoretic_reference": game_metadata,
    }


def save_outputs(results: Dict[str, Any], params: dict) -> None:
    out_cfg = params.get("output", {})
    results_dir = Path(out_cfg.get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "experiment_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved experiment summary -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run marl-av seeded ablations and evaluation.")
    parser.add_argument("params_file", nargs="?", default="params.yaml")
    parser.add_argument("--params", dest="params_file_flag", default=None)
    args = parser.parse_args()

    params_path = args.params_file_flag or args.params_file
    if not os.path.exists(params_path):
        print(f"ERROR: params file not found: {params_path}")
        sys.exit(1)

    params = load_params(params_path)
    seeds = params.get("seeds", [0, 1, 2])
    configs = _ablation_configs(params)

    all_seed_results: Dict[str, List[Dict[str, Any]]] = {cfg["label"]: [] for cfg in configs}
    direct_eval_per_seed: List[Dict[str, Any]] = []
    empirical_game_per_seed: List[Dict[str, Any]] = []
    trained_agents_by_seed: Dict[int, Dict[str, Tuple[Any, Any]]] = {}
    train_metrics_by_config: Dict[str, List[Dict[str, Any]]] = {cfg["label"]: [] for cfg in configs}

    for seed in seeds:
        trained_agents_by_seed[seed] = {}
        for config in configs:
            print(f"\nTraining {config['label']} with seed {seed}")
            agent_1, agent_2, train_metrics = train_one_seed(params, config, seed)
            trained_agents_by_seed[seed][config["label"]] = (agent_1, agent_2)
            train_metrics_by_config[config["label"]].append(train_metrics)

        empirical_game_reference = _analyze_empirical_games_for_seed(
            params=params,
            seed=seed,
            trained_agents=trained_agents_by_seed[seed],
        )
        empirical_game_per_seed.append(empirical_game_reference)
        policy_universe = _empirical_game_universes(trained_agents_by_seed[seed])["with_svo"]
        baseline_pairs, baseline_metadata = _baseline_pairs_from_game(
            empirical_game_reference["with_svo"],
            policy_universe,
        )
        baseline_metadata["opponent_policy_map"] = {
            opponent_name: opponent_name.split("_vs_")[-1]
            for opponent_name in _opponent_pairs(params)
            if "_vs_" in opponent_name
        }

        for config in configs:
            rl_opponents = {
                label: agents
                for label, agents in trained_agents_by_seed[seed].items()
                if label != config["label"]
            }
            agent_1, agent_2 = trained_agents_by_seed[seed][config["label"]]
            print(f"Evaluating {config['label']} with seed {seed}")
            seed_result = evaluate_one_seed(
                params=params,
                config=config,
                seed=seed,
                agent_1=agent_1,
                agent_2=agent_2,
                rl_opponents=rl_opponents,
                baseline_pairs=baseline_pairs,
                baseline_metadata=baseline_metadata,
                empirical_game_reference=empirical_game_reference,
            )
            all_seed_results[config["label"]].append(seed_result)

        direct_eval_per_seed.append(
            _evaluate_method_tables_for_seed(
                params=params,
                seed=seed,
                trained_agents=trained_agents_by_seed[seed],
                baseline_pairs=baseline_pairs,
                game_metadata=baseline_metadata,
            )
        )

    aggregate_results = {
        label: {
            "train_metrics": _aggregate_numeric(train_metrics_by_config[label]),
            "evaluation": _aggregate_numeric(all_seed_results[label]),
        }
        for label in all_seed_results
    }

    final_results = {
        "params_path": params_path,
        "seeds": seeds,
        "per_seed": all_seed_results,
        "aggregate": aggregate_results,
        "direct_method_evaluation": {
            "per_seed": direct_eval_per_seed,
            "aggregate": _aggregate_numeric(direct_eval_per_seed),
        },
        "empirical_game_analysis": {
            "per_seed": empirical_game_per_seed,
            "aggregate": _aggregate_numeric(empirical_game_per_seed),
        },
    }
    save_outputs(final_results, params)


if __name__ == "__main__":
    main()
