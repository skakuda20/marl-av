"""
Experiment runner: train, test, and evaluate Q-learning agents.

Reads all parameters from params.yaml (or a custom path passed on the
command line) so every run is fully reproducible.

Usage
-----
    python3 run_experiment.py                      # uses params.yaml
    python3 run_experiment.py my_params.yaml       # custom file
    python3 run_experiment.py --params ablation.yaml

Output
------
If output.results_dir is set, the following files are written there:
    q_tables.npz        — Q-tables + SVO angles for both agents
    train_metrics.json  — windowed training metrics
    eval_results.json   — per-method evaluation results
"""

import sys
import os
import json
import math
import argparse
from pathlib import Path

import yaml
import numpy as np

# Allow running from the repo root without installing the package
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from env.intersection_env import IntersectionEnv
from env.rewards import RewardFunction, apply_svo_transform
from agents.rl_agent import RLAgent
from agents.heuristic import create_heuristic_pair
from evaluation.compare_methods import run_eval_episodes, compare_methods
from evaluation.metrics import (
    compute_success_rate, compute_collision_rate, compute_efficiency
)


# ---------------------------------------------------------------------------
# Parameter loading
# ---------------------------------------------------------------------------

def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _svo_radians(degrees: float) -> float:
    return math.radians(degrees)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_training(params: dict):
    """
    Train two Q-learning agents using parameters from params.yaml.

    Returns:
        (agent_1, agent_2, train_metrics)
    """
    env_cfg   = params["env"]
    rew_cfg   = params["rewards"]
    svo_cfg   = params["svo"]
    ql_cfg    = params["q_learning"]
    tr_cfg    = params["training"]

    svo_1 = _svo_radians(svo_cfg["agent_1_degrees"])
    svo_2 = _svo_radians(svo_cfg["agent_2_degrees"])

    env = IntersectionEnv(
        dt=env_cfg["dt"],
        max_steps=env_cfg["max_steps"],
        goal_threshold=env_cfg["goal_threshold"],
    )

    # Patch the environment's reward function with params.yaml values
    env.reward_fn = RewardFunction(
        collision_penalty=rew_cfg["collision_penalty"],
        goal_reward=rew_cfg["goal_reward"],
        time_penalty=rew_cfg["time_penalty"],
        efficiency_weight=rew_cfg["efficiency_weight"],
        min_safe_distance=rew_cfg["min_safe_distance"],
        progress_weight=rew_cfg.get("progress_weight", 0.1),
    )

    agent_1 = RLAgent(
        "agent_1",
        svo_angle=svo_1,
        learning_rate=ql_cfg["learning_rate"],
        discount=ql_cfg["discount"],
        epsilon_start=ql_cfg["epsilon_start"],
        epsilon_min=ql_cfg["epsilon_min"],
        epsilon_decay=ql_cfg["epsilon_decay"],
    )
    agent_2 = RLAgent(
        "agent_2",
        svo_angle=svo_2,
        learning_rate=ql_cfg["learning_rate"],
        discount=ql_cfg["discount"],
        epsilon_start=ql_cfg["epsilon_start"],
        epsilon_min=ql_cfg["epsilon_min"],
        epsilon_decay=ql_cfg["epsilon_decay"],
    )

    num_episodes = tr_cfg["num_episodes"]
    log_interval = tr_cfg["log_interval"]

    metrics = {
        "collision_rate": [], "success_rate": [],
        "avg_reward_1": [], "avg_reward_2": [], "avg_steps": [],
    }

    win_collisions = win_successes = 0
    win_r1, win_r2, win_steps = [], [], []

    print(f"\n{'='*60}")
    print(f"TRAINING  ({num_episodes} episodes)")
    print(f"  agent_1 SVO: {svo_cfg['agent_1_degrees']}°  |  "
          f"agent_2 SVO: {svo_cfg['agent_2_degrees']}°")
    print(f"  α={ql_cfg['learning_rate']}  γ={ql_cfg['discount']}  "
          f"ε: {ql_cfg['epsilon_start']} → {ql_cfg['epsilon_min']}  "
          f"decay={ql_cfg['epsilon_decay']}")
    print(f"{'='*60}")

    for episode in range(num_episodes):
        obs, _ = env.reset()
        total_r1 = total_r2 = 0.0
        terminated = truncated = False

        while not (terminated or truncated):
            obs_1, obs_2 = obs["agent_1"], obs["agent_2"]
            a1 = agent_1.get_action(obs_1, training=True)
            a2 = agent_2.get_action(obs_2, training=True)

            next_obs, reward, terminated, truncated, info = env.step(
                np.array([a1, a2])
            )
            done = terminated or truncated

            raw_r1, raw_r2 = reward["agent_1"], reward["agent_2"]
            svo_r1 = apply_svo_transform(raw_r1, raw_r2, agent_1.svo_angle)
            svo_r2 = apply_svo_transform(raw_r2, raw_r1, agent_2.svo_angle)

            agent_1.update(obs_1, a1, svo_r1, next_obs["agent_1"], done)
            agent_2.update(obs_2, a2, svo_r2, next_obs["agent_2"], done)

            obs = next_obs
            total_r1 += raw_r1
            total_r2 += raw_r2

        agent_1.decay_epsilon()
        agent_2.decay_epsilon()

        win_collisions += int(info.get("collision", False))
        win_successes  += int(info.get("done_1", False) and info.get("done_2", False))
        win_r1.append(total_r1)
        win_r2.append(total_r2)
        win_steps.append(info["steps"])

        if (episode + 1) % log_interval == 0:
            metrics["collision_rate"].append(win_collisions / log_interval)
            metrics["success_rate"].append(win_successes / log_interval)
            metrics["avg_reward_1"].append(float(np.mean(win_r1)))
            metrics["avg_reward_2"].append(float(np.mean(win_r2)))
            metrics["avg_steps"].append(float(np.mean(win_steps)))
            print(
                f"  ep {episode + 1:>5} | "
                f"coll {metrics['collision_rate'][-1]:.2f} | "
                f"succ {metrics['success_rate'][-1]:.2f} | "
                f"R1 {metrics['avg_reward_1'][-1]:>7.2f} | "
                f"R2 {metrics['avg_reward_2'][-1]:>7.2f} | "
                f"steps {metrics['avg_steps'][-1]:.1f} | "
                f"ε {agent_1.epsilon:.3f}"
            )
            win_collisions = win_successes = 0
            win_r1, win_r2, win_steps = [], [], []

    return agent_1, agent_2, metrics


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluation(agent_1: RLAgent, agent_2: RLAgent, params: dict) -> dict:
    """
    Evaluate trained RL agents against heuristic baselines.

    Returns:
        Full results dict from compare_methods.
    """
    ev_cfg  = params["evaluation"]
    env_cfg = params["env"]
    rew_cfg = params["rewards"]
    svo_cfg = params["svo"]

    num_eval = ev_cfg["num_eval_episodes"]

    # Build an eval env that matches the training configuration
    eval_env = IntersectionEnv(
        dt=env_cfg["dt"],
        max_steps=env_cfg["max_steps"],
        goal_threshold=env_cfg["goal_threshold"],
    )
    eval_env.reward_fn = RewardFunction(
        collision_penalty=rew_cfg["collision_penalty"],
        goal_reward=rew_cfg["goal_reward"],
        time_penalty=rew_cfg["time_penalty"],
        efficiency_weight=rew_cfg["efficiency_weight"],
        min_safe_distance=rew_cfg["min_safe_distance"],
        progress_weight=rew_cfg.get("progress_weight", 0.1),
    )

    method_dict = {
        f"RL  (A1={svo_cfg['agent_1_degrees']}° vs A2={svo_cfg['agent_2_degrees']}°)":
            (agent_1, agent_2),
    }
    for pair in ev_cfg.get("baselines", []):
        label = f"heuristic  {pair[0]} vs {pair[1]}"
        method_dict[label] = create_heuristic_pair(pair[0], pair[1])

    print(f"\n{'='*60}")
    print(f"EVALUATION  ({num_eval} episodes per method)")
    print(f"{'='*60}")

    results = compare_methods(
        num_eval_episodes=num_eval,
        method_dict=method_dict,
        env=eval_env,
    )
    return results


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_outputs(agent_1, agent_2, train_metrics, eval_results, params):
    out_cfg = params.get("output", {})
    if not out_cfg.get("save_q_tables", True):
        return

    results_dir = Path(out_cfg.get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    q_path = results_dir / "q_tables.npz"
    np.savez(
        q_path,
        q_table_1=agent_1.q_table,
        q_table_2=agent_2.q_table,
        svo_1=agent_1.svo_angle,
        svo_2=agent_2.svo_angle,
    )
    print(f"\nSaved Q-tables  → {q_path}")

    tm_path = results_dir / "train_metrics.json"
    with open(tm_path, "w") as f:
        json.dump(train_metrics, f, indent=2)
    print(f"Saved training metrics → {tm_path}")

    # eval_results may contain None (avg_time_to_clear); convert for JSON
    def _jsonify(obj):
        if obj is None:
            return None
        if isinstance(obj, float):
            return obj
        return obj

    ev_path = results_dir / "eval_results.json"
    serialisable = {
        label: {k: _jsonify(v) for k, v in m.items()}
        for label, m in eval_results.items()
    }
    with open(ev_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"Saved eval results     → {ev_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate marl-av Q-learning agents."
    )
    parser.add_argument(
        "params_file",
        nargs="?",
        default="params.yaml",
        help="Path to the YAML parameter file (default: params.yaml)",
    )
    parser.add_argument(
        "--params",
        dest="params_file_flag",
        default=None,
        help="Alternative flag-style path: --params my_params.yaml",
    )
    args = parser.parse_args()

    params_path = args.params_file_flag or args.params_file
    if not os.path.exists(params_path):
        print(f"ERROR: params file not found: {params_path}")
        sys.exit(1)

    print(f"Loading parameters from: {params_path}")
    params = load_params(params_path)

    # ---- TRAIN ----
    agent_1, agent_2, train_metrics = run_training(params)

    # ---- EVALUATE ----
    eval_results = run_evaluation(agent_1, agent_2, params)

    # ---- SAVE ----
    save_outputs(agent_1, agent_2, train_metrics, eval_results, params)

    print("\nDone.")


if __name__ == "__main__":
    main()
