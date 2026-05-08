"""
Parse results/experiment_summary.json and generate a compact analysis package.

Outputs
-------
- PNG plots in results/summary_plots/
- Markdown report with tables in results/summary_plots/experiment_summary_report.md

Usage
-----
    python3 evaluation/plot_experiment_summary.py
    python3 evaluation/plot_experiment_summary.py --input results/experiment_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = "results/experiment_summary.json"
DEFAULT_OUTPUT_DIR = "results/summary_plots"
LOG_INTERVAL = 100


@dataclass(frozen=True)
class ExperimentRow:
    label: str
    display: str
    agent_1_degrees: float
    agent_2_degrees: float
    use_svo: bool
    seen_success: float
    unseen_success: float
    seen_collision: float
    unseen_collision: float
    seen_time: float
    unseen_time: float
    fairness: float
    reward_1: float
    reward_2: float
    reward_gap: float
    generalization_reward_gap: float


def metric_unwrap(metric_payload):
    if isinstance(metric_payload, dict) and "mean" in metric_payload:
        return metric_payload["mean"]
    return metric_payload


def load_summary(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def metric_mean(metric_payload: Dict) -> float:
    if isinstance(metric_payload, dict) and "mean" in metric_payload:
        return float(metric_payload["mean"])
    return float(metric_payload)


def metric_std(metric_payload: Dict) -> float:
    if isinstance(metric_payload, dict) and "std" in metric_payload:
        return float(metric_payload["std"])
    return 0.0


def config_value(value):
    if isinstance(value, dict) and "mean" in value:
        return value["mean"]
    return value


def to_display_name(label: str, config: Dict) -> str:
    aliases = {
        "selfish_vs_selfish": "Selfish vs selfish",
        "non_svo_rl_baseline": "Non-SVO RL baseline",
        "asymmetric_svo_main": "Asymmetric SVO main (15°)",
        "approx_nash": "Approximate Nash",
        "empirical_best_response": "Empirical best response",
    }
    if label in aliases:
        return aliases[label]

    if label.startswith("svo_sweep_"):
        angle = label.split("_")[-1]
        return f"SVO sweep {angle}°"
    if label.startswith("asymmetric_svo_"):
        angle = label.split("_")[-1]
        return f"Asymmetric SVO {angle}°"

    a1 = float(config_value(config.get("agent_1_degrees", 0.0)))
    a2 = float(config_value(config.get("agent_2_degrees", 0.0)))
    suffix = "SVO" if bool(config_value(config.get("use_svo", False))) else "No SVO"
    return f"{label} ({a1:.1f}°/{a2:.1f}°, {suffix})"


def collect_experiment_rows(summary: Dict) -> List[ExperimentRow]:
    rows: List[ExperimentRow] = []
    for label, payload in summary["aggregate"].items():
        evaluation = payload["evaluation"]
        config = evaluation["config"]
        seen = evaluation["seen_self_play"]
        unseen = evaluation["unseen_self_play"]
        gap = evaluation["generalization_gap"]
        reward_1 = metric_mean(unseen["avg_reward_1"])
        reward_2 = metric_mean(unseen["avg_reward_2"])
        rows.append(
            ExperimentRow(
                label=label,
                display=to_display_name(label, config),
                agent_1_degrees=float(config_value(config.get("agent_1_degrees", 0.0))),
                agent_2_degrees=float(config_value(config.get("agent_2_degrees", 0.0))),
                use_svo=bool(config_value(config.get("use_svo", False))),
                seen_success=metric_mean(seen["success_rate"]),
                unseen_success=metric_mean(unseen["success_rate"]),
                seen_collision=metric_mean(seen["collision_rate"]),
                unseen_collision=metric_mean(unseen["collision_rate"]),
                seen_time=metric_mean(seen["avg_time_to_clear"]),
                unseen_time=metric_mean(unseen["avg_time_to_clear"]),
                fairness=metric_mean(unseen["reward_fairness"]),
                reward_1=reward_1,
                reward_2=reward_2,
                reward_gap=reward_1 - reward_2,
                generalization_reward_gap=metric_mean(gap["avg_reward_1"]),
            )
        )
    return rows


def save_overview_table(rows: Sequence[ExperimentRow], output_dir: str) -> str:
    headers = [
        "Configuration",
        "Seen success",
        "Unseen success",
        "Seen collision",
        "Unseen collision",
        "Unseen time",
        "Fairness",
    ]
    table_rows = [
        [
            row.display,
            f"{row.seen_success:.1%}",
            f"{row.unseen_success:.1%}",
            f"{row.seen_collision:.1%}",
            f"{row.unseen_collision:.1%}",
            f"{row.unseen_time:.2f}s",
            f"{row.fairness:.3f}",
        ]
        for row in rows
    ]

    fig, ax = plt.subplots(figsize=(13, max(3.0, 0.55 * len(table_rows) + 1.3)))
    ax.axis("off")
    table = ax.table(
        cellText=table_rows,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#dbeafe")
        elif row_idx % 2 == 0:
            cell.set_facecolor("#f8fafc")
        if col_idx == 0:
            cell.set_width(0.28)
        else:
            cell.set_width(0.12)

    #ax.set_title("Aggregate experiment overview", fontsize=14, fontweight="bold", pad=16)
    output_path = os.path.join(output_dir, "overview_table.png")
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_training_curves(summary: Dict, rows: Sequence[ExperimentRow], output_dir: str) -> str:
    selected_labels = []
    for preferred in [
        "selfish_vs_selfish",
        "non_svo_rl_baseline",
        "asymmetric_svo_main",
        "asymmetric_svo_20",
    ]:
        if preferred in summary["aggregate"]:
            selected_labels.append(preferred)

    if not selected_labels:
        selected_labels = [row.label for row in rows[:4]]

    colors = ["#2563eb", "#16a34a", "#ea580c", "#7c3aed"]
    episodes = None
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    metrics = [
        ("success_rate", "Success rate", axes[0], (0.0, 1.05)),
        ("collision_rate", "Collision rate", axes[1], (0.0, 1.05)),
        ("avg_reward_1", "Agent 1 reward", axes[2], None),
    ]

    for color, label in zip(colors, selected_labels):
        train_metrics = summary["aggregate"][label]["train_metrics"]
        display = next(row.display for row in rows if row.label == label)
        for metric_name, ylabel, ax, y_limits in metrics:
            mean = np.asarray(train_metrics[metric_name]["mean"], dtype=float)
            std = np.asarray(train_metrics[metric_name]["std"], dtype=float)
            if episodes is None:
                episodes = np.arange(1, len(mean) + 1) * LOG_INTERVAL
            ax.plot(episodes, mean, label=display, linewidth=2, color=color)
            ax.fill_between(
                episodes,
                mean - std,
                mean + std,
                color=color,
                alpha=0.18,
            )
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
            if y_limits is not None:
                ax.set_ylim(*y_limits)

    axes[0].set_title("Training curves with mean ± std across seeds", fontsize=14, fontweight="bold")
    axes[2].set_xlabel("Training episode")
    axes[0].legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    output_path = os.path.join(output_dir, "training_curves_aggregate.png")
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_generalization(rows: Sequence[ExperimentRow], output_dir: str) -> str:
    ordered = sorted(rows, key=lambda row: row.unseen_success, reverse=True)
    labels = [row.display for row in ordered]
    success_gap = [(row.unseen_success - row.seen_success) * 100.0 for row in ordered]
    reward_gap = [row.generalization_reward_gap for row in ordered]

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4.5, 0.55 * len(labels))))
    colors_success = ["#16a34a" if value >= 0 else "#dc2626" for value in success_gap]
    colors_reward = ["#16a34a" if value >= 0 else "#dc2626" for value in reward_gap]

    axes[0].barh(labels, success_gap, color=colors_success)
    axes[0].axvline(0.0, color="black", linewidth=1)
    axes[0].set_xlabel("Unseen minus seen success (percentage points)")
    axes[0].set_title("Generalization in success rate")
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].barh(labels, reward_gap, color=colors_reward)
    axes[1].axvline(0.0, color="black", linewidth=1)
    axes[1].set_xlabel("Generalization gap in agent 1 reward")
    axes[1].set_title("Generalization in reward")
    axes[1].grid(axis="x", alpha=0.25)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "generalization_gaps.png")
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def get_cross_play_matrix(summary: Dict, experiment_label: str, split: str) -> Tuple[List[str], List[str], np.ndarray]:
    section_name = f"cross_play_{split}"
    section = summary["aggregate"][experiment_label]["evaluation"][section_name]
    row_key = next(iter(section))
    opponents = list(section[row_key].keys())
    values = np.array(
        [[metric_mean(section[row_key][opponent]["success_rate"]) for opponent in opponents]],
        dtype=float,
    )
    return [experiment_label], opponents, values


def plot_cross_play(summary: Dict, rows: Sequence[ExperimentRow], output_dir: str) -> str:
    selected = [row for row in rows if row.label in {"selfish_vs_selfish", "non_svo_rl_baseline", "asymmetric_svo_main", "asymmetric_svo_20"}]
    if not selected:
        selected = rows[: min(4, len(rows))]

    matrices = []
    column_labels = None
    for row in selected:
        _, columns, matrix = get_cross_play_matrix(summary, row.label, "unseen")
        column_labels = columns
        matrices.append(matrix[0])
    values = np.vstack(matrices)

    fig, ax = plt.subplots(figsize=(max(10, 0.85 * len(column_labels)), 4.8))
    image = ax.imshow(values, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(column_labels)))
    ax.set_xticklabels(column_labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(selected)))
    ax.set_yticklabels([row.display for row in selected])
    ax.set_title("Unseen cross-play success rate", fontsize=14, fontweight="bold")

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Success rate")
    plt.tight_layout()
    output_path = os.path.join(output_dir, "cross_play_heatmap_unseen.png")
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def extract_svo_sweep(rows: Sequence[ExperimentRow]) -> List[ExperimentRow]:
    sweep_rows = [
        row for row in rows
        if row.label.startswith("svo_sweep_") or row.label.startswith("asymmetric_svo_")
    ]
    return sorted(sweep_rows, key=lambda row: row.agent_1_degrees)


def plot_svo_sweep(rows: Sequence[ExperimentRow], output_dir: str) -> str:
    sweep_rows = extract_svo_sweep(rows)
    if not sweep_rows:
        return ""

    angles = [row.agent_1_degrees for row in sweep_rows]
    seen_success = [row.seen_success for row in sweep_rows]
    unseen_success = [row.unseen_success for row in sweep_rows]
    fairness = [row.fairness for row in sweep_rows]
    unseen_time = [row.unseen_time for row in sweep_rows]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    panels = [
        (axes[0, 0], seen_success, "Seen success", "#2563eb", (0.0, 1.05)),
        (axes[0, 1], unseen_success, "Unseen success", "#16a34a", (0.0, 1.05)),
        (axes[1, 0], fairness, "Unseen reward fairness", "#ea580c", (0.0, 1.05)),
        (axes[1, 1], unseen_time, "Unseen avg time to clear", "#7c3aed", None),
    ]

    for ax, values, title, color, y_limits in panels:
        ax.plot(angles, values, marker="o", linewidth=2.5, color=color)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        ax.set_xticks(angles)

    axes[1, 0].set_xlabel("Agent 1 SVO angle (degrees)")
    axes[1, 1].set_xlabel("Agent 1 SVO angle (degrees)")
    fig.suptitle("SVO sweep summary", fontsize=14, fontweight="bold")
    plt.tight_layout()
    output_path = os.path.join(output_dir, "svo_sweep_summary.png")
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_regret(summary: Dict, rows: Sequence[ExperimentRow], output_dir: str) -> str:
    selected = [
        row for row in rows
        if not row.label.startswith("svo_sweep_") and not row.label.startswith("asymmetric_svo_")
    ]
    if not selected:
        selected = list(rows)

    opponent_names = list(
        summary["aggregate"][selected[0].label]["evaluation"]["regret_against_fixed_opponents"].keys()
    )
    values = np.array(
        [
            [
                metric_mean(
                    summary["aggregate"][row.label]["evaluation"]["regret_against_fixed_opponents"][opponent]["regret"]
                )
                for opponent in opponent_names
            ]
            for row in selected
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.7 * len(selected))))
    image = ax.imshow(values, cmap="OrRd", aspect="auto")
    ax.set_xticks(np.arange(len(opponent_names)))
    ax.set_xticklabels(opponent_names, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(selected)))
    ax.set_yticklabels([row.display for row in selected])
    ax.set_title("Regret against fixed heuristic opponents", fontsize=14, fontweight="bold")

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Regret")
    plt.tight_layout()
    output_path = os.path.join(output_dir, "regret_heatmap.png")
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_empirical_game_comparison(summary: Dict, output_dir: str) -> str:
    if "empirical_game_analysis" not in summary:
        return ""

    agg = summary["empirical_game_analysis"]["aggregate"]
    universes = ["without_svo", "with_svo"]
    labels = ["Without SVO", "With SVO"]
    payoff_joint = [
        metric_mean(agg[name]["equilibrium"]["payoff_joint"])
        for name in universes
    ]
    exploitability = [
        metric_mean(agg[name]["equilibrium"]["exploitability"])
        for name in universes
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    colors = ["#64748b", "#0f766e"]

    axes[0].bar(labels, payoff_joint, color=colors)
    axes[0].set_title("Equilibrium joint utility")
    axes[0].set_ylabel("Joint utility")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(labels, exploitability, color=colors)
    axes[1].set_title("Equilibrium exploitability")
    axes[1].set_ylabel("Exploitability")
    axes[1].grid(axis="y", alpha=0.25)

    for ax, values in zip(axes, [payoff_joint, exploitability]):
        for idx, value in enumerate(values):
            ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Empirical Game Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    output_path = os.path.join(output_dir, "empirical_game_comparison.png")
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def direct_method_rows(summary: Dict) -> List[Tuple[str, float, float, float, float]]:
    if "direct_method_evaluation" not in summary:
        return []
    section = summary["direct_method_evaluation"]["aggregate"]["method_comparison_unseen"]
    rows = []
    for label, metrics in section.items():
        rows.append(
            (
                label,
                metric_mean(metrics["success_rate"]),
                metric_mean(metrics["collision_rate"]),
                metric_mean(metrics["reward_fairness"]),
                metric_mean(metrics["multi_objective_utility_joint"]),
            )
        )
    rows.sort(key=lambda item: item[4], reverse=True)
    return rows


def learned_method_rows(summary: Dict) -> List[Tuple[str, float, float, float, float]]:
    rows = direct_method_rows(summary)
    learned_prefixes = ("selfish_vs_selfish", "non_svo_rl_baseline", "asymmetric_svo_")
    return [row for row in rows if row[0].startswith(learned_prefixes)]


def empirical_game_summary(summary: Dict) -> Dict[str, Dict]:
    if "empirical_game_analysis" not in summary:
        return {}
    agg = summary["empirical_game_analysis"]["aggregate"]
    return {
        "without_svo": {
            "equilibrium": {k: metric_unwrap(v) for k, v in agg["without_svo"]["equilibrium"].items()},
        },
        "with_svo": {
            "equilibrium": {k: metric_unwrap(v) for k, v in agg["with_svo"]["equilibrium"].items()},
        },
    }


def plot_with_svo_payoff_heatmap(summary: Dict, output_dir: str) -> str:
    if "empirical_game_analysis" not in summary:
        return ""

    matrix = summary["empirical_game_analysis"]["aggregate"]["with_svo"]["payoff_matrix"]
    row_labels = list(matrix.keys())
    col_labels = list(next(iter(matrix.values())).keys())
    values = np.array(
        [
            [metric_mean(matrix[row][col]["utility_joint"]) for col in col_labels]
            for row in row_labels
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(max(10, 0.5 * len(col_labels)), max(7, 0.35 * len(row_labels))))
    image = ax.imshow(values, cmap="YlOrBr", aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title("With-SVO Empirical Game Payoff Matrix\n(joint utility)", fontsize=14, fontweight="bold")

    if values.shape[0] <= 14 and values.shape[1] <= 14:
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, f"{values[i, j]:.1f}", ha="center", va="center", fontsize=6)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Joint utility")
    plt.tight_layout()
    output_path = os.path.join(output_dir, "with_svo_payoff_heatmap.png")
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def format_markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def best_rows(rows: Sequence[ExperimentRow]) -> Dict[str, ExperimentRow]:
    return {
        "best_unseen_success": max(rows, key=lambda row: row.unseen_success),
        "lowest_unseen_collision": min(rows, key=lambda row: row.unseen_collision),
        "best_fairness": max(rows, key=lambda row: row.fairness),
        "fastest_unseen_clearance": min(rows, key=lambda row: row.unseen_time),
    }


def summarize_cross_play(summary: Dict, row: ExperimentRow) -> List[Tuple[str, float, float]]:
    section = summary["aggregate"][row.label]["evaluation"]["cross_play_unseen"]
    row_key = next(iter(section))
    items = []
    for opponent, metrics in section[row_key].items():
        items.append(
            (
                opponent,
                metric_mean(metrics["success_rate"]),
                metric_mean(metrics["collision_rate"]),
            )
        )
    items.sort(key=lambda item: item[1], reverse=True)
    return items[:5]


def write_report(summary: Dict, rows: Sequence[ExperimentRow], output_dir: str, generated_files: Sequence[str]) -> str:
    leaders = best_rows(rows)
    direct_rows = direct_method_rows(summary)
    learned_rows = learned_method_rows(summary)
    empirical_game = empirical_game_summary(summary)

    overview_table = format_markdown_table(
        [
            "Configuration",
            "Seen success",
            "Unseen success",
            "Unseen collision",
            "Unseen time",
            "Fairness",
            "Reward gap A1-A2",
        ],
        [
            [
                row.display,
                f"{row.seen_success:.1%}",
                f"{row.unseen_success:.1%}",
                f"{row.unseen_collision:.1%}",
                f"{row.unseen_time:.2f}s",
                f"{row.fairness:.3f}",
                f"{row.reward_gap:+.2f}",
            ]
            for row in sorted(rows, key=lambda row: row.unseen_success, reverse=True)
        ],
    )

    cross_play_leader = leaders["best_unseen_success"]
    cross_play_rows = summarize_cross_play(summary, cross_play_leader)
    cross_play_table = format_markdown_table(
        ["Opponent", "Success", "Collision"],
        [[name, f"{success:.1%}", f"{collision:.1%}"] for name, success, collision in cross_play_rows],
    )

    direct_table = ""
    if direct_rows:
        direct_table = format_markdown_table(
            ["Method", "Success", "Collision", "Fairness", "Joint utility"],
            [
                [label, f"{success:.1%}", f"{collision:.1%}", f"{fairness:.3f}", f"{joint:.2f}"]
                for label, success, collision, fairness, joint in direct_rows[:10]
            ],
        )

    learned_table = ""
    if learned_rows:
        learned_table = format_markdown_table(
            ["Learned method", "Success", "Collision", "Fairness", "Joint utility"],
            [
                [label, f"{success:.1%}", f"{collision:.1%}", f"{fairness:.3f}", f"{joint:.2f}"]
                for label, success, collision, fairness, joint in learned_rows
            ],
        )

    empirical_game_lines: List[str] = []
    if empirical_game:
        wo = empirical_game["without_svo"]["equilibrium"]
        ws = empirical_game["with_svo"]["equilibrium"]
        empirical_game_lines = [
            "## Empirical Game Comparison",
            "",
            f"- Without SVO equilibrium: **{wo['policy_1']}** vs **{wo['policy_2']}**, joint utility **{wo['payoff_joint']:.2f}**, exploitability **{wo['exploitability']:.2f}**.",
            f"- With SVO equilibrium: **{ws['policy_1']}** vs **{ws['policy_2']}**, joint utility **{ws['payoff_joint']:.2f}**, exploitability **{ws['exploitability']:.2f}**.",
            f"- Net equilibrium joint-utility change from adding SVO policies: **{ws['payoff_joint'] - wo['payoff_joint']:+.2f}**.",
            "",
        ]

    interpretations = [
        "Success rate is the primary completion metric: higher is better.",
        "Collision rate is the primary safety metric: lower is better, even when success is high.",
        "Average time to clear measures efficiency among successful interactions: lower means faster resolution.",
        "Reward fairness near 1.0 means both agents receive similar returns; lower values indicate more asymmetric outcomes.",
        "A negative generalization gap means performance drops on the unseen test split relative to the seen split.",
        "Regret measures how much multi-objective value a policy leaves on the table relative to the best response in the shared policy universe; lower is better.",
    ]

    presenter_notes = [
        "The strongest headline metric is no longer raw self-play reward; use unseen success, unseen collision, and direct-method joint utility.",
        "Lead with the direct-method table and the empirical-game comparison, then use the SVO sweep to show where performance peaks and where it degrades.",
        "If two methods have similar self-play scores, use cross-play against `approx_nash`, `empirical_best_response`, and mixed populations to separate them.",
    ]

    report_lines = [
        "# Experiment Summary Report",
        "",
        f"Input file: `{DEFAULT_INPUT}`",
        f"Seeds: {', '.join(str(seed) for seed in summary['seeds'])}",
        "",
        "## Key takeaways",
        "",
        f"- Highest unseen success: **{leaders['best_unseen_success'].display}** at **{leaders['best_unseen_success'].unseen_success:.1%}**.",
        f"- Lowest unseen collision: **{leaders['lowest_unseen_collision'].display}** at **{leaders['lowest_unseen_collision'].unseen_collision:.1%}**.",
        f"- Best fairness: **{leaders['best_fairness'].display}** at **{leaders['best_fairness'].fairness:.3f}**.",
        f"- Fastest unseen clearance: **{leaders['fastest_unseen_clearance'].display}** at **{leaders['fastest_unseen_clearance'].unseen_time:.2f}s**.",
        *(
            [
                f"- Top method in direct unseen comparison: **{direct_rows[0][0]}** with joint utility **{direct_rows[0][4]:.2f}**."
            ]
            if direct_rows else []
        ),
        "",
        "## Aggregate table",
        "",
        overview_table,
        "",
        *([ "## Direct Unseen Method Ranking", "", direct_table, "" ] if direct_table else []),
        *([ "## Learned-Only Ranking", "", learned_table, "" ] if learned_table else []),
        *empirical_game_lines,
        "## Cross-play snapshot",
        "",
        f"Top unseen cross-play results for the strongest unseen self-play configuration, **{cross_play_leader.display}**:",
        "",
        cross_play_table,
        "",
        "## Presenter Notes",
        "",
        *[f"- {line}" for line in presenter_notes],
        "",
        "## How to interpret the plots",
        "",
        *[f"- {line}" for line in interpretations],
        "",
        "## Generated files",
        "",
        *[f"- `{os.path.basename(path)}`" for path in generated_files if path],
        "",
    ]

    output_path = os.path.join(output_dir, "experiment_summary_report.md")
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report_lines))
    return output_path


def write_presentation_report(summary: Dict, rows: Sequence[ExperimentRow], output_dir: str) -> str:
    leaders = best_rows(rows)
    learned_rows = learned_method_rows(summary)
    empirical_game = empirical_game_summary(summary)
    top_learned = learned_rows[0] if learned_rows else None

    lines = [
        "# Presentation Summary",
        "",
        "## Main findings",
        "",
        f"- Best learned policy on unseen self-play: **{leaders['best_unseen_success'].display}** with **{leaders['best_unseen_success'].unseen_success:.1%}** success and **{leaders['best_unseen_success'].unseen_collision:.1%}** collision.",
        f"- Best fairness among learned policies: **{leaders['best_fairness'].display}** at **{leaders['best_fairness'].fairness:.3f}**.",
    ]

    if top_learned:
        lines.append(
            f"- Top learned method in direct unseen comparison: **{top_learned[0]}** with joint utility **{top_learned[4]:.2f}**."
        )

    if empirical_game:
        wo = empirical_game["without_svo"]["equilibrium"]
        ws = empirical_game["with_svo"]["equilibrium"]
        lines.extend(
            [
                f"- Empirical game without SVO converges to **{wo['policy_1']} vs {wo['policy_2']}** with joint utility **{wo['payoff_joint']:.2f}**.",
                f"- Empirical game with SVO converges to **{ws['policy_1']} vs {ws['policy_2']}** with joint utility **{ws['payoff_joint']:.2f}**.",
                f"- Adding SVO policies improves equilibrium joint utility by **{ws['payoff_joint'] - wo['payoff_joint']:+.2f}**.",
            ]
        )

    lines.extend(
        [
            "",
            "## Slide framing",
            "",
            "- Lead with unseen success and collision, not raw reward.",
            "- Present the learned-only ranking before showing oracle-style references such as empirical best response.",
            "- Use the with-vs-without-SVO equilibrium comparison as the core game-theoretic result.",
            "- Use the SVO sweep to show that small asymmetric SVO helps, while larger angles degrade performance.",
            "",
            "## Recommended figures",
            "",
            "- `overview_table.png`",
            "- `svo_sweep_summary.png`",
            "- `empirical_game_comparison.png`",
            "- `with_svo_payoff_heatmap.png`",
            "",
        ]
    )

    output_path = os.path.join(output_dir, "presentation_summary.md")
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return output_path


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to experiment_summary.json")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for plots and report")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    summary = load_summary(args.input)
    rows = collect_experiment_rows(summary)

    generated_files = [
        save_overview_table(rows, args.output_dir),
        plot_training_curves(summary, rows, args.output_dir),
        plot_generalization(rows, args.output_dir),
        plot_cross_play(summary, rows, args.output_dir),
        plot_svo_sweep(rows, args.output_dir),
        plot_regret(summary, rows, args.output_dir),
        plot_empirical_game_comparison(summary, args.output_dir),
        plot_with_svo_payoff_heatmap(summary, args.output_dir),
    ]
    report_path = write_report(summary, rows, args.output_dir, generated_files)
    presentation_report_path = write_presentation_report(summary, rows, args.output_dir)

    print(f"Wrote analysis assets to {args.output_dir}")
    for path in generated_files:
        if path:
            print(f"- {path}")
    print(f"- {report_path}")
    print(f"- {presentation_report_path}")


if __name__ == "__main__":
    main()
