"""
Visualization suite for MARL intersection experiment results.

Produces five types of plots:
  1. Training curves      — loss/success/reward over training episodes
  2. Method comparison    — success/collision rates and efficiency across policies
  3. SVO decomposition    — diagram explaining how SVO angle weights rewards
  4. SVO sweep            — how varying agent 1's SVO angle affects joint outcomes
  5. SVO vs No-SVO        — head-to-head training curves and eval comparison

Usage
-----
    # Fast (no retraining, use existing data only):
    python evaluation/plot_results.py --skip-sweep --skip-comparison

    # SVO vs No-SVO comparison only (~20 min, skips sweep):
    python evaluation/plot_results.py --skip-sweep

    # Full run (sweep + comparison, ~35 min total):
    python evaluation/plot_results.py

    # Re-run comparison even if cached results exist:
    python evaluation/plot_results.py --overwrite-comparison --skip-sweep

All plots are saved to results/plots/.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

from env.rewards import (
    SVO_SELFISH, SVO_PROSOCIAL, SVO_ALTRUISTIC, SVO_COMPETITIVE,
)
from evaluation.compare_methods import run_eval_episodes
from evaluation.metrics import compute_success_rate, compute_collision_rate, compute_efficiency
from training.train_rl import train_rl_agent, load_rl_agents
from env.intersection_env import IntersectionEnv

SAVE_DIR = "results/plots"
TRAIN_METRICS_PATH = "results/train_metrics.json"
EVAL_RESULTS_PATH = "results/eval_results.json"
SWEEP_CACHE_PATH = "results/svo_sweep.json"
COMPARISON_CACHE_PATH = "results/svo_comparison.json"
LOG_INTERVAL = 100  # episodes per logged data point in train_metrics.json


# ---------------------------------------------------------------------------
# 1. Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(metrics_path: str = TRAIN_METRICS_PATH,
                         save_dir: str = SAVE_DIR) -> None:
    """
    Plot training progress from train_metrics.json.

    Three panels (shared x-axis):
      - Success rate & collision rate (with smoothed overlay)
      - Average reward for each agent (highlights SVO-induced asymmetry)
      - Average episode length in steps
    """
    with open(metrics_path) as f:
        m = json.load(f)

    epochs = np.arange(1, len(m["success_rate"]) + 1)
    episodes = epochs * LOG_INTERVAL

    def smooth(arr, w=5):
        kernel = np.ones(w) / w
        return np.convolve(arr, kernel, mode='same')

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.suptitle("Training Curves\n(Agent 1: SVO 45° prosocial  ·  Agent 2: SVO 0° selfish)",
                 fontsize=13, fontweight='bold')

    # --- Panel 1: Success & collision rate ---
    ax = axes[0]
    sr = np.array(m["success_rate"])
    cr = np.array(m["collision_rate"])
    ax.plot(episodes, sr, color='#55aa55', alpha=0.35, linewidth=1)
    ax.plot(episodes, cr, color='#cc4444', alpha=0.35, linewidth=1)
    ax.plot(episodes, smooth(sr), color='#228b22', linewidth=2.5,
            label='Success rate (smoothed)')
    ax.plot(episodes, smooth(cr), color='#aa1111', linewidth=2.5,
            label='Collision rate (smoothed)')
    ax.set_ylabel("Rate", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Episode Outcomes", fontsize=11)

    # --- Panel 2: Average reward per agent ---
    ax = axes[1]
    r1 = np.array(m["avg_reward_1"])
    r2 = np.array(m["avg_reward_2"])
    ax.plot(episodes, r1, color='#3366cc', alpha=0.35, linewidth=1)
    ax.plot(episodes, r2, color='#cc7733', alpha=0.35, linewidth=1)
    ax.plot(episodes, smooth(r1), color='#1144aa', linewidth=2.5,
            label='Agent 1 avg reward (φ=45°)')
    ax.plot(episodes, smooth(r2), color='#aa5500', linewidth=2.5,
            label='Agent 2 avg reward (φ=0°)')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_ylabel("Cumulative reward", fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Per-Agent Reward (SVO asymmetry visible in scale)", fontsize=11)

    # --- Panel 3: Average episode length ---
    ax = axes[2]
    st = np.array(m["avg_steps"])
    ax.plot(episodes, st, color='#9966cc', alpha=0.35, linewidth=1)
    ax.plot(episodes, smooth(st), color='#663399', linewidth=2.5,
            label='Avg steps (smoothed)')
    ax.set_ylabel("Steps", fontsize=11)
    ax.set_xlabel("Training episode", fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Episode Length (longer = agents learning to avoid early crashes)", fontsize=11)

    plt.tight_layout()
    out = os.path.join(save_dir, "training_curves.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# 2. Method comparison
# ---------------------------------------------------------------------------

def plot_method_comparison(eval_path: str = EVAL_RESULTS_PATH,
                           save_dir: str = SAVE_DIR) -> None:
    """
    Two figures comparing heuristic/RL methods:
      A) Grouped bar: success rate vs collision rate per method
      B) Horizontal bar: methods ranked by success, annotated with time-to-clear
    """
    with open(eval_path) as f:
        data = json.load(f)

    labels = list(data.keys())
    # Shorten long RL label for display
    short_labels = [
        lbl.replace("RL  (A1=45.0° vs A2=0.0°)", "RL (45° vs 0°)")
           .replace("heuristic  ", "")
        for lbl in labels
    ]

    success  = [data[l]["success_rate"]   * 100 for l in labels]
    collision = [data[l]["collision_rate"] * 100 for l in labels]
    t_clear  = [data[l]["avg_time_to_clear"] for l in labels]

    x = np.arange(len(labels))
    w = 0.35

    # --- Figure A: grouped bar ---
    fig, ax = plt.subplots(figsize=(10, 5))
    bars_s = ax.bar(x - w / 2, success,   w, label='Success rate (%)',   color='#55aa55')
    bars_c = ax.bar(x + w / 2, collision, w, label='Collision rate (%)', color='#cc4444')

    for bar in bars_s:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                f'{h:.0f}%', ha='center', va='bottom', fontsize=9)
    for bar in bars_c:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                f'{h:.0f}%', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel("Rate (%)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_title("Success & Collision Rates by Method", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out_a = os.path.join(save_dir, "method_comparison.png")
    plt.savefig(out_a, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_a}")

    # --- Figure B: horizontal efficiency bar ---
    # Sort by success rate descending
    order = sorted(range(len(labels)), key=lambda i: success[i], reverse=True)
    sorted_labels  = [short_labels[i] for i in order]
    sorted_success = [success[i] for i in order]
    sorted_tclear  = [t_clear[i] for i in order]

    colors = ['#228b22' if s >= 50 else '#cc4444' for s in sorted_success]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(sorted_labels, sorted_success, color=colors, alpha=0.85)
    ax.axvline(50, color='gray', linestyle='--', linewidth=1, alpha=0.7,
               label='50% threshold')

    for i, (bar, tc) in enumerate(zip(bars, sorted_tclear)):
        w_val = bar.get_width()
        tc_str = f"{tc:.1f}s" if tc is not None else "N/A"
        ax.text(min(w_val + 1.5, 103), bar.get_y() + bar.get_height() / 2,
                f"T-clear: {tc_str}", va='center', fontsize=9)

    ax.set_xlabel("Success rate (%)", fontsize=11)
    ax.set_xlim(0, 120)
    ax.set_title("Methods Ranked by Success Rate\n(annotated with time-to-clear)",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    out_b = os.path.join(save_dir, "method_efficiency.png")
    plt.savefig(out_b, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_b}")


# ---------------------------------------------------------------------------
# 3. SVO decomposition diagram
# ---------------------------------------------------------------------------

def plot_svo_decomposition(save_dir: str = SAVE_DIR) -> None:
    """
    Two-panel figure explaining SVO:
      Left)  Unit circle with annotated angle presets and weight vectors
      Right) Bar chart: cos(φ) (weight_self) vs sin(φ) (weight_other)
             for the four standard presets
    """
    presets = [
        ("Competitive\n(−45°)",  SVO_COMPETITIVE,  '#cc3333'),
        ("Selfish\n(0°)",        SVO_SELFISH,       '#cc7733'),
        ("Prosocial\n(45°)",     SVO_PROSOCIAL,     '#3366cc'),
        ("Altruistic\n(90°)",    SVO_ALTRUISTIC,    '#228b22'),
    ]

    fig, (ax_circle, ax_bar) = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("Social Value Orientation (SVO): reward weighting explained",
                 fontsize=13, fontweight='bold')

    # --- Left panel: unit circle ---
    theta = np.linspace(0, 2 * np.pi, 300)
    ax_circle.plot(np.cos(theta), np.sin(theta), 'lightgray', linewidth=1.5)
    ax_circle.axhline(0, color='lightgray', linewidth=0.8)
    ax_circle.axvline(0, color='lightgray', linewidth=0.8)

    for name, phi, color in presets:
        cos_phi, sin_phi = np.cos(phi), np.sin(phi)
        ax_circle.annotate(
            '', xy=(cos_phi, sin_phi), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color=color, lw=2.5)
        )
        # Offset labels slightly outward
        offset = 1.18
        ax_circle.text(cos_phi * offset, sin_phi * offset, name,
                       ha='center', va='center', fontsize=9,
                       color=color, fontweight='bold')

    ax_circle.set_xlim(-1.6, 1.6)
    ax_circle.set_ylim(-1.6, 1.6)
    ax_circle.set_aspect('equal')
    ax_circle.set_xlabel("weight_self  =  cos(φ)", fontsize=10)
    ax_circle.set_ylabel("weight_other  =  sin(φ)", fontsize=10)
    ax_circle.set_title("SVO Angle Presets on Unit Circle", fontsize=11)

    # Annotate formula
    ax_circle.text(0, -1.5,
                   r"$r_{svo} = r_{self} \cdot \cos\varphi + r_{other} \cdot \sin\varphi$",
                   ha='center', va='center', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # --- Right panel: grouped bar ---
    labels_bar = [p[0].replace('\n', ' ') for p in presets]
    w_self  = [np.cos(p[1]) for p in presets]
    w_other = [np.sin(p[1]) for p in presets]

    x = np.arange(len(presets))
    bw = 0.35
    ax_bar.bar(x - bw / 2, w_self,  bw, label='cos(φ)  weight_self',  color='#3366cc', alpha=0.85)
    ax_bar.bar(x + bw / 2, w_other, bw, label='sin(φ)  weight_other', color='#cc3333', alpha=0.85)
    ax_bar.axhline(0, color='black', linewidth=0.8)

    for i, (ws, wo) in enumerate(zip(w_self, w_other)):
        ax_bar.text(i - bw / 2, ws + (0.04 if ws >= 0 else -0.08),
                    f'{ws:.2f}', ha='center', va='bottom', fontsize=8)
        ax_bar.text(i + bw / 2, wo + (0.04 if wo >= 0 else -0.08),
                    f'{wo:.2f}', ha='center', va='bottom', fontsize=8)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels_bar, fontsize=9)
    ax_bar.set_ylabel("Weight", fontsize=11)
    ax_bar.set_ylim(-1.2, 1.2)
    ax_bar.set_title("Self vs Other Reward Weight per SVO Preset", fontsize=11)
    ax_bar.legend(fontsize=9)
    ax_bar.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(save_dir, "svo_decomposition.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# 4. SVO sweep
# ---------------------------------------------------------------------------

def run_svo_sweep(
    save_dir: str = SAVE_DIR,
    angles_deg: list = None,
    n_train: int = 2000,
    n_eval: int = 100,
    overwrite: bool = False,
) -> None:
    """
    Train RL agents across a range of SVO angles for agent 1 (agent 2 held at
    0°/selfish) and plot how joint outcomes change with increasing prosociality.

    Results are cached to results/svo_sweep.json. Set overwrite=True to re-run.
    """
    if angles_deg is None:
        angles_deg = [0, 15, 30, 45, 60, 75, 90]

    # --- Load or compute sweep data ---
    if os.path.exists(SWEEP_CACHE_PATH) and not overwrite:
        print(f"Loading cached SVO sweep from {SWEEP_CACHE_PATH}")
        with open(SWEEP_CACHE_PATH) as f:
            sweep = json.load(f)
    else:
        print(f"Running SVO sweep over angles: {angles_deg}°")
        print(f"  {n_train} training episodes × {len(angles_deg)} angles — this may take 10-15 min")
        sweep = {"angles_deg": [], "success_rate": [], "collision_rate": [],
                 "avg_reward_1": [], "avg_reward_2": []}

        env = IntersectionEnv()
        for deg in angles_deg:
            phi = np.radians(deg)
            print(f"  Training φ = {deg}°...", end=' ', flush=True)
            agent_1, agent_2, _ = train_rl_agent(
                num_episodes=n_train,
                agent_1_svo=phi,
                agent_2_svo=SVO_SELFISH,
                verbose=False,
            )
            episodes = run_eval_episodes(agent_1, agent_2, env, n_episodes=n_eval)

            sr = compute_success_rate(episodes)
            cr = compute_collision_rate(episodes)
            r1 = float(np.mean([e["total_reward_1"] for e in episodes]))
            r2 = float(np.mean([e["total_reward_2"] for e in episodes]))
            print(f"success={sr:.2f}  collision={cr:.2f}  R1={r1:.1f}  R2={r2:.1f}")

            sweep["angles_deg"].append(deg)
            sweep["success_rate"].append(sr)
            sweep["collision_rate"].append(cr)
            sweep["avg_reward_1"].append(r1)
            sweep["avg_reward_2"].append(r2)

        with open(SWEEP_CACHE_PATH, 'w') as f:
            json.dump(sweep, f, indent=2)
        print(f"Saved sweep cache to {SWEEP_CACHE_PATH}")

    # --- Plot ---
    angles = sweep["angles_deg"]
    sr     = sweep["success_rate"]
    cr     = sweep["collision_rate"]
    r1     = sweep["avg_reward_1"]
    r2     = sweep["avg_reward_2"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Effect of Agent 1's SVO Angle on Joint Outcomes\n"
        "(Agent 2 held at φ=0° selfish; Agent 1 varies from selfish → altruistic)",
        fontsize=12, fontweight='bold'
    )

    # Left: success & collision vs angle
    ax1.plot(angles, [s * 100 for s in sr], 'o-', color='#228b22',
             linewidth=2.5, markersize=7, label='Success rate (%)')
    ax1.plot(angles, [c * 100 for c in cr], 's-', color='#cc4444',
             linewidth=2.5, markersize=7, label='Collision rate (%)')
    ax1.axvline(45, color='gray', linestyle=':', linewidth=1.2,
                label='φ=45° (prosocial)')
    ax1.set_xlabel("Agent 1 SVO angle (degrees)", fontsize=11)
    ax1.set_ylabel("Rate (%)", fontsize=11)
    ax1.set_ylim(-5, 105)
    ax1.set_xticks(angles)
    ax1.set_title("Success & Collision vs SVO Angle", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right: reward per agent vs angle
    ax2.plot(angles, r1, 'o-', color='#3366cc', linewidth=2.5,
             markersize=7, label='Agent 1 avg reward (prosocial)')
    ax2.plot(angles, r2, 's-', color='#cc7733', linewidth=2.5,
             markersize=7, label='Agent 2 avg reward (selfish)')
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax2.axvline(45, color='gray', linestyle=':', linewidth=1.2,
                label='φ=45° (prosocial)')
    ax2.set_xlabel("Agent 1 SVO angle (degrees)", fontsize=11)
    ax2.set_ylabel("Average cumulative reward", fontsize=11)
    ax2.set_xticks(angles)
    ax2.set_title("Per-Agent Reward vs SVO Angle", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(save_dir, "svo_sweep.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# 5. SVO vs No-SVO comparison
# ---------------------------------------------------------------------------

def plot_svo_vs_nosvo(
    save_dir: str = SAVE_DIR,
    n_train: int = 5000,
    n_eval: int = 200,
    overwrite: bool = False,
) -> None:
    """
    Train two RL configurations and compare their outcomes:
      - With SVO:    Agent 1 φ=45° (prosocial) vs Agent 2 φ=0° (selfish)
      - Without SVO: Both agents update on raw rewards only (φ ignored)

    Produces two plots:
      svo_vs_nosvo_training.png — paired training curves (success, collision, steps)
      svo_vs_nosvo_eval.png     — grouped bar with heuristic baselines included

    Results cached to results/svo_comparison.json; set overwrite=True to retrain.
    """
    if os.path.exists(COMPARISON_CACHE_PATH) and not overwrite:
        print(f"Loading cached comparison from {COMPARISON_CACHE_PATH}")
        with open(COMPARISON_CACHE_PATH) as f:
            cache = json.load(f)
    else:
        print(f"Training SVO vs No-SVO ({n_train} episodes each — ~20 min total)")
        env = IntersectionEnv()

        # --- Config A: With SVO (prosocial vs selfish) ---
        print("  [1/2] With SVO  (φ₁=45°, φ₂=0°)...")
        svo_path = "results/q_tables_svo.npz"
        a1_svo, a2_svo, metrics_svo = train_rl_agent(
            num_episodes=n_train,
            agent_1_svo=SVO_PROSOCIAL,
            agent_2_svo=SVO_SELFISH,
            use_svo=True,
            save_path=svo_path,
            verbose=False,
        )
        eps_svo = run_eval_episodes(a1_svo, a2_svo, env, n_episodes=n_eval)
        eff_svo = compute_efficiency(eps_svo)
        eval_svo = {
            "success_rate":      compute_success_rate(eps_svo),
            "collision_rate":    compute_collision_rate(eps_svo),
            "avg_time_to_clear": eff_svo["avg_time_to_clear"],
            "avg_delay":         eff_svo["avg_delay"],
            "avg_steps":         eff_svo["avg_steps"],
        }
        print(f"     success={eval_svo['success_rate']:.2f}  "
              f"collision={eval_svo['collision_rate']:.2f}")

        # --- Config B: Without SVO (raw rewards, both selfish-equivalent) ---
        print("  [2/2] Without SVO (raw rewards, φ ignored)...")
        nosvo_path = "results/q_tables_no_svo.npz"
        a1_ns, a2_ns, metrics_nosvo = train_rl_agent(
            num_episodes=n_train,
            agent_1_svo=SVO_SELFISH,
            agent_2_svo=SVO_SELFISH,
            use_svo=False,
            save_path=nosvo_path,
            verbose=False,
        )
        eps_nosvo = run_eval_episodes(a1_ns, a2_ns, env, n_episodes=n_eval)
        eff_nosvo = compute_efficiency(eps_nosvo)
        eval_nosvo = {
            "success_rate":      compute_success_rate(eps_nosvo),
            "collision_rate":    compute_collision_rate(eps_nosvo),
            "avg_time_to_clear": eff_nosvo["avg_time_to_clear"],
            "avg_delay":         eff_nosvo["avg_delay"],
            "avg_steps":         eff_nosvo["avg_steps"],
        }
        print(f"     success={eval_nosvo['success_rate']:.2f}  "
              f"collision={eval_nosvo['collision_rate']:.2f}")

        cache = {
            "svo_train":   metrics_svo,
            "nosvo_train": metrics_nosvo,
            "svo_eval":    eval_svo,
            "nosvo_eval":  eval_nosvo,
        }
        with open(COMPARISON_CACHE_PATH, 'w') as f:
            json.dump(cache, f, indent=2)
        print(f"Saved comparison cache to {COMPARISON_CACHE_PATH}")

    # -----------------------------------------------------------------------
    # Plot A: paired training curves
    # -----------------------------------------------------------------------
    ms = cache["svo_train"]
    mn = cache["nosvo_train"]

    def smooth(arr, w=5):
        kernel = np.ones(w) / w
        return np.convolve(arr, kernel, mode='same')

    n_pts = len(ms["success_rate"])
    episodes = np.arange(1, n_pts + 1) * LOG_INTERVAL

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(
        "With SVO  vs  Without SVO — Training Curves",
        fontsize=13, fontweight='bold'
    )

    svo_kw   = dict(color='#3366cc', linewidth=2.5)
    nosvo_kw = dict(color='#cc7733', linewidth=2.5)

    # Panel 1: success rate
    ax = axes[0]
    ax.plot(episodes, ms["success_rate"], color='#3366cc', alpha=0.25, linewidth=1)
    ax.plot(episodes, mn["success_rate"], color='#cc7733', alpha=0.25, linewidth=1)
    ax.plot(episodes, smooth(ms["success_rate"]), label='With SVO (φ₁=45°)', **svo_kw)
    ax.plot(episodes, smooth(mn["success_rate"]), label='No SVO (raw rewards)', **nosvo_kw)
    ax.set_ylabel("Success rate", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Success Rate", fontsize=11)

    # Panel 2: collision rate
    ax = axes[1]
    ax.plot(episodes, ms["collision_rate"], color='#3366cc', alpha=0.25, linewidth=1)
    ax.plot(episodes, mn["collision_rate"], color='#cc7733', alpha=0.25, linewidth=1)
    ax.plot(episodes, smooth(ms["collision_rate"]), label='With SVO', **svo_kw)
    ax.plot(episodes, smooth(mn["collision_rate"]), label='No SVO', **nosvo_kw)
    ax.set_ylabel("Collision rate", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Collision Rate", fontsize=11)

    # Panel 3: avg steps
    ax = axes[2]
    ax.plot(episodes, ms["avg_steps"], color='#3366cc', alpha=0.25, linewidth=1)
    ax.plot(episodes, mn["avg_steps"], color='#cc7733', alpha=0.25, linewidth=1)
    ax.plot(episodes, smooth(ms["avg_steps"]), label='With SVO', **svo_kw)
    ax.plot(episodes, smooth(mn["avg_steps"]), label='No SVO', **nosvo_kw)
    ax.set_ylabel("Avg episode steps", fontsize=11)
    ax.set_xlabel("Training episode", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Episode Length", fontsize=11)

    plt.tight_layout()
    out_a = os.path.join(save_dir, "svo_vs_nosvo_training.png")
    plt.savefig(out_a, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_a}")

    # -----------------------------------------------------------------------
    # Plot B: eval comparison — RL configs + heuristic baselines
    # -----------------------------------------------------------------------
    with open(EVAL_RESULTS_PATH) as f:
        heuristic_data = json.load(f)

    # Build unified ordered dict: RL entries first, then heuristics
    all_methods = {
        "RL + SVO (φ₁=45°, φ₂=0°)": cache["svo_eval"],
        "RL no SVO":                  cache["nosvo_eval"],
    }
    for lbl, vals in heuristic_data.items():
        short = (lbl.replace("RL  (A1=45.0° vs A2=0.0°)", "RL (45° vs 0°)")
                    .replace("heuristic  ", ""))
        all_methods[short] = vals

    labels  = list(all_methods.keys())
    success  = [all_methods[l]["success_rate"]   * 100 for l in labels]
    collision = [all_methods[l]["collision_rate"] * 100 for l in labels]

    x  = np.arange(len(labels))
    bw = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    bars_s = ax.bar(x - bw / 2, success,   bw, label='Success rate (%)',   color='#55aa55')
    bars_c = ax.bar(x + bw / 2, collision, bw, label='Collision rate (%)', color='#cc4444')

    for bar in bars_s:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                f'{h:.0f}%', ha='center', va='bottom', fontsize=8)
    for bar in bars_c:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                f'{h:.0f}%', ha='center', va='bottom', fontsize=8)

    # Shade RL columns to distinguish from heuristics
    for i in range(2):
        ax.axvspan(i - 0.5, i + 0.5, alpha=0.06, color='gray', zorder=0)
    ax.text(0.5, 107, 'RL agents', ha='center', fontsize=8,
            color='gray', transform=ax.get_xaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel("Rate (%)", fontsize=11)
    ax.set_ylim(0, 120)
    ax.set_title("SVO vs No-SVO vs Heuristic Baselines",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out_b = os.path.join(save_dir, "svo_vs_nosvo_eval.png")
    plt.savefig(out_b, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_b}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate visualizations for the MARL intersection experiment."
    )
    parser.add_argument(
        '--skip-sweep', action='store_true',
        help='Skip the SVO sweep training (~10-15 min). Use cached results if available.'
    )
    parser.add_argument(
        '--overwrite-sweep', action='store_true',
        help='Re-run SVO sweep training even if cached results exist.'
    )
    parser.add_argument(
        '--skip-comparison', action='store_true',
        help='Skip the SVO vs No-SVO comparison training (~20 min). Use cached results if available.'
    )
    parser.add_argument(
        '--overwrite-comparison', action='store_true',
        help='Re-run SVO vs No-SVO comparison training even if cached results exist.'
    )
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)

    print("=" * 60)
    print("1/5  Training curves")
    print("=" * 60)
    plot_training_curves()

    print("=" * 60)
    print("2/5  Method comparison")
    print("=" * 60)
    plot_method_comparison()

    print("=" * 60)
    print("3/5  SVO decomposition diagram")
    print("=" * 60)
    plot_svo_decomposition()

    if args.skip_sweep:
        print("=" * 60)
        print("4/5  SVO sweep — SKIPPED (--skip-sweep flag set)")
        if os.path.exists(SWEEP_CACHE_PATH):
            print(f"     Cached results found at {SWEEP_CACHE_PATH}; plotting from cache.")
            run_svo_sweep(overwrite=False)
        else:
            print("     No cache found. Run without --skip-sweep to generate.")
        print("=" * 60)
    else:
        print("=" * 60)
        print("4/5  SVO sweep (training required — this takes ~10-15 min)")
        print("=" * 60)
        run_svo_sweep(overwrite=args.overwrite_sweep)

    if args.skip_comparison:
        print("=" * 60)
        print("5/5  SVO vs No-SVO comparison — SKIPPED (--skip-comparison flag set)")
        if os.path.exists(COMPARISON_CACHE_PATH):
            print(f"     Cached results found at {COMPARISON_CACHE_PATH}; plotting from cache.")
            plot_svo_vs_nosvo(overwrite=False)
        else:
            print("     No cache found. Run without --skip-comparison to generate.")
        print("=" * 60)
    else:
        print("=" * 60)
        print("5/5  SVO vs No-SVO comparison (training required — ~20 min)")
        print("=" * 60)
        plot_svo_vs_nosvo(overwrite=args.overwrite_comparison)

    print("\nAll plots saved to", SAVE_DIR)


if __name__ == "__main__":
    main()
