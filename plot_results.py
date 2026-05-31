#!/usr/bin/env python3
"""
Generate all figures for the poster and final report.

Produces:
  figures/policy_comparison_bar.png   — bar chart: all policies × metrics
  figures/policy_comparison_gen.png   — generalization to cluster 4
  figures/training_curves.png         — true_sat training curves per policy
  figures/reward_alignment.png        — T-REX reward vs. oracle true_sat

Usage
-----
    python plot_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)

POLICY_LABELS = {
    "random":       "Random",
    "engagement":   "Engagement\n(π_E)",
    "trex":         "T-REX\n(π_T)",
    "ground_truth": "Oracle\n(π_GT)",
}

POLICY_COLORS = {
    "random":       "#aaaaaa",
    "engagement":   "#4e79a7",
    "trex":         "#f28e2b",
    "ground_truth": "#59a14f",
}

HISTORY_FILES = {
    "engagement":   "checkpoints/ppo_train_history_engagement.json",
    "trex":         "checkpoints/ppo_train_history_trex.json",
    "ground_truth": "checkpoints/ppo_train_history_ground_truth.json",
}


def load_eval_report() -> dict | None:
    p = Path("checkpoints/policy_eval_report.json")
    if not p.exists():
        print(f"Warning: {p} not found. Run compare_all_policies.py first.")
        return None
    with open(p) as f:
        return json.load(f)


def load_history(policy: str) -> list | None:
    p = Path(HISTORY_FILES.get(policy, ""))
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Figure 1: Policy comparison bar chart (train posts)
# ---------------------------------------------------------------------------

def plot_comparison_bar(report: dict, post_type: str = "train_posts", suffix: str = "train"):
    data = report.get(post_type, {})
    conditions = [c for c in ["random", "engagement", "trex", "ground_truth"] if c in data]

    metrics = [
        ("mean_true_sat",   "True Satisfaction",  [0, 1]),
        ("return_rate",     "Return Rate",         [0, 1]),
        ("ndcg_at_8",       "NDCG@8",             [0, 1]),
        ("precision_at_40", "Precision@40",        [0, 1]),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4.5))
    fig.suptitle(
        "Policy Comparison — " + ("Training Posts (clusters 0–3)" if suffix == "train" else "Held-Out Cluster 4"),
        fontsize=13, fontweight="bold", y=1.02,
    )

    x = np.arange(len(conditions))
    width = 0.65

    for ax, (key, label, ylim) in zip(axes, metrics):
        vals = [data[c].get(key, 0.0) for c in conditions]
        colors = [POLICY_COLORS[c] for c in conditions]
        bars = ax.bar(x, vals, width=width, color=colors, edgecolor="white", linewidth=0.8)

        # Value labels on bars
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8,
            )

        ax.set_title(label, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([POLICY_LABELS[c] for c in conditions], fontsize=9)
        ax.set_ylim(ylim[0], min(ylim[1], max(vals) * 1.18 + 0.02))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=9)

    plt.tight_layout()
    out = FIGURES / f"policy_comparison_{suffix}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# Figure 2: Training curves (true_sat over iterations, all policies)
# ---------------------------------------------------------------------------

def plot_training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: rollout true_sat
    ax1 = axes[0]
    ax1.set_title("Rollout True Satisfaction (training)", fontsize=12)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Mean True Satisfaction")

    # Right: eval true_sat
    ax2 = axes[1]
    ax2.set_title("Eval True Satisfaction (greedy, every 20 iters)", fontsize=12)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Mean True Satisfaction")

    any_plotted = False
    for policy in ["engagement", "trex", "ground_truth"]:
        history = load_history(policy)
        if history is None:
            continue
        any_plotted = True
        color = POLICY_COLORS[policy]
        label = POLICY_LABELS[policy].replace("\n", " ")

        iters = [r["iteration"] for r in history]
        rollout_sat = [r["rollout_mean_true_sat"] for r in history]

        # Smooth rollout curve with 10-iter rolling mean
        smoothed = np.convolve(rollout_sat, np.ones(10) / 10, mode="same")
        ax1.plot(iters, smoothed, color=color, lw=2, label=label)

        eval_rows = [(r["iteration"], r["eval_mean_true_sat"])
                     for r in history if "eval_mean_true_sat" in r]
        if eval_rows:
            e_iters, e_vals = zip(*eval_rows)
            ax2.plot(e_iters, e_vals, color=color, lw=2, marker="o", ms=5, label=label)

    if not any_plotted:
        print("Warning: no training histories found, skipping training curves figure.")
        plt.close()
        return

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=9)
        ax.tick_params(labelsize=9)

    plt.tight_layout()
    out = FIGURES / "training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# Figure 3: Reward alignment — T-REX reward vs. oracle true_sat per policy
# ---------------------------------------------------------------------------

def plot_reward_alignment(report: dict):
    """Bar: T-REX reward score vs. true_sat for each policy (train posts)."""
    data = report.get("train_posts", {})
    conditions = [c for c in ["random", "engagement", "trex", "ground_truth"] if c in data]
    trex_rewards = [data[c].get("mean_reward", None) for c in conditions]
    true_sats = [data[c].get("mean_true_sat", 0.0) for c in conditions]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(conditions))
    width = 0.38

    # Normalize T-REX rewards to [0,1] for visual alignment with true_sat
    valid = [r for r in trex_rewards if r is not None]
    if valid:
        rmin, rmax = min(valid), max(valid)
        norm_rewards = [
            (r - rmin) / (rmax - rmin + 1e-9) if r is not None else 0.0
            for r in trex_rewards
        ]

        bars1 = ax.bar(x - width / 2, norm_rewards, width, color="#f28e2b",
                       alpha=0.85, label="T-REX reward (normalized)")
    bars2 = ax.bar(x + width / 2, true_sats, width, color="#59a14f",
                   alpha=0.85, label="True satisfaction (oracle)")

    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_LABELS[c].replace("\n", " ") for c in conditions], fontsize=10)
    ax.set_ylabel("Score")
    ax.set_title("T-REX Reward vs. Oracle Satisfaction by Policy\n(T-REX reward normalized to [0,1])", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = FIGURES / "reward_alignment.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# Figure 4: Delta from random baseline
# ---------------------------------------------------------------------------

def plot_delta_from_random(report: dict):
    """Show improvement over random for each policy on key metrics."""
    for post_type, suffix in [("train_posts", "train"), ("cluster4_eval", "cluster4")]:
        data = report.get(post_type, {})
        if "random" not in data:
            continue

        baseline = data["random"]
        conditions = [c for c in ["engagement", "trex", "ground_truth"] if c in data]
        metrics = [
            ("mean_true_sat",   "Δ True Satisfaction"),
            ("return_rate",     "Δ Return Rate"),
            ("ndcg_at_8",       "Δ NDCG@8"),
        ]

        fig, axes = plt.subplots(1, len(metrics), figsize=(10, 4))
        title = "Train Posts" if suffix == "train" else "Cluster 4 (Held-Out)"
        fig.suptitle(f"Improvement Over Random Baseline — {title}", fontsize=12, fontweight="bold")

        x = np.arange(len(conditions))
        width = 0.55

        for ax, (key, label) in zip(axes, metrics):
            deltas = [data[c].get(key, 0.0) - baseline.get(key, 0.0) for c in conditions]
            colors = [POLICY_COLORS[c] for c in conditions]
            bars = ax.bar(x, deltas, width=width, color=colors, edgecolor="white")

            for bar, val in zip(bars, deltas):
                ypos = bar.get_height() + (0.002 if val >= 0 else -0.012)
                ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                        f"{val:+.3f}", ha="center", va="bottom", fontsize=9)

            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_title(label, fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels([POLICY_LABELS[c].replace("\n", " ") for c in conditions], fontsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        plt.tight_layout()
        out = FIGURES / f"delta_from_random_{suffix}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    report = load_eval_report()

    # Always generate training curves (uses history files, not eval report)
    plot_training_curves()

    if report is None:
        print("Skipping bar charts and alignment figures (no eval report).")
        return

    plot_comparison_bar(report, post_type="train_posts", suffix="train")
    plot_comparison_bar(report, post_type="cluster4_eval", suffix="cluster4")
    plot_delta_from_random(report)
    plot_reward_alignment(report)

    print(f"\nAll figures saved to {FIGURES}/")


if __name__ == "__main__":
    main()
