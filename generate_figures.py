#!/usr/bin/env python3
"""Generate report figures from saved training histories."""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

FIGS = Path("figures")
CKPT = Path("checkpoints")

ENGAGEMENT_COLOR = "#E05C5C"
TREX_COLOR = "#5C8BE0"

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})


# ---------------------------------------------------------------------------
# Figure 1: Reward model training
# ---------------------------------------------------------------------------

def plot_reward_model():
    data = json.load(open(CKPT / "reward_train_history.json"))
    epochs      = [d["epoch"] for d in data]
    train_loss  = [d["train_bt_loss"] for d in data]
    val_loss    = [d["val_bt_loss"] for d in data]
    val_acc     = [d["val_ranking_accuracy"] for d in data]
    pearson     = [d["all_sessions_pearson_true_sat"] for d in data]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Loss
    ax = axes[0]
    ax.plot(epochs, train_loss, label="Train BT Loss", color="#444")
    ax.plot(epochs, val_loss,   label="Val BT Loss",   color=TREX_COLOR)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Bradley–Terry Loss")
    ax.set_title("Reward Model: Loss")
    ax.legend(fontsize=9)

    # Ranking accuracy
    ax = axes[1]
    ax.plot(epochs, val_acc, color=ENGAGEMENT_COLOR)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Pairwise Ranking Accuracy")
    ax.set_title("Reward Model: Val Ranking Accuracy")
    ax.set_ylim(0.75, 1.0)

    # Pearson
    ax = axes[2]
    ax.plot(epochs, pearson, color="#6BAA6B")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Pearson r")
    ax.set_title("Reward Model: Pearson r (Reward vs True Sat)")
    ax.set_ylim(0.0, 1.0)

    fig.tight_layout()
    out = FIGS / "reward_model_training.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 2: Engagement vs T-REX training comparison
# ---------------------------------------------------------------------------

def smooth(values, window=5):
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_policy_comparison():
    eng  = json.load(open(CKPT / "ppo_train_history.json"))
    trex = json.load(open(CKPT / "ppo_train_history_trex.json"))

    def extract(data, key):
        return [d["iteration"] for d in data if key in d], \
               [d[key]        for d in data if key in d]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # --- rollout true_sat ---
    ax = axes[0]
    iters_e, sat_e = extract(eng,  "rollout_mean_true_sat")
    iters_t, sat_t = extract(trex, "rollout_mean_true_sat")
    w = 5
    ax.plot(iters_e[w-1:], smooth(sat_e), color=ENGAGEMENT_COLOR, label="Engagement PPO", linewidth=1.8)
    ax.plot(iters_t[w-1:], smooth(sat_t), color=TREX_COLOR,       label="T-REX PPO",      linewidth=1.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("True Satisfaction")
    ax.set_title("Rollout: True Satisfaction")
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=9)

    # --- eval true_sat ---
    ax = axes[1]
    iters_e, sat_e = extract(eng,  "eval_mean_true_sat")
    iters_t, sat_t = extract(trex, "eval_mean_true_sat")
    ax.plot(iters_e, sat_e, color=ENGAGEMENT_COLOR, label="Engagement PPO", linewidth=1.8, marker="o", markersize=4)
    ax.plot(iters_t, sat_t, color=TREX_COLOR,       label="T-REX PPO",      linewidth=1.8, marker="s", markersize=4)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("True Satisfaction")
    ax.set_title("Eval: True Satisfaction")
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=9)

    # --- entropy ---
    ax = axes[2]
    iters_e, ent_e = extract(eng,  "entropy")
    iters_t, ent_t = extract(trex, "entropy")
    w = 5
    ax.plot(iters_e[w-1:], smooth(ent_e), color=ENGAGEMENT_COLOR, label="Engagement PPO", linewidth=1.8)
    ax.plot(iters_t[w-1:], smooth(ent_t), color=TREX_COLOR,       label="T-REX PPO",      linewidth=1.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Policy Entropy")
    ax.set_title("Policy Entropy (Exploration)")
    ax.legend(fontsize=9)

    fig.suptitle("PPO Training: Engagement Reward vs T-REX Reward", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = FIGS / "policy_training_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 3: Individual T-REX training curves
# ---------------------------------------------------------------------------

def plot_single(data, key, ylabel, title, out_path, color=TREX_COLOR, ylim=None, smoothed=False):
    iters = [d["iteration"] for d in data if key in d]
    vals  = [d[key]         for d in data if key in d]
    w = 5
    if smoothed:
        iters = iters[w-1:]
        vals  = list(smooth(vals))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(iters, vals, color=color, linewidth=1.6)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_trex_individual():
    trex = json.load(open(CKPT / "ppo_train_history_trex.json"))

    plot_single(trex, "rollout_mean_true_sat", "True Satisfaction", "rollout/true_sat",
                FIGS / "rollout_true_sat_trex.png", smoothed=True)
    plot_single(trex, "eval_mean_true_sat", "True Satisfaction", "eval/true_sat",
                FIGS / "eval_true_sat_trex.png")
    plot_single(trex, "rollout_mean_reward", "Reward", "rollout/reward",
                FIGS / "rollout_reward_trex.png", smoothed=True)
    plot_single(trex, "entropy", "Policy Entropy", "train/entropy",
                FIGS / "train_entropy_trex.png", smoothed=True)


# ---------------------------------------------------------------------------
# Figure 4: Method diagrams
# ---------------------------------------------------------------------------

def make_box(ax, x, y, w, h, text, color, fontsize=11, radius=0.04):
    box = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=color, edgecolor="white", linewidth=1.5, zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color="white", zorder=4, multialignment="center")


def arrow(ax, x0, y0, x1, y1, color="#555555"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.8, mutation_scale=18), zorder=2)


def plot_diagram_reward_pipeline():
    """Diagram 1: T-REX reward learning pipeline."""

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    BOX_H = 1.1
    BOX_W = 2.2
    Y = 2.0

    BLUE   = "#4A7FC1"
    ORANGE = "#E07B39"
    GREEN  = "#4A9E6B"
    PURPLE = "#7B5EA7"

    boxes = [
        (1.4,  Y, "User\nSessions",         BLUE),
        (4.0,  Y, "Behavioral\nSignals",    BLUE),
        (6.6,  Y, "Pairwise\nRankings",     ORANGE),
        (9.2,  Y, "Reward\nModel  r_θ",    PURPLE),
        (12.2, Y, "Learned\nReward",        GREEN),
    ]

    for x, y, txt, col in boxes:
        make_box(ax, x, y, BOX_W, BOX_H, txt, col, fontsize=10.5)

    # Arrows between boxes
    gaps = [(2.5, 3.0), (5.1, 5.6), (7.7, 8.2), (10.3, 11.1)]
    labels = [
        "early exit\nreturn prob\nengagement",
        "Bradley-Terry\npairwise loss",
        "session pairs\nranked by signal",
        "",
    ]
    for (x0, x1), lbl in zip(gaps, labels):
        arrow(ax, x0, Y, x1, Y)
        if lbl:
            ax.text((x0 + x1) / 2, Y + 0.78, lbl,
                    ha="center", va="bottom", fontsize=8, color="#444444",
                    multialignment="center")

    ax.set_title("Diagram 1 — T-REX Reward Learning Pipeline",
                 fontsize=13, fontweight="bold", pad=8, color="#222222")

    fig.tight_layout()
    out = FIGS / "diagram_reward_pipeline.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def plot_diagram_training_loop():
    """Diagram 2: PPO training loop — clean orthogonal layout."""

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    BLUE   = "#4A7FC1"
    ORANGE = "#E07B39"
    GRAY   = "#7A8A9A"
    PURPLE = "#7B5EA7"

    BH = 0.90

    # ── Boxes: vertical center column + Reward Model to the left ──────────
    make_box(ax, 5.5, 6.6, 3.0, BH, "Social Feed\nEnvironment",              BLUE,   fontsize=10)
    make_box(ax, 5.5, 4.5, 3.2, BH, "PPO Policy  π_T-REX\n(Actor-Critic + user emb.)", PURPLE, fontsize=9.5)
    make_box(ax, 5.5, 2.4, 3.0, BH, "User Simulator",                         GRAY,   fontsize=10)
    make_box(ax, 1.5, 4.5, 2.6, BH, "Reward Model  r_θ\n(frozen)",            ORANGE, fontsize=9.5)

    def sa(x0, y0, x1, y1, col="#555555"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=col,
                                   lw=1.8, mutation_scale=18), zorder=2)

    # ── Arrow 1: Env → Policy (straight down) ─────────────────────────────
    sa(5.5, 6.15, 5.5, 4.95)
    ax.text(5.72, 5.55, "state: user + post\nembeddings",
            ha="left", va="center", fontsize=8.5, color="#333333")

    # ── Arrow 2: Policy → User Sim (straight down) ────────────────────────
    sa(5.5, 4.05, 5.5, 2.85)
    ax.text(5.72, 3.45, "action: ranked\npost selection",
            ha="left", va="center", fontsize=8.5, color="#333333")

    # ── Arrow 3: User Sim → Reward (L-shape: left then up) ────────────────
    # Path: User Sim left edge → horizontal left → vertical up → Reward bottom
    ax.plot([4.0, 1.5], [2.4, 2.4], color="#555555", lw=1.8, zorder=2,
            solid_capstyle="round")
    ax.plot([1.5, 1.5], [2.4, 4.05], color="#555555", lw=1.8, zorder=2,
            solid_capstyle="round")
    sa(1.5, 3.92, 1.5, 4.05)   # arrowhead cap at top of vertical segment
    ax.text(2.75, 2.05, "session behavior",
            ha="center", va="top", fontsize=8.5, color="#333333")

    # ── Arrow 4: Reward → Policy (straight horizontal right) ──────────────
    sa(2.8, 4.5, 3.9, 4.5)
    ax.text(3.35, 4.72, "reward  r",
            ha="center", va="bottom", fontsize=8.5, color="#333333")

    # ── Arrow 5: User Sim → Env (curved arc on right, next session) ───────
    ax.annotate("", xy=(7.0, 6.15), xytext=(7.0, 2.85),
                arrowprops=dict(arrowstyle="-|>", color="#aaaaaa",
                               lw=1.5, mutation_scale=15,
                               connectionstyle="arc3,rad=-0.45"), zorder=2)
    ax.text(8.7, 4.5, "next\nsession",
            ha="center", va="center", fontsize=8.5, color="#888888")

    ax.set_title("Diagram 2 — PPO Training Loop",
                 fontsize=13, fontweight="bold", pad=8, color="#222222")

    fig.tight_layout()
    out = FIGS / "diagram_training_loop.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    plot_reward_model()
    plot_policy_comparison()
    plot_trex_individual()
    plot_diagram_reward_pipeline()
    plot_diagram_training_loop()
    print("Done.")
