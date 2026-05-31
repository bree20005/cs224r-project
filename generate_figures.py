#!/usr/bin/env python3
"""Generate report figures from saved training histories."""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


if __name__ == "__main__":
    plot_reward_model()
    plot_policy_comparison()
    plot_trex_individual()
    print("Done.")
