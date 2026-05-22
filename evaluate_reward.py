#!/usr/bin/env python3
"""
Reward-network evaluation.

Metrics:
  - Pearson r(predicted reward, true_satisfaction) on all sessions
  - Pairwise ranking accuracy on behavioral labels (held-out users)
  - Optional: sessions containing eval-cluster posts (cluster 4)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from models.reward_net import FrozenRewardScorer
from pipeline.dataset import load_sessions_and_pairs, train_val_split_pairs
from train_reward import pearson_vs_true_sat, ranking_accuracy


def main():
    data_dir = Path("data")
    ckpt = Path("checkpoints/reward_net_frozen.pt")
    if not ckpt.exists():
        raise SystemExit("Train first: python train_reward.py")

    sessions, pairs, post_embeddings = load_sessions_and_pairs(data_dir)
    sessions_by_id = {s["session_id"]: s for s in sessions}
    _, val_pairs = train_val_split_pairs(pairs, val_fraction=0.15, seed=42)

    scorer = FrozenRewardScorer(ckpt, data_dir=data_dir)
    model = scorer.model

    import torch
    device = torch.device("cpu")
    model.to(device)

    preds = scorer.score_batch(sessions)
    true_sat = np.array([s["true_satisfaction"] for s in sessions])
    pearson_all = float(np.corrcoef(preds, true_sat)[0, 1])

    val_rank = ranking_accuracy(
        model, val_pairs, sessions_by_id, post_embeddings, device
    )

    # Sessions with any held-out cluster post (generalization probe)
    eval_cluster_sessions = [
        s for s in sessions
        if any(p["cluster"] == 4 for p in s["posts"])
    ]
    if eval_cluster_sessions:
        ec_preds = scorer.score_batch(eval_cluster_sessions)
        ec_sat = [s["true_satisfaction"] for s in eval_cluster_sessions]
        pearson_cluster4 = float(np.corrcoef(ec_preds, ec_sat)[0, 1])
    else:
        pearson_cluster4 = None
        print(
            "Note: default sessions.json uses train posts only. "
            "Re-simulate with use_train_posts=False for cluster-4 eval."
        )

    report = {
        "pearson_reward_vs_true_sat_all": pearson_all,
        "val_pairwise_ranking_accuracy": val_rank,
        "n_val_pairs": len(val_pairs),
        "pearson_cluster4_sessions": pearson_cluster4,
        "n_cluster4_sessions": len(eval_cluster_sessions),
    }
    print(json.dumps(report, indent=2))
    with open("checkpoints/reward_eval_report.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
