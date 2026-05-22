#!/usr/bin/env python3
"""
Train trajectory reward network with Bradley-Terry pairwise loss.

Never uses true_satisfaction, true_affinity, or user_preferences.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.reward_net import (
    TrajectoryRewardNet,
    bradley_terry_loss,
    session_to_tensor,
)
from pipeline.dataset import (
    PairwisePreferenceDataset,
    load_sessions_and_pairs,
    train_val_split_pairs,
)


def ranking_accuracy(model, pairs, sessions_by_id, post_embeddings, device) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for p in pairs:
            win = sessions_by_id[p["winner_id"]]
            lose = sessions_by_id[p["loser_id"]]
            rw = model(session_to_tensor(win, post_embeddings, device=device).unsqueeze(0))
            rl = model(session_to_tensor(lose, post_embeddings, device=device).unsqueeze(0))
            if rw.item() > rl.item():
                correct += 1
    return correct / max(len(pairs), 1)


def pearson_vs_true_sat(model, sessions, post_embeddings, device) -> float:
    """Diagnostic only — true_sat is never a training target."""
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for s in sessions:
            r = model(session_to_tensor(s, post_embeddings, device=device).unsqueeze(0))
            preds.append(r.item())
            truths.append(s["true_satisfaction"])
    return float(np.corrcoef(preds, truths)[0, 1])


def train(
    data_dir: str = "data",
    epochs: int = 40,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_fraction: float = 0.15,
    seed: int = 42,
    checkpoint_dir: str = "checkpoints",
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sessions, pairs, post_embeddings = load_sessions_and_pairs(data_dir)
    sessions_by_id = {s["session_id"]: s for s in sessions}
    train_pairs, val_pairs = train_val_split_pairs(pairs, val_fraction=val_fraction, seed=seed)

    train_ds = PairwisePreferenceDataset(train_pairs, sessions_by_id, post_embeddings)
    val_ds = PairwisePreferenceDataset(val_pairs, sessions_by_id, post_embeddings)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = TrajectoryRewardNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for x_win, x_lose in train_loader:
            x_win, x_lose = x_win.to(device), x_lose.to(device)
            r_win = model(x_win)
            r_lose = model(x_lose)
            loss = bradley_terry_loss(r_win, r_lose)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
            n_batches += 1

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for x_win, x_lose in val_loader:
                x_win, x_lose = x_win.to(device), x_lose.to(device)
                val_loss += bradley_terry_loss(model(x_win), model(x_lose)).item()
                val_batches += 1

        val_rank_acc = ranking_accuracy(
            model, val_pairs, sessions_by_id, post_embeddings, device
        )
        val_pearson = pearson_vs_true_sat(model, sessions, post_embeddings, device)

        row = {
            "epoch": epoch,
            "train_bt_loss": train_loss / max(n_batches, 1),
            "val_bt_loss": val_loss / max(val_batches, 1),
            "val_ranking_accuracy": val_rank_acc,
            "all_sessions_pearson_true_sat": val_pearson,
        }
        history.append(row)
        print(
            f"epoch {epoch:3d}  train_bt={row['train_bt_loss']:.4f}  "
            f"val_bt={row['val_bt_loss']:.4f}  val_rank_acc={val_rank_acc:.3f}  "
            f"pearson(sat)={val_pearson:.4f}"
        )

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Freeze for policy use — eval mode + no grad
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    frozen_path = ckpt_dir / "reward_net_frozen.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": {"embed_dim": 16, "behavior_dim": 3, "hidden_dim": 64},
            "train_pairs": len(train_pairs),
            "val_pairs": len(val_pairs),
            "epochs": epochs,
            "final_val_ranking_accuracy": history[-1]["val_ranking_accuracy"],
            "note": "Use models.reward_net.FrozenRewardScorer for PPO integration",
        },
        frozen_path,
    )

    with open(ckpt_dir / "reward_train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nFrozen reward network → {frozen_path}")
    return history


if __name__ == "__main__":
    train()
