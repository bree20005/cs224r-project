"""PyTorch dataset over ranked session pairs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from models.reward_net import session_to_tensor


class PairwisePreferenceDataset(Dataset):
    def __init__(
        self,
        pairs: List[dict],
        sessions_by_id: Dict[str, dict],
        post_embeddings: np.ndarray,
    ):
        self.pairs = pairs
        self.sessions_by_id = sessions_by_id
        self.post_embeddings = post_embeddings

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self.pairs[idx]
        win = self.sessions_by_id[p["winner_id"]]
        lose = self.sessions_by_id[p["loser_id"]]
        return (
            session_to_tensor(win, self.post_embeddings),
            session_to_tensor(lose, self.post_embeddings),
        )


def load_sessions_and_pairs(
    data_dir: str | Path,
) -> Tuple[List[dict], List[dict], np.ndarray]:
    data_dir = Path(data_dir)
    with open(data_dir / "sessions.json") as f:
        sessions = json.load(f)
    with open(data_dir / "ranked_pairs.json") as f:
        pairs = json.load(f)
    post_embeddings = np.load(data_dir / "post_embeddings.npy")
    return sessions, pairs, post_embeddings


def train_val_split_pairs(
    pairs: List[dict],
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[List[dict], List[dict]]:
    """Split by user so no user's pairs leak across train/val."""
    rng = np.random.default_rng(seed)
    users = sorted({p["user_id"] for p in pairs})
    rng.shuffle(users)
    n_val = max(1, int(len(users) * val_fraction))
    val_users = set(users[:n_val])
    train_pairs = [p for p in pairs if p["user_id"] not in val_users]
    val_pairs = [p for p in pairs if p["user_id"] in val_users]
    return train_pairs, val_pairs
