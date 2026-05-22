"""
Reward network: trajectory (sequence of post interactions) → scalar reward.

Uses frozen post embeddings from data/post_embeddings.npy; only behavioral
features (click, dwell, scroll) are observed at inference time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from sim.config import EMBED_DIM, POSTS_PER_SESSION


def _post_behavior_features(post: dict) -> np.ndarray:
    """3-d behavioral vector per post (no ground-truth affinity)."""
    dwell = min(post["dwell_time"] / 60.0, 1.0)
    return np.array(
        [float(post["clicked"]), dwell, post["scroll_depth"]],
        dtype=np.float32,
    )


def session_to_tensor(
    session: dict,
    post_embeddings: np.ndarray,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Build (POSTS_PER_SESSION, EMBED_DIM + 3) tensor; zero-pad if user exited early.
    """
    rows = []
    for post in session["posts"]:
        emb = post_embeddings[post["post_id"]].astype(np.float32)
        beh = _post_behavior_features(post)
        rows.append(np.concatenate([emb, beh]))
    # Pad to fixed length for batched training
    while len(rows) < POSTS_PER_SESSION:
        rows.append(np.zeros(EMBED_DIM + 3, dtype=np.float32))
    x = np.stack(rows[:POSTS_PER_SESSION], axis=0)
    return torch.from_numpy(x).to(device)


class TrajectoryRewardNet(nn.Module):
    """
    Per-post MLP → masked mean pool over valid posts → scalar reward.
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        behavior_dim: int = 3,
        hidden_dim: int = 64,
    ):
        super().__init__()
        in_dim = embed_dim + behavior_dim
        self.post_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, D) post feature sequences
        lengths : (B,) optional count of non-padding posts per trajectory
        """
        h = self.post_encoder(x)  # (B, T, H)
        if lengths is None:
            # Mask: non-zero embedding norm indicates a real post
            mask = (x[..., :EMBED_DIM].pow(2).sum(dim=-1) > 1e-8).float()
        else:
            mask = torch.zeros(x.size(0), x.size(1), device=x.device)
            for b, L in enumerate(lengths):
                mask[b, : int(L)] = 1.0
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (h * mask.unsqueeze(-1)).sum(dim=1) / denom
        return self.head(pooled).squeeze(-1)  # (B,)


def bradley_terry_loss(r_win: torch.Tensor, r_lose: torch.Tensor) -> torch.Tensor:
    """-log σ(r_win - r_lose) = softplus(-(r_win - r_lose))."""
    return torch.nn.functional.softplus(-(r_win - r_lose)).mean()


class FrozenRewardScorer:
    """Load checkpoint for PPO / evaluation (friend's integration point)."""

    def __init__(self, checkpoint_path: str | Path, data_dir: str | Path = "data"):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model = TrajectoryRewardNet(**ckpt["model_config"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.post_embeddings = np.load(Path(data_dir) / "post_embeddings.npy")

    @torch.no_grad()
    def score(self, session: dict) -> float:
        x = session_to_tensor(session, self.post_embeddings)
        return float(self.model(x.unsqueeze(0)).item())

    @torch.no_grad()
    def score_batch(self, sessions: List[dict], device: str = "cpu") -> np.ndarray:
        self.model.to(device)
        xs = [
            session_to_tensor(s, self.post_embeddings, device=device)
            for s in sessions
        ]
        batch = torch.stack(xs, dim=0)
        return self.model(batch).cpu().numpy()
