"""Rollout buffer: stores episode transitions and computes GAE advantages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch


@dataclass
class Transition:
    state: torch.Tensor      # (STATE_DIM,)
    post_embs: torch.Tensor  # (N_avail, E)  — available posts at this step
    action: torch.Tensor     # scalar int — index into post_embs
    old_log_prob: float
    value: float
    reward: float
    done: bool


class RolloutBuffer:
    def __init__(self):
        self._episodes: List[List[Transition]] = []
        self._flat: List[Transition] = []
        self._advantages: List[float] = []
        self._returns: List[float] = []

    def add_episode(self, episode: List[Transition]) -> None:
        self._episodes.append(episode)
        self._flat.extend(episode)

    def compute_gae(self, gamma: float = 0.99, gae_lambda: float = 0.95) -> None:
        self._advantages.clear()
        self._returns.clear()
        for episode in self._episodes:
            T = len(episode)
            advs = [0.0] * T
            gae = 0.0
            next_val = 0.0
            for t in reversed(range(T)):
                tr = episode[t]
                cont = 0.0 if tr.done else 1.0
                delta = tr.reward + gamma * next_val * cont - tr.value
                gae = delta + gamma * gae_lambda * cont * gae
                advs[t] = gae
                next_val = tr.value
            rets = [advs[t] + episode[t].value for t in range(T)]
            self._advantages.extend(advs)
            self._returns.extend(rets)

    def normalized_advantages(self) -> np.ndarray:
        a = np.array(self._advantages, dtype=np.float32)
        return (a - a.mean()) / (a.std() + 1e-8)

    def shuffled_indices(self) -> List[int]:
        idxs = list(range(len(self._flat)))
        np.random.shuffle(idxs)
        return idxs

    def get(self, idx: int) -> Tuple[Transition, float, float]:
        return self._flat[idx], self._advantages[idx], self._returns[idx]

    def __len__(self) -> int:
        return len(self._flat)

    def clear(self) -> None:
        self._episodes.clear()
        self._flat.clear()
        self._advantages.clear()
        self._returns.clear()
