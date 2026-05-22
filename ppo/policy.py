"""
ActorCritic for the feed-ranking PPO agent.

State = user_emb (16) + mean-pooled behavioral history (16+3=19) = 35-d.
Actor scores each candidate post as f(state, post_emb, user·post) -> scalar logit.
The explicit dot-product similarity feature lets the policy generalize to
unseen post clusters by directly measuring user-post affinity.
Critic estimates V(state).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from sim.config import EMBED_DIM

BEHAVIOR_DIM = 3
HISTORY_DIM = EMBED_DIM + BEHAVIOR_DIM   # 19
STATE_DIM = EMBED_DIM + HISTORY_DIM      # 35


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim), nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        nn.Linear(hidden_dim, out_dim),
    )


class ActorCritic(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM, hidden_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        state_dim = embed_dim + HISTORY_DIM
        # +1 for explicit user·post dot-product similarity feature
        self.actor = _mlp(state_dim + embed_dim + 1, hidden_dim, 1)
        self.critic = _mlp(state_dim, hidden_dim, 1)

    def encode_state(
        self,
        user_emb: torch.Tensor,   # (E,)
        history: torch.Tensor,    # (T, E+B) — T=0 means no history yet
    ) -> torch.Tensor:            # (STATE_DIM,)
        if history.shape[0] > 0:
            hist_mean = history.mean(dim=0)
        else:
            hist_mean = torch.zeros(HISTORY_DIM, device=user_emb.device)
        return torch.cat([user_emb, hist_mean])

    def actor_logits(
        self,
        state: torch.Tensor,      # (STATE_DIM,)
        post_embs: torch.Tensor,  # (N, E)
    ) -> torch.Tensor:            # (N,)
        N = post_embs.shape[0]
        user_emb = state[:self.embed_dim]                    # (E,)
        sim = (post_embs @ user_emb).unsqueeze(-1)           # (N, 1) explicit u·p affinity
        s = state.unsqueeze(0).expand(N, -1)                 # (N, STATE_DIM)
        return self.actor(torch.cat([s, post_embs, sim], dim=-1)).squeeze(-1)

    def value(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic(state).squeeze(-1)

    def get_action_and_value(
        self,
        state: torch.Tensor,
        post_embs: torch.Tensor,
        action: torch.Tensor | None = None,
        greedy: bool = False,
    ):
        logits = self.actor_logits(state, post_embs)
        dist = Categorical(logits=logits)
        if action is None:
            action = logits.argmax() if greedy else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.value(state)
