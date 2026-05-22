"""Reward functions for different policy training objectives."""

from __future__ import annotations

import numpy as np


def get_reward_fn(reward_type: str):
    """Get reward function for given type."""
    if reward_type == "engagement":
        return reward_engagement
    elif reward_type == "trex":
        return reward_trex
    elif reward_type == "ground_truth":
        return reward_ground_truth
    else:
        raise ValueError(f"Unknown reward_type: {reward_type}")


def reward_engagement(session_dict: dict, **kwargs) -> float:
    """Engagement score: behavioral ranking signal (pre-reward-network)."""
    from pipeline.ranking import score_session
    return score_session(session_dict)


def reward_trex(session_dict: dict, scorer=None, **kwargs) -> float:
    """Learned reward from implicit signals via reward network."""
    if scorer is None:
        raise ValueError("scorer required for T-REX reward")
    return scorer.score(session_dict)


def reward_ground_truth(session_dict: dict, post_records=None, **kwargs) -> float:
    """Oracle: mean true affinity (ground truth, never used in training)."""
    if post_records is None:
        return 0.0
    return float(np.mean([p.true_affinity for p in post_records]))
