#!/usr/bin/env python3
"""
Final evaluation: PPO policy vs random baseline.

Metrics (all on held-out greedy rollouts, ground truth never used in training):
  - mean reward (FrozenRewardScorer — what the policy optimizes)
  - mean true_satisfaction (oracle signal — measures actual user welfare)
  - next-session return rate
  - early-exit rate
  - generalization: true_satisfaction on sessions containing cluster-4 posts

Usage
-----
    python evaluate_ppo.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from models.reward_net import FrozenRewardScorer
from ppo.policy import ActorCritic
from ppo.runner import collect_episode
from sim.config import RANDOM_SEED, POSTS_PER_SESSION
from sim.content import ContentLibrary
from sim.env import SocialFeedEnv
from sim.users import UserPopulation

N_EVAL_USERS = 200


def run_ppo_eval(
    policy: ActorCritic,
    env: SocialFeedEnv,
    scorer: FrozenRewardScorer,
    users: UserPopulation,
    content: ContentLibrary,
    device: torch.device,
    post_pool: np.ndarray,
) -> dict:
    """Greedy rollouts; post_pool controls which posts are available (train or eval)."""
    original_train_ids = content.train_post_ids
    content.train_post_ids = post_pool  # temporarily swap pool

    rewards, true_sats, early_exits, return_flags = [], [], [], []
    for uid in range(N_EVAL_USERS):
        _, stats = collect_episode(
            policy, env, scorer, uid, content, users, device, greedy=True
        )
        rewards.append(stats["reward"])
        true_sats.append(stats["true_satisfaction"])
        early_exits.append(stats["early_exit"])
        # Simulate next-session return probability from true_sat
        ret_prob = 1 / (1 + np.exp(-8.0 * stats["true_satisfaction"]))
        return_flags.append(float(np.random.random() < ret_prob))

    content.train_post_ids = original_train_ids  # restore
    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_true_sat": float(np.mean(true_sats)),
        "early_exit_rate": float(np.mean(early_exits)),
        "return_rate": float(np.mean(return_flags)),
    }


def run_random_eval(
    env: SocialFeedEnv,
    scorer: FrozenRewardScorer,
    users: UserPopulation,
    content: ContentLibrary,
    post_pool: np.ndarray,
) -> dict:
    rng = np.random.default_rng(1)
    rewards, true_sats, early_exits, return_flags = [], [], [], []
    for uid in range(N_EVAL_USERS):
        post_ids = rng.choice(post_pool, size=min(POSTS_PER_SESSION, len(post_pool)), replace=False)
        rec = env.simulate_session(uid, session_idx=998, post_ids=post_ids)
        session_dict = {
            "posts": [
                {
                    "post_id": p.post_id,
                    "clicked": p.clicked,
                    "dwell_time": p.dwell_time,
                    "scroll_depth": p.scroll_depth,
                }
                for p in rec.posts
            ]
        }
        rewards.append(scorer.score(session_dict))
        true_sats.append(rec.true_satisfaction)
        early_exits.append(rec.early_exit)
        ret_prob = 1 / (1 + np.exp(-8.0 * rec.true_satisfaction))
        return_flags.append(float(np.random.random() < ret_prob))
    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_true_sat": float(np.mean(true_sats)),
        "early_exit_rate": float(np.mean(early_exits)),
        "return_rate": float(np.mean(return_flags)),
    }


def main():
    ckpt = Path("checkpoints/ppo_policy.pt")
    reward_ckpt = Path("checkpoints/reward_net_frozen.pt")
    if not ckpt.exists():
        raise SystemExit("Train first: python train_ppo.py")

    np.random.seed(RANDOM_SEED)
    device = torch.device("cpu")

    content = ContentLibrary(seed=RANDOM_SEED)
    users = UserPopulation(content, seed=RANDOM_SEED + 1)
    env = SocialFeedEnv(content, users, seed=RANDOM_SEED + 20)
    scorer = FrozenRewardScorer(reward_ckpt, data_dir="data")

    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    policy = ActorCritic(**saved["model_config"]).to(device)
    policy.load_state_dict(saved["state_dict"])
    policy.eval()

    train_pool = content.train_post_ids   # clusters 0–3, 400 posts
    eval_pool = content.eval_post_ids     # cluster 4, 100 posts

    print("Evaluating on train-distribution posts (clusters 0–3)…")
    ppo_train = run_ppo_eval(policy, env, scorer, users, content, device, train_pool)
    rand_train = run_random_eval(env, scorer, users, content, train_pool)

    print("Evaluating generalization on held-out cluster 4 posts…")
    ppo_eval = run_ppo_eval(policy, env, scorer, users, content, device, eval_pool)
    rand_eval = run_random_eval(env, scorer, users, content, eval_pool)

    report = {
        "train_posts": {
            "ppo": ppo_train,
            "random": rand_train,
            "delta_true_sat": round(ppo_train["mean_true_sat"] - rand_train["mean_true_sat"], 4),
            "delta_return_rate": round(ppo_train["return_rate"] - rand_train["return_rate"], 4),
        },
        "cluster4_generalization": {
            "ppo": ppo_eval,
            "random": rand_eval,
            "delta_true_sat": round(ppo_eval["mean_true_sat"] - rand_eval["mean_true_sat"], 4),
        },
    }

    print("\n" + "=" * 60)
    print("TRAIN POSTS (clusters 0–3)")
    print(f"  {'':20s}  {'PPO':>10}  {'Random':>10}  {'Delta':>8}")
    for k in ("mean_true_sat", "return_rate", "early_exit_rate", "mean_reward"):
        print(
            f"  {k:20s}  {ppo_train[k]:10.4f}  {rand_train[k]:10.4f}  "
            f"{ppo_train[k]-rand_train[k]:+8.4f}"
        )
    print("\nCLUSTER 4 GENERALIZATION (held-out)")
    print(f"  {'':20s}  {'PPO':>10}  {'Random':>10}  {'Delta':>8}")
    for k in ("mean_true_sat", "return_rate", "early_exit_rate", "mean_reward"):
        print(
            f"  {k:20s}  {ppo_eval[k]:10.4f}  {rand_eval[k]:10.4f}  "
            f"{ppo_eval[k]-rand_eval[k]:+8.4f}"
        )
    print("=" * 60)

    with open("checkpoints/ppo_eval_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nReport saved → checkpoints/ppo_eval_report.json")


if __name__ == "__main__":
    main()
