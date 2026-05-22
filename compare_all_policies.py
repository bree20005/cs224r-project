#!/usr/bin/env python3
"""
Compare all 4 policies: π_random, π_engagement, π_T-REX, π_ground_truth.

Evaluates on:
  - Train posts (clusters 0–3)
  - Eval posts (cluster 4, held-out)

Usage
-----
    python compare_all_policies.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from models.reward_net import FrozenRewardScorer
from ppo.policy import ActorCritic
from ppo.reward_fn import get_reward_fn
from ppo.runner import collect_episode
from sim.config import RANDOM_SEED
from sim.content import ContentLibrary
from sim.env import SocialFeedEnv
from sim.users import UserPopulation

POLICIES = ["engagement", "trex", "ground_truth"]
N_EVAL_USERS = 200


def evaluate_policy(
    policy: ActorCritic,
    env: SocialFeedEnv,
    users: UserPopulation,
    content: ContentLibrary,
    reward_type: str,
    scorer=None,
    post_pool=None,
    device=torch.device("cpu"),
) -> dict:
    if post_pool is None:
        post_pool = content.train_post_ids

    original_pool = content.train_post_ids
    content.train_post_ids = post_pool

    reward_fn = get_reward_fn(reward_type)
    rewards, true_sats, early_exits, return_flags = [], [], [], []

    for uid in range(N_EVAL_USERS):
        _, stats = collect_episode(
            policy, env, uid, content, users, device,
            reward_fn=reward_fn,
            scorer=scorer,
            greedy=True,
        )
        rewards.append(stats["reward"])
        true_sats.append(stats["true_satisfaction"])
        early_exits.append(stats["early_exit"])

        # Simulate next-session return
        ret_prob = 1 / (1 + np.exp(-8.0 * stats["true_satisfaction"]))
        return_flags.append(float(np.random.random() < ret_prob))

    content.train_post_ids = original_pool
    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_true_sat": float(np.mean(true_sats)),
        "early_exit_rate": float(np.mean(early_exits)),
        "return_rate": float(np.mean(return_flags)),
    }


def main():
    np.random.seed(RANDOM_SEED)
    device = torch.device("cpu")

    content = ContentLibrary(seed=RANDOM_SEED)
    users = UserPopulation(content, seed=RANDOM_SEED + 1)
    env = SocialFeedEnv(content, users, seed=RANDOM_SEED + 30)

    scorer = FrozenRewardScorer("checkpoints/reward_net_frozen.pt", data_dir="data")

    # Load all policies
    policies = {}
    for policy_type in POLICIES:
        ckpt_file = Path(f"checkpoints/ppo_policy_{policy_type}.pt")
        if not ckpt_file.exists():
            print(f"Warning: {ckpt_file} not found. Run train_all_policies.py first.")
            continue

        saved = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        policy = ActorCritic(**saved["model_config"]).to(device)
        policy.load_state_dict(saved["state_dict"])
        policy.eval()
        policies[policy_type] = policy

    if not policies:
        raise SystemExit("No trained policies found.")

    report = {}

    for post_type, post_pool in [
        ("train_posts", content.train_post_ids),
        ("cluster4_eval", content.eval_post_ids),
    ]:
        print(f"\nEvaluating on {post_type}…")
        report[post_type] = {}

        for policy_type, policy in policies.items():
            eval_stats = evaluate_policy(
                policy, env, users, content, policy_type,
                scorer=scorer,
                post_pool=post_pool,
                device=device,
            )
            report[post_type][policy_type] = eval_stats
            print(f"  {policy_type:15s}  true_sat={eval_stats['mean_true_sat']:.4f}  "
                  f"return_rate={eval_stats['return_rate']:.2%}  "
                  f"reward={eval_stats['mean_reward']:.4f}")

    # Print comparison table
    print("\n" + "="*70)
    print("POLICY COMPARISON: TRAIN POSTS (clusters 0–3)")
    print("="*70)
    print(f"{'Policy':15s}  {'True Sat':>10s}  {'Return Rate':>12s}  {'Early Exit':>11s}")
    for policy in POLICIES:
        if policy in report["train_posts"]:
            s = report["train_posts"][policy]
            print(f"{policy:15s}  {s['mean_true_sat']:10.4f}  {s['return_rate']:12.2%}  "
                  f"{s['early_exit_rate']:11.2%}")

    print("\n" + "="*70)
    print("POLICY COMPARISON: CLUSTER 4 (HELD-OUT GENERALIZATION)")
    print("="*70)
    print(f"{'Policy':15s}  {'True Sat':>10s}  {'Return Rate':>12s}  {'Early Exit':>11s}")
    for policy in POLICIES:
        if policy in report["cluster4_eval"]:
            s = report["cluster4_eval"][policy]
            print(f"{policy:15s}  {s['mean_true_sat']:10.4f}  {s['return_rate']:12.2%}  "
                  f"{s['early_exit_rate']:11.2%}")
    print("="*70)

    with open("checkpoints/policy_eval_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved → checkpoints/policy_eval_report.json")


if __name__ == "__main__":
    main()
