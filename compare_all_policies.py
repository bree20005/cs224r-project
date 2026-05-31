#!/usr/bin/env python3
"""
Compare all policies: π_random, π_engagement, π_T-REX, π_ground_truth.

Metrics (computed on held-out greedy rollouts):
  - mean_true_sat       — oracle satisfaction signal (never used in training)
  - return_rate         — simulated next-session return probability
  - early_exit_rate     — fraction of sessions where user bailed early
  - ndcg_at_8          — NDCG@8 measuring ranking quality vs. oracle affinity
  - precision_at_40    — fraction of shown posts in true top-40 for each user

Evaluates on:
  - Train posts (clusters 0–3)
  - Eval posts (cluster 4, held-out generalization)

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
from sim.config import RANDOM_SEED, POSTS_PER_SESSION
from sim.content import ContentLibrary
from sim.env import SocialFeedEnv
from sim.users import UserPopulation

POLICIES = ["engagement", "trex", "ground_truth"]
N_EVAL_USERS = 200


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------

def ndcg_at_k(
    policy_post_affinities: list,
    all_post_affinities: list,
    k: int = 8,
) -> float:
    """
    NDCG@k comparing the policy's ordered session against the oracle top-k.

    policy_post_affinities : true-affinity values in selection order
    all_post_affinities    : true-affinity values for every post in the pool
    """
    dcg = sum(
        max(0.0, aff) / np.log2(i + 2)
        for i, aff in enumerate(policy_post_affinities[:k])
    )
    ideal = sorted(all_post_affinities, reverse=True)[:k]
    idcg = sum(max(0.0, aff) / np.log2(i + 2) for i, aff in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(
    policy_post_ids: list,
    all_post_ids: list,
    users: UserPopulation,
    user_id: int,
    content,
    k: int = 40,
) -> float:
    """Fraction of policy's shown posts that fall in the user's true top-k."""
    affinities = [(pid, users.true_affinity(user_id, content.embeddings[pid]))
                  for pid in all_post_ids]
    top_k = {pid for pid, _ in sorted(affinities, key=lambda x: x[1], reverse=True)[:k]}
    hits = sum(1 for pid in policy_post_ids if pid in top_k)
    return hits / max(len(policy_post_ids), 1)


# ---------------------------------------------------------------------------
# Policy evaluation
# ---------------------------------------------------------------------------

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
    ndcgs, precs = [], []

    all_pool_affinities_cache: dict[int, list] = {}

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

        ret_prob = 1 / (1 + np.exp(-8.0 * stats["true_satisfaction"]))
        return_flags.append(float(np.random.random() < ret_prob))

        # Compute NDCG and Precision using post_records returned by collect_episode
        post_records = stats.get("post_records", [])
        if post_records:
            if uid not in all_pool_affinities_cache:
                all_pool_affinities_cache[uid] = [
                    users.true_affinity(uid, content.embeddings[pid])
                    for pid in post_pool
                ]
            all_affs = all_pool_affinities_cache[uid]
            session_affs = [pr.true_affinity for pr in post_records]
            session_ids = [pr.post_id for pr in post_records]
            ndcgs.append(ndcg_at_k(session_affs, all_affs, k=POSTS_PER_SESSION))
            precs.append(precision_at_k(session_ids, list(post_pool), users, uid, content, k=40))

    content.train_post_ids = original_pool
    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_true_sat": float(np.mean(true_sats)),
        "early_exit_rate": float(np.mean(early_exits)),
        "return_rate": float(np.mean(return_flags)),
        "ndcg_at_8": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "precision_at_40": float(np.mean(precs)) if precs else 0.0,
    }


def evaluate_random(
    env: SocialFeedEnv,
    users: UserPopulation,
    content: ContentLibrary,
    post_pool=None,
) -> dict:
    """Random post selection baseline with same metrics."""
    if post_pool is None:
        post_pool = content.train_post_ids

    rng = np.random.default_rng(1)
    true_sats, early_exits, return_flags, ndcgs, precs = [], [], [], [], []

    for uid in range(N_EVAL_USERS):
        post_ids = rng.choice(post_pool, size=min(POSTS_PER_SESSION, len(post_pool)), replace=False)
        rec = env.simulate_session(uid, session_idx=998, post_ids=post_ids)

        true_sats.append(rec.true_satisfaction)
        early_exits.append(rec.early_exit)
        ret_prob = 1 / (1 + np.exp(-8.0 * rec.true_satisfaction))
        return_flags.append(float(np.random.random() < ret_prob))

        all_affs = [users.true_affinity(uid, content.embeddings[pid]) for pid in post_pool]
        session_affs = [p.true_affinity for p in rec.posts]
        session_ids = [p.post_id for p in rec.posts]
        ndcgs.append(ndcg_at_k(session_affs, all_affs, k=POSTS_PER_SESSION))
        precs.append(precision_at_k(session_ids, list(post_pool), users, uid, content, k=40))

    return {
        "mean_reward": 0.0,  # random has no learned reward
        "mean_true_sat": float(np.mean(true_sats)),
        "early_exit_rate": float(np.mean(early_exits)),
        "return_rate": float(np.mean(return_flags)),
        "ndcg_at_8": float(np.mean(ndcgs)),
        "precision_at_40": float(np.mean(precs)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    np.random.seed(RANDOM_SEED)
    device = torch.device("cpu")

    content = ContentLibrary(seed=RANDOM_SEED)
    users = UserPopulation(content, seed=RANDOM_SEED + 1)
    env = SocialFeedEnv(content, users, seed=RANDOM_SEED + 30)

    scorer = FrozenRewardScorer("checkpoints/reward_net_frozen.pt", data_dir="data")

    # Load all trained policies
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
        print(f"Loaded {policy_type} policy from {ckpt_file}")

    if not policies:
        raise SystemExit("No trained policies found. Run train_all_policies.py first.")

    report = {}

    for post_type, post_pool in [
        ("train_posts", content.train_post_ids),
        ("cluster4_eval", content.eval_post_ids),
    ]:
        print(f"\nEvaluating on {post_type}…")
        report[post_type] = {}

        # Random baseline
        print("  random baseline…")
        report[post_type]["random"] = evaluate_random(env, users, content, post_pool)

        # Learned policies
        for policy_type, policy in policies.items():
            print(f"  {policy_type}…")
            report[post_type][policy_type] = evaluate_policy(
                policy, env, users, content, policy_type,
                scorer=scorer,
                post_pool=post_pool,
                device=device,
            )

    # Print comparison tables
    all_conditions = ["random"] + [p for p in POLICIES if p in policies]
    metrics = [
        ("mean_true_sat",   "True Sat",      ".4f"),
        ("return_rate",     "Return Rate",   ".2%"),
        ("early_exit_rate", "Early Exit",    ".2%"),
        ("ndcg_at_8",       "NDCG@8",        ".4f"),
        ("precision_at_40", "Prec@40",       ".4f"),
    ]

    for post_type in ("train_posts", "cluster4_eval"):
        label = "TRAIN POSTS (clusters 0–3)" if post_type == "train_posts" else "CLUSTER 4 (HELD-OUT GENERALIZATION)"
        print(f"\n{'='*80}")
        print(f"POLICY COMPARISON: {label}")
        print(f"{'='*80}")
        header = f"{'Policy':15s}"
        for _, col, _ in metrics:
            header += f"  {col:>13s}"
        print(header)
        print("-" * 80)
        for cond in all_conditions:
            if cond not in report[post_type]:
                continue
            s = report[post_type][cond]
            row = f"{cond:15s}"
            for key, _, fmt in metrics:
                val = s.get(key, 0.0)
                row += f"  {format(val, fmt.lstrip('.')):>13}"
            print(row)
        print("=" * 80)

    with open("checkpoints/policy_eval_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved → checkpoints/policy_eval_report.json")


if __name__ == "__main__":
    main()
