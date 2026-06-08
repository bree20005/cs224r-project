#!/usr/bin/env python3
"""
Train all 4 policies for comparison: π_random, π_engagement, π_T-REX, π_ground_truth.
"""

from __future__ import annotations

import json
from pathlib import Path

from train_ppo import train

POLICIES = ["engagement", "trex", "ground_truth"]


def train_all(
    n_iterations: int = 200,
    use_wandb: bool = False,
    seed: int = 42,
):
    results = {}

    for policy_type in POLICIES:
        ckpt_file = Path(f"checkpoints/ppo_policy_{policy_type}_seed{seed}.pt")
        if ckpt_file.exists():
            print(f"\nSkipping {policy_type.upper()} — checkpoint already exists: {ckpt_file}")
            continue

        print(f"\n{'='*60}")
        print(f"Training {policy_type.upper()}")
        print(f"{'='*60}")

        history, base = train(
            seed=seed,
            use_wandb=use_wandb,
            reward_type=policy_type,
            wandb_run_name=f"ppo-{policy_type}",
        )

        last_eval = next(
            (r for r in reversed(history) if "eval_mean_true_sat" in r), None
        )
        if last_eval:
            results[policy_type] = {
                "baseline_true_sat": base["mean_true_sat"],
                "baseline_reward": base["mean_reward"],
                "final_eval_true_sat": last_eval["eval_mean_true_sat"],
                "final_eval_reward": last_eval["eval_mean_reward"],
                "true_sat_delta": last_eval["eval_mean_true_sat"] - base["mean_true_sat"],
            }

    print(f"\n{'='*60}")
    print("POLICY COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Policy':15s}  {'Final True Sat':>15s}  {'True Sat Δ':>12s}")
    for policy in POLICIES:
        if policy in results:
            r = results[policy]
            print(
                f"{policy:15s}  {r['final_eval_true_sat']:15.4f}  "
                f"{r['true_sat_delta']:+12.4f}"
            )

    with open("checkpoints/policy_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nComparison saved → checkpoints/policy_comparison.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train all PPO policies")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    train_all(use_wandb=args.wandb, seed=args.seed)
