#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from models.reward_net import FrozenRewardScorer
from ppo.policy import ActorCritic
from ppo.reward_fn import get_reward_fn
from ppo.runner import collect_episode, collect_rollouts
from sim.config import RANDOM_SEED
from sim.content import ContentLibrary
from sim.env import SocialFeedEnv
from sim.users import UserPopulation


N_ITERATIONS = 200
N_EPISODES_PER_ITER = 64
PPO_EPOCHS = 4
CLIP_EPS = 0.2
VF_COEF = 0.5
ENT_COEF = 0.01
GAMMA = 0.99
GAE_LAMBDA = 0.95
LR = 3e-4
MAX_GRAD_NORM = 0.5
HIDDEN_DIM = 64
EVAL_INTERVAL = 20    
N_EVAL_EPISODES = 200


def evaluate_policy(
    policy: ActorCritic,
    env: SocialFeedEnv,
    users: UserPopulation,
    content: ContentLibrary,
    device: torch.device,
    reward_fn: callable = None,
    scorer = None,
    n_episodes: int = N_EVAL_EPISODES,
) -> dict:
    """Greedy rollouts over the first n_episodes users."""
    rewards, true_sats, early_exits = [], [], []
    for uid in range(n_episodes):
        _, stats = collect_episode(
            policy, env, uid, content, users, device,
            reward_fn=reward_fn,
            scorer=scorer,
            greedy=True,
        )
        rewards.append(stats["reward"])
        true_sats.append(stats["true_satisfaction"])
        early_exits.append(stats["early_exit"])
    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_true_sat": float(np.mean(true_sats)),
        "early_exit_rate": float(np.mean(early_exits)),
    }


def random_baseline(
    env: SocialFeedEnv,
    users: UserPopulation,
    content: ContentLibrary,
    reward_fn: callable = None,
    scorer = None,
    n_episodes: int = N_EVAL_EPISODES,
) -> dict:
    """Evaluate random post selection as the baseline."""
    rewards, true_sats, early_exits = [], [], []
    rng = np.random.default_rng(0)
    for uid in range(n_episodes):
        post_ids = rng.choice(content.train_post_ids, size=8, replace=False)
        rec = env.simulate_session(uid, session_idx=999, post_ids=post_ids)
        session_dict = {
            "posts": [
                {
                    "post_id": p.post_id,
                    "clicked": p.clicked,
                    "dwell_time": p.dwell_time,
                    "scroll_depth": p.scroll_depth,
                }
                for p in rec.posts
            ],
            "early_exit": rec.early_exit,
            "next_session_return": rec.next_session_return,
        }

        if reward_fn is None:
            from ppo.reward_fn import reward_trex
            reward_fn = reward_trex

        reward = reward_fn(session_dict, scorer=scorer, post_records=rec.posts)
        rewards.append(reward)
        true_sats.append(rec.true_satisfaction)
        early_exits.append(rec.early_exit)
    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_true_sat": float(np.mean(true_sats)),
        "early_exit_rate": float(np.mean(early_exits)),
    }


def train(
    data_dir: str = "data",
    checkpoint_dir: str = "checkpoints",
    reward_ckpt: str = None,
    seed: int = RANDOM_SEED,
    use_wandb: bool = False,
    wandb_project: str = "cs224r-trex",
    wandb_run_name: str = "ppo",
    reward_type: str = "trex",
):
    if reward_ckpt is None:
        reward_ckpt = f"{checkpoint_dir}/reward_net_frozen_seed{seed}.pt"
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"reward_type: {reward_type}")

    
    content = ContentLibrary(seed=seed)
    users = UserPopulation(content, seed=seed + 1)
    env = SocialFeedEnv(content, users, seed=seed + 10)

    
    scorer = None
    if reward_type == "trex":
        scorer = FrozenRewardScorer(reward_ckpt, data_dir=data_dir)

    reward_fn = get_reward_fn(reward_type)

    policy = ActorCritic(hidden_dim=HIDDEN_DIM).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=LR)

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    
    print("Computing random baseline…")
    base = random_baseline(env, users, content, reward_fn=reward_fn, scorer=scorer)
    print(
        f"  random baseline  reward={base['mean_reward']:.4f}  "
        f"true_sat={base['mean_true_sat']:.4f}  "
        f"early_exit={base['early_exit_rate']:.2%}"
    )

    if use_wandb:
        import wandb
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "reward_type": reward_type,
                "n_iterations": N_ITERATIONS,
                "n_episodes_per_iter": N_EPISODES_PER_ITER,
                "ppo_epochs": PPO_EPOCHS,
                "clip_eps": CLIP_EPS,
                "vf_coef": VF_COEF,
                "ent_coef": ENT_COEF,
                "gamma": GAMMA,
                "gae_lambda": GAE_LAMBDA,
                "lr": LR,
                "hidden_dim": HIDDEN_DIM,
                "seed": seed,
            },
        )
        wandb.run.summary["baseline/reward"] = base["mean_reward"]
        wandb.run.summary["baseline/true_sat"] = base["mean_true_sat"]
        wandb.run.summary["baseline/early_exit_rate"] = base["early_exit_rate"]

    history = []

    for iteration in range(1, N_ITERATIONS + 1):
        policy.eval()
        buffer, rollout_stats = collect_rollouts(
            policy, env, users, content,
            n_episodes=N_EPISODES_PER_ITER,
            device=device,
            reward_fn=reward_fn,
            scorer=scorer,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
        )

        # Normalize advantages across the full buffer
        norm_advs = buffer.normalized_advantages()

        policy.train()
        pg_losses, vf_losses, entropies = [], [], []

        for _ in range(PPO_EPOCHS):
            for idx in buffer.shuffled_indices():
                trans, _, ret = buffer.get(idx)
                adv = float(norm_advs[idx])

                state = trans.state.to(device)
                post_embs = trans.post_embs.to(device)
                action = trans.action.to(device)
                ret_t = torch.tensor(ret, dtype=torch.float32, device=device)
                adv_t = torch.tensor(adv, dtype=torch.float32, device=device)
                old_lp = torch.tensor(trans.old_log_prob, dtype=torch.float32, device=device)

                _, new_lp, entropy, new_val = policy.get_action_and_value(
                    state, post_embs, action
                )

                ratio = (new_lp - old_lp).exp()
                pg_loss = torch.max(
                    -adv_t * ratio,
                    -adv_t * ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS),
                )
                vf_loss = (new_val - ret_t).pow(2)
                loss = pg_loss + VF_COEF * vf_loss - ENT_COEF * entropy

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
                opt.step()

                pg_losses.append(pg_loss.item())
                vf_losses.append(vf_loss.item())
                entropies.append(entropy.item())

        row = {
            "iteration": iteration,
            "rollout_mean_reward": rollout_stats["mean_reward"],
            "rollout_mean_true_sat": rollout_stats["mean_true_sat"],
            "rollout_early_exit_rate": rollout_stats["early_exit_rate"],
            "pg_loss": float(np.mean(pg_losses)),
            "vf_loss": float(np.mean(vf_losses)),
            "entropy": float(np.mean(entropies)),
        }

        eval_stats = None
        if iteration % EVAL_INTERVAL == 0:
            policy.eval()
            eval_stats = evaluate_policy(
                policy, env, users, content, device,
                reward_fn=reward_fn,
                scorer=scorer,
            )
            row["eval_mean_reward"] = eval_stats["mean_reward"]
            row["eval_mean_true_sat"] = eval_stats["mean_true_sat"]
            row["eval_early_exit_rate"] = eval_stats["early_exit_rate"]
            print(
                f"[{iteration:4d}]  "
                f"rollout_reward={row['rollout_mean_reward']:.4f}  "
                f"eval_reward={eval_stats['mean_reward']:.4f}  "
                f"eval_true_sat={eval_stats['mean_true_sat']:.4f}  "
                f"entropy={row['entropy']:.3f}"
            )
        else:
            print(
                f"[{iteration:4d}]  "
                f"rollout_reward={row['rollout_mean_reward']:.4f}  "
                f"pg_loss={row['pg_loss']:.4f}  "
                f"vf_loss={row['vf_loss']:.4f}  "
                f"entropy={row['entropy']:.3f}"
            )

        if use_wandb:
            import wandb
            log = {
                "train/pg_loss": row["pg_loss"],
                "train/vf_loss": row["vf_loss"],
                "train/entropy": row["entropy"],
                "rollout/reward": row["rollout_mean_reward"],
                "rollout/true_sat": row["rollout_mean_true_sat"],
                "rollout/early_exit_rate": row["rollout_early_exit_rate"],
                "baseline/reward": base["mean_reward"],
                "baseline/true_sat": base["mean_true_sat"],
            }
            if eval_stats is not None:
                log.update({
                    "eval/reward": eval_stats["mean_reward"],
                    "eval/true_sat": eval_stats["mean_true_sat"],
                    "eval/early_exit_rate": eval_stats["early_exit_rate"],
                    "eval/true_sat_delta": eval_stats["mean_true_sat"] - base["mean_true_sat"],
                    "eval/reward_delta": eval_stats["mean_reward"] - base["mean_reward"],
                })
            wandb.log(log, step=iteration)

        history.append(row)

        if iteration % 50 == 0:
            torch.save(
                {
                    "state_dict": policy.state_dict(),
                    "model_config": {"hidden_dim": HIDDEN_DIM},
                    "iteration": iteration,
                    "reward_type": reward_type,
                    "seed": seed,
                },
                ckpt_dir / f"ppo_policy_{reward_type}_seed{seed}_iter{iteration}.pt",
            )

    torch.save(
        {
            "state_dict": policy.state_dict(),
            "model_config": {"hidden_dim": HIDDEN_DIM},
            "iteration": N_ITERATIONS,
            "reward_type": reward_type,
            "seed": seed,
            "random_baseline": base,
        },
        ckpt_dir / f"ppo_policy_{reward_type}_seed{seed}.pt",
    )
    with open(ckpt_dir / f"ppo_train_history_{reward_type}_seed{seed}.json", "w") as f:
        json.dump(history, f, indent=2)

    if use_wandb:
        import wandb
        last_eval = next(
            (r for r in reversed(history) if "eval_mean_true_sat" in r), None
        )
        if last_eval:
            wandb.run.summary["final/eval_true_sat"] = last_eval["eval_mean_true_sat"]
            wandb.run.summary["final/eval_reward"] = last_eval["eval_mean_reward"]
            wandb.run.summary["final/true_sat_delta"] = (
                last_eval["eval_mean_true_sat"] - base["mean_true_sat"]
            )
        artifact = wandb.Artifact(f"ppo_policy_{reward_type}_seed{seed}", type="model")
        artifact.add_file(str(ckpt_dir / f"ppo_policy_{reward_type}_seed{seed}.pt"))
        artifact.add_file(str(ckpt_dir / f"ppo_train_history_{reward_type}_seed{seed}.json"))
        wandb.log_artifact(artifact)
        wandb.finish()

    print(f"\nFinal policy ({reward_type}) saved → {ckpt_dir / f'ppo_policy_{reward_type}_seed{seed}.pt'}")
    return history, base


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train PPO policy")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--reward-type", default="trex", choices=["trex", "engagement", "ground_truth"])
    parser.add_argument("--reward-ckpt", default=None, help="Override reward checkpoint path")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="cs224r-trex")
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()

    run_name = args.wandb_run_name or f"ppo-{args.reward_type}-seed{args.seed}"
    train(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        reward_ckpt=args.reward_ckpt,
        seed=args.seed,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=run_name,
        reward_type=args.reward_type,
    )
