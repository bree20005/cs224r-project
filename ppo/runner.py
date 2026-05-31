"""Episode collection: runs the policy in SocialFeedEnv and fills a RolloutBuffer."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from models.reward_net import FrozenRewardScorer
from ppo.buffer import RolloutBuffer, Transition
from ppo.policy import ActorCritic, HISTORY_DIM
from sim.config import POSTS_PER_SESSION
from sim.content import ContentLibrary
from sim.env import SocialFeedEnv
from sim.users import UserPopulation


def collect_episode(
    policy: ActorCritic,
    env: SocialFeedEnv,
    user_id: int,
    content: ContentLibrary,
    users: UserPopulation,
    device: torch.device,
    reward_fn: callable = None,
    scorer: FrozenRewardScorer = None,
    greedy: bool = False,
) -> Tuple[List[Transition], Dict]:
    user_emb = torch.from_numpy(
        users.preferences[user_id].astype(np.float32)
    ).to(device)

    available_ids: List[int] = list(content.train_post_ids)
    history: List[torch.Tensor] = []   # grows: each entry is (E+B,)
    post_records = []
    transitions: List[Transition] = []

    running_affinity = 0.0
    early_exit = False

    for step in range(POSTS_PER_SESSION):
        hist_t = (
            torch.stack(history)
            if history
            else torch.zeros(0, HISTORY_DIM, device=device)
        )
        state = policy.encode_state(user_emb, hist_t)

        avail_embs = torch.from_numpy(
            content.embeddings[available_ids].astype(np.float32)
        ).to(device)

        with torch.no_grad():
            action_idx, log_prob, _, value = policy.get_action_and_value(
                state, avail_embs, greedy=greedy
            )

        post_id = available_ids[action_idx.item()]
        post_rec = env._simulate_post(user_id, post_id)
        post_records.append(post_rec)

        beh = np.array(
            [
                float(post_rec.clicked),
                min(post_rec.dwell_time / 60.0, 1.0),
                post_rec.scroll_depth,
            ],
            dtype=np.float32,
        )
        history.append(
            torch.from_numpy(
                np.concatenate([content.embeddings[post_id].astype(np.float32), beh])
            ).to(device)
        )
        available_ids.remove(post_id)

        transitions.append(
            Transition(
                state=state.cpu(),
                post_embs=avail_embs.cpu(),
                action=action_idx.cpu(),
                old_log_prob=log_prob.item(),
                value=value.item(),
                reward=0.0,
                done=False,
            )
        )

        # Replicate env early-exit dynamics (uses true_affinity from sim, not policy input)
        running_affinity += post_rec.true_affinity
        if step >= 2:
            mean_aff = running_affinity / (step + 1)
            exit_p = float(np.clip(-mean_aff, 0.0, 1.0)) * 0.25
            if env.rng.random() < exit_p:
                early_exit = True
                break

    true_sat = float(np.mean([p.true_affinity for p in post_records]))

    # Simulate next-session return — observed behavioral signal (mirrors env.py)
    ret_prob = float(1 / (1 + np.exp(-8.0 * true_sat)))
    next_session_return = bool(env.rng.random() < ret_prob)

    session_dict = {
        "posts": [
            {
                "post_id": p.post_id,
                "clicked": p.clicked,
                "dwell_time": p.dwell_time,
                "scroll_depth": p.scroll_depth,
            }
            for p in post_records
        ],
        "early_exit": early_exit,
        "next_session_return": next_session_return,
    }

    if reward_fn is None:
        from ppo.reward_fn import reward_trex
        reward_fn = reward_trex

    reward = reward_fn(
        session_dict,
        scorer=scorer,
        post_records=post_records,
    )

    transitions[-1].reward = reward
    transitions[-1].done = True

    return transitions, {
        "reward": reward,
        "true_satisfaction": true_sat,
        "early_exit": early_exit,
        "n_posts_shown": len(post_records),
        "post_records": post_records,
    }


def collect_rollouts(
    policy: ActorCritic,
    env: SocialFeedEnv,
    users: UserPopulation,
    content: ContentLibrary,
    n_episodes: int,
    device: torch.device,
    reward_fn: callable = None,
    scorer: FrozenRewardScorer = None,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[RolloutBuffer, Dict]:
    buffer = RolloutBuffer()
    rewards, true_sats, early_exits = [], [], []

    user_ids = np.random.randint(0, len(users.preferences), size=n_episodes)
    for uid in user_ids:
        ep_trans, stats = collect_episode(
            policy, env, int(uid), content, users, device,
            reward_fn=reward_fn,
            scorer=scorer,
        )
        buffer.add_episode(ep_trans)
        rewards.append(stats["reward"])
        true_sats.append(stats["true_satisfaction"])
        early_exits.append(stats["early_exit"])

    buffer.compute_gae(gamma=gamma, gae_lambda=gae_lambda)
    return buffer, {
        "mean_reward": float(np.mean(rewards)),
        "mean_true_sat": float(np.mean(true_sats)),
        "early_exit_rate": float(np.mean(early_exits)),
    }
