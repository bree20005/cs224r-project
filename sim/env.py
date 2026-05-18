"""
SocialFeedEnv: simulates users browsing a chronological social feed.

Each session consists of POSTS_PER_SESSION posts shown sequentially.
For every post the user can click, dwell, scroll, or bail early.
At the end of the session we record whether the user returned for the
next session — this is the primary behavioral ranking signal.

Engagement model
----------------
True affinity:  a = u · p  (dot product of unit vectors, ∈ [-1, 1])
Click prob:     P(click | a) = σ( a / noise_level )
Dwell time:     Lognormal(μ=1.5 + 2*a, σ=0.5) if clicked, else Uniform(0.5, 2)
Scroll depth:   clipped Beta-ish draw correlated with a
Early exit:     probabilistic after each post based on running mean affinity
Next-session return:  P(return) = σ( 8 * mean_session_affinity )
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List

from sim.config import (
    NUM_USERS, POSTS_PER_SESSION, SESSIONS_PER_USER,
    DEFAULT_NOISE, RANDOM_SEED,
)
from sim.content import ContentLibrary
from sim.users import UserPopulation


@dataclass
class PostRecord:
    post_id: int
    cluster: int
    embedding: np.ndarray           # (16,)
    clicked: bool
    dwell_time: float               # seconds
    scroll_depth: float             # 0.0 – 1.0
    true_affinity: float            # u·p — ground truth, never used in training


@dataclass
class SessionRecord:
    session_id: str
    user_id: int
    session_idx: int
    posts: List[PostRecord]
    early_exit: bool
    next_session_return: bool
    true_satisfaction: float        # mean u·p over seen posts — ground truth
    engagement_score: float         # derived ranking signal for T-REX pairing


class SocialFeedEnv:
    """
    Simulates the full corpus of behavioral traces for the T-REX pipeline.

    Parameters
    ----------
    noise_level : float
        Temperature τ in P(click) = σ(a / τ).
        Low τ  → clicks tightly track true affinity (clean signal).
        High τ → clicks are nearly random (noisy signal).
    """

    def __init__(
        self,
        content: ContentLibrary,
        users: UserPopulation,
        noise_level: float = DEFAULT_NOISE,
        seed: int = RANDOM_SEED + 2,
    ):
        self.content = content
        self.users = users
        self.noise_level = noise_level
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Per-post engagement model
    # ------------------------------------------------------------------

    def _click_prob(self, affinity: float) -> float:
        """σ(a / τ) — sigmoid with temperature τ = noise_level."""
        return float(1 / (1 + np.exp(-affinity / self.noise_level)))

    def _dwell_time(self, affinity: float, clicked: bool) -> float:
        if not clicked:
            return float(self.rng.uniform(0.5, 2.0))
        # Lognormal: higher affinity → longer dwell
        mu = 1.5 + 2.0 * affinity      # at a=0.7 → mu≈2.9 → median≈18s
        return float(np.exp(self.rng.normal(mu, 0.5)))

    def _scroll_depth(self, affinity: float, clicked: bool) -> float:
        if not clicked:
            # Brief scroll-past
            return float(np.clip(self.rng.beta(1, 4), 0.0, 1.0))
        # Scroll deeper for content that aligns with preferences
        base = (affinity + 1) / 2        # map [-1,1] → [0,1]
        noisy = base + self.rng.normal(0, 0.1)
        return float(np.clip(noisy, 0.0, 1.0))

    def _simulate_post(self, user_id: int, post_id: int) -> PostRecord:
        emb = self.content.embeddings[post_id]
        a = self.users.true_affinity(user_id, emb)

        # Add observation noise to the affinity seen by the engagement model
        a_obs = a + self.rng.normal(0, self.noise_level * 0.4)
        click_p = self._click_prob(a_obs)
        clicked = bool(self.rng.random() < click_p)

        return PostRecord(
            post_id=post_id,
            cluster=int(self.content.cluster_ids[post_id]),
            embedding=emb.copy(),
            clicked=clicked,
            dwell_time=self._dwell_time(a_obs, clicked),
            scroll_depth=self._scroll_depth(a_obs, clicked),
            true_affinity=a,        # no noise — this is the ground truth
        )

    # ------------------------------------------------------------------
    # Session simulation
    # ------------------------------------------------------------------

    def simulate_session(
        self,
        user_id: int,
        session_idx: int,
        post_ids: np.ndarray,
    ) -> SessionRecord:
        """Simulate one session; post_ids is an ordered array of POSTS_PER_SESSION ids."""
        posts: List[PostRecord] = []
        running_affinity = 0.0
        early_exit = False

        for t, pid in enumerate(post_ids):
            rec = self._simulate_post(user_id, int(pid))
            posts.append(rec)
            running_affinity += rec.true_affinity

            # Early-exit model: user may quit mid-session if feed is boring
            if t >= 2:
                mean_so_far = running_affinity / (t + 1)
                # Exit only when mean affinity is negative (user is annoyed)
                exit_p = np.clip(-mean_so_far, 0.0, 1.0) * 0.25
                if self.rng.random() < exit_p:
                    early_exit = True
                    break

        affinities = [p.true_affinity for p in posts]
        true_satisfaction = float(np.mean(affinities))

        # Next-session return — primary ranking signal
        # σ(8 * satisfaction): at sat≈0.15 → return_p≈0.77; at sat<0 → return_p<0.5
        return_logit = 8.0 * true_satisfaction
        return_prob = float(1 / (1 + np.exp(-return_logit)))
        next_session_return = bool(self.rng.random() < return_prob)

        # Composite engagement score for Komal's ranking pipeline
        # Primary weight on next-session return; remaining signals as tiebreakers
        click_rate = float(np.mean([p.clicked for p in posts]))
        mean_dwell = float(np.mean([p.dwell_time for p in posts]))
        norm_dwell = min(mean_dwell / 60.0, 1.0)   # cap at 60 s
        engagement_score = (
            0.50 * float(next_session_return)
            + 0.20 * click_rate
            + 0.20 * norm_dwell
            + 0.10 * float(not early_exit)
        )

        return SessionRecord(
            session_id=f"u{user_id:03d}_s{session_idx:02d}",
            user_id=user_id,
            session_idx=session_idx,
            posts=posts,
            early_exit=early_exit,
            next_session_return=next_session_return,
            true_satisfaction=true_satisfaction,
            engagement_score=engagement_score,
        )

    # ------------------------------------------------------------------
    # Full corpus generation
    # ------------------------------------------------------------------

    def simulate_all(self, use_train_posts: bool = True) -> List[SessionRecord]:
        """
        Simulate all 200 users × 20 sessions.
        Posts are sampled uniformly at random from the training pool (clusters 0–3)
        to produce the behavioral traces that T-REX learns from.
        """
        pool = (
            self.content.train_post_ids if use_train_posts
            else self.content.eval_post_ids
        )
        all_sessions: List[SessionRecord] = []
        for uid in range(NUM_USERS):
            for sid in range(SESSIONS_PER_USER):
                post_ids = self.rng.choice(pool, size=POSTS_PER_SESSION, replace=False)
                all_sessions.append(self.simulate_session(uid, sid, post_ids))
        return all_sessions
