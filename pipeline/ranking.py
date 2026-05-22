"""
Trajectory ranking pipeline — behavioral labels only (no human labels, no ground truth).

Primary signal: next_session_return
Tiebreakers: click rate, normalized dwell, session completion (not early_exit)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from sim.config import NUM_USERS, SESSIONS_PER_USER

# Weights mirror sim/env.py — keep in sync for reproducible T-REX pairing
W_RETURN = 0.50
W_CLICK = 0.20
W_DWELL = 0.20
W_COMPLETE = 0.10
DWELL_CAP_SEC = 60.0


def score_session(session: dict) -> float:
    """Weighted engagement score; next-session return is the dominant term."""
    posts = session["posts"]
    click_rate = float(np.mean([p["clicked"] for p in posts]))
    mean_dwell = float(np.mean([p["dwell_time"] for p in posts]))
    norm_dwell = min(mean_dwell / DWELL_CAP_SEC, 1.0)
    return (
        W_RETURN * float(session["next_session_return"])
        + W_CLICK * click_rate
        + W_DWELL * norm_dwell
        + W_COMPLETE * float(not session["early_exit"])
    )


def primary_rank_key(session: dict) -> Tuple:
    """
    Lexicographic ranking: next_session_return first, then engagement_score.
    Ensures return dominates when scores are close.
    """
    return (
        int(session["next_session_return"]),
        score_session(session),
    )


@dataclass
class RankedPair:
    user_id: int
    winner_id: str
    loser_id: str
    winner_score: float
    loser_score: float
    winner_return: bool
    loser_return: bool


def build_pairs_for_user(sessions: List[dict]) -> List[RankedPair]:
    """All unordered session pairs for one user; skip ties on primary_rank_key."""
    pairs: List[RankedPair] = []
    n = len(sessions)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = sessions[i], sessions[j]
            key_a, key_b = primary_rank_key(a), primary_rank_key(b)
            if key_a == key_b:
                continue
            if key_a > key_b:
                win, lose = a, b
            else:
                win, lose = b, a
            pairs.append(
                RankedPair(
                    user_id=win["user_id"],
                    winner_id=win["session_id"],
                    loser_id=lose["session_id"],
                    winner_score=score_session(win),
                    loser_score=score_session(lose),
                    winner_return=win["next_session_return"],
                    loser_return=lose["next_session_return"],
                )
            )
    return pairs


def generate_ranked_pairs(
    sessions_path: str | Path,
    output_path: str | Path,
    *,
    verify_scores: bool = True,
) -> Dict:
    """
    Load sessions.json, score, and emit ~38k pairwise preferences (200 users × C(20,2)).

    Returns summary stats dict.
    """
    sessions_path = Path(sessions_path)
    output_path = Path(output_path)

    with open(sessions_path) as f:
        sessions: List[dict] = json.load(f)

    by_user: Dict[int, List[dict]] = {u: [] for u in range(NUM_USERS)}
    for s in sessions:
        by_user[s["user_id"]].append(s)

    all_pairs: List[RankedPair] = []
    score_mismatches = 0
    for uid in range(NUM_USERS):
        user_sess = sorted(by_user[uid], key=lambda x: x["session_idx"])
        if len(user_sess) != SESSIONS_PER_USER:
            raise ValueError(
                f"user {uid}: expected {SESSIONS_PER_USER} sessions, got {len(user_sess)}"
            )
        if verify_scores:
            for s in user_sess:
                stored = s["engagement_score"]
                recomputed = round(score_session(s), 6)
                if abs(stored - recomputed) > 1e-4:
                    score_mismatches += 1
        all_pairs.extend(build_pairs_for_user(user_sess))

    records = [
        {
            "user_id": p.user_id,
            "winner_id": p.winner_id,
            "loser_id": p.loser_id,
            "winner_score": round(p.winner_score, 6),
            "loser_score": round(p.loser_score, 6),
            "winner_return": p.winner_return,
            "loser_return": p.loser_return,
        }
        for p in all_pairs
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f)

    return_rate = np.mean([s["next_session_return"] for s in sessions])
    return {
        "n_sessions": len(sessions),
        "n_pairs": len(records),
        "expected_pairs_max": NUM_USERS * SESSIONS_PER_USER * (SESSIONS_PER_USER - 1) // 2,
        "score_mismatches_vs_json": score_mismatches,
        "fraction_winner_returned": float(np.mean([p.winner_return for p in all_pairs])),
        "fraction_loser_returned": float(np.mean([p.loser_return for p in all_pairs])),
        "session_return_rate": float(return_rate),
        "output": str(output_path),
    }
