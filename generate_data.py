"""
generate_data.py — run once to produce all raw session data.

Outputs (all in data/)
----------------------
post_embeddings.npy     (500, 16)  float64
post_clusters.npy       (500,)     int64
user_preferences.npy    (200, 16)  float64  — ground truth, never expose in training
train_post_ids.npy      (400,)     int64
eval_post_ids.npy       (100,)     int64

sessions.json                    4 000 sessions at noise=0.3 (default)
sessions_noise{τ}.json           same corpus re-simulated at each ablation noise level

sessions.json record schema
---------------------------
{
  "session_id":          "u000_s00",
  "user_id":             0,
  "session_idx":         0,
  "early_exit":          false,
  "next_session_return": true,
  "true_satisfaction":   0.3456,   // GROUND TRUTH — never use as training signal
  "engagement_score":    0.6234,   // ranking signal for T-REX pairing
  "posts": [
    {
      "post_id":       42,
      "cluster":       1,
      "clicked":       true,
      "dwell_time":    12.5,
      "scroll_depth":  0.78,
      "true_affinity": 0.65       // GROUND TRUTH — never use as training signal
    }, ...
  ]
}

Note: embeddings are stored separately in post_embeddings.npy; index by post_id.
"""

import json
import os
import numpy as np

from sim.config import (
    NUM_USERS, SESSIONS_PER_USER, POSTS_PER_SESSION,
    DEFAULT_NOISE, NOISE_LEVELS, RANDOM_SEED,
)
from sim.content import ContentLibrary
from sim.users import UserPopulation
from sim.env import SocialFeedEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def session_to_dict(s) -> dict:
    return {
        "session_id": s.session_id,
        "user_id": s.user_id,
        "session_idx": s.session_idx,
        "early_exit": s.early_exit,
        "next_session_return": s.next_session_return,
        "true_satisfaction": round(s.true_satisfaction, 6),
        "engagement_score": round(s.engagement_score, 6),
        "posts": [
            {
                "post_id": p.post_id,
                "cluster": p.cluster,
                "clicked": p.clicked,
                "dwell_time": round(p.dwell_time, 3),
                "scroll_depth": round(p.scroll_depth, 4),
                "true_affinity": round(p.true_affinity, 6),
            }
            for p in s.posts
        ],
    }


def print_stats(sessions, label: str = ""):
    true_sat = np.array([s.true_satisfaction for s in sessions])
    eng = np.array([s.engagement_score for s in sessions])
    r = float(np.corrcoef(eng, true_sat)[0, 1])
    return_rate = np.mean([s.next_session_return for s in sessions])
    exit_rate = np.mean([s.early_exit for s in sessions])
    click_rate = np.mean([p.clicked for s in sessions for p in s.posts])
    print(f"  {label}")
    print(f"    sessions          : {len(sessions)}")
    print(f"    mean true_sat     : {true_sat.mean():.4f}  (std {true_sat.std():.4f})")
    print(f"    mean eng_score    : {eng.mean():.4f}  (std {eng.std():.4f})")
    print(f"    Pearson r(eng,sat): {r:.4f}   ← how well engagement tracks true preference")
    print(f"    next-session ret  : {return_rate:.2%}")
    print(f"    early-exit rate   : {exit_rate:.2%}")
    print(f"    click rate        : {click_rate:.2%}")


def save_json(sessions, path: str):
    with open(path, "w") as f:
        json.dump([session_to_dict(s) for s in sessions], f)
    print(f"  saved → {path}  ({os.path.getsize(path) // 1024} KB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs("data", exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Build world model (content + users)
    # ------------------------------------------------------------------
    print("Building content library (500 posts, 5 clusters)…")
    content = ContentLibrary(seed=RANDOM_SEED)

    print("Building user population (200 users)…")
    users = UserPopulation(content, seed=RANDOM_SEED + 1)

    # Persist static world state
    np.save("data/post_embeddings.npy", content.embeddings)
    np.save("data/post_clusters.npy", content.cluster_ids)
    np.save("data/user_preferences.npy", users.preferences)
    np.save("data/train_post_ids.npy", content.train_post_ids)
    np.save("data/eval_post_ids.npy", content.eval_post_ids)
    print(f"  train posts : {len(content.train_post_ids)}  (clusters 0–3)")
    print(f"  eval posts  : {len(content.eval_post_ids)}   (cluster 4, held-out)")

    # Quick sanity check on affinity signal strength
    pref_affinities, cross_affinities = [], []
    for uid in range(NUM_USERS):
        pc = users.primary_clusters[uid]
        for pid in content.train_post_ids[:20]:
            a = users.true_affinity(uid, content.embeddings[pid])
            if content.cluster_ids[pid] == pc:
                pref_affinities.append(a)
            else:
                cross_affinities.append(a)
    print(f"  affinity check → preferred cluster: {np.mean(pref_affinities):.3f}, "
          f"other clusters: {np.mean(cross_affinities):.3f}")

    # ------------------------------------------------------------------
    # 2. Default simulation (noise = 0.3)
    # ------------------------------------------------------------------
    print(f"\nSimulating {NUM_USERS} users × {SESSIONS_PER_USER} sessions "
          f"(noise={DEFAULT_NOISE})…")
    env = SocialFeedEnv(content, users, noise_level=DEFAULT_NOISE, seed=RANDOM_SEED + 2)
    sessions = env.simulate_all()
    print_stats(sessions, label=f"noise={DEFAULT_NOISE} (default)")
    save_json(sessions, "data/sessions.json")

    # ------------------------------------------------------------------
    # 3. Noise ablation: re-simulate at each noise level
    # ------------------------------------------------------------------
    print("\nNoise ablation datasets…")
    ablation_stats = {}
    for noise in NOISE_LEVELS:
        env_n = SocialFeedEnv(content, users, noise_level=noise,
                              seed=RANDOM_SEED + 200 + int(noise * 100))
        sess_n = env_n.simulate_all()
        true_sat = np.array([s.true_satisfaction for s in sess_n])
        eng = np.array([s.engagement_score for s in sess_n])
        r = float(np.corrcoef(eng, true_sat)[0, 1])
        ablation_stats[noise] = {"pearson_r": r}
        print(f"  noise={noise:.1f}  Pearson r(engagement, true_sat) = {r:.4f}")
        save_json(sess_n, f"data/sessions_noise{noise:.1f}.json")

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    summary = {
        "n_users": NUM_USERS,
        "n_posts": 500,
        "n_clusters": 5,
        "held_out_cluster": 4,
        "embed_dim": 16,
        "sessions_per_user": SESSIONS_PER_USER,
        "posts_per_session": POSTS_PER_SESSION,
        "default_noise": DEFAULT_NOISE,
        "total_sessions": len(sessions),
        "approx_ranked_pairs": len(sessions) * (len(sessions) // NUM_USERS) // 2,
        "noise_ablation_pearson_r": ablation_stats,
    }
    with open("data/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone. All files written to data/")
    print(f"  Approximate ranked pairs available (per user): "
          f"{SESSIONS_PER_USER * (SESSIONS_PER_USER - 1) // 2} "
          f"× {NUM_USERS} users = "
          f"{NUM_USERS * SESSIONS_PER_USER * (SESSIONS_PER_USER - 1) // 2:,}")


if __name__ == "__main__":
    main()
