import numpy as np
from sim.config import (
    NUM_USERS, NUM_CLUSTERS, HELD_OUT_CLUSTER, EMBED_DIM, RANDOM_SEED,
)
from sim.content import ContentLibrary


class UserPopulation:
    """
    200 users, each with a latent preference vector in the same 16-d embedding
    space as posts.

    Each user prefers 1 or 2 topic clusters drawn from the training clusters
    (0–3 only — no user is initialized to prefer the held-out cluster 4).
    The preference vector is a normalized weighted sum of the preferred cluster
    centers, plus a small amount of individual noise.

    self.preferences: (200, 16) float array  — GROUND TRUTH, never exposed
                                               during T-REX training
    self.primary_clusters: (200,) int array  — each user's dominant cluster
    """

    def __init__(self, content: ContentLibrary, seed: int = RANDOM_SEED + 1):
        rng = np.random.default_rng(seed)

        train_clusters = [c for c in range(NUM_CLUSTERS) if c != HELD_OUT_CLUSTER]
        cluster_centers = content.cluster_centers  # (5, 16)

        preferences = np.zeros((NUM_USERS, EMBED_DIM))
        primary_clusters = np.zeros(NUM_USERS, dtype=int)

        for uid in range(NUM_USERS):
            # 60 % of users have one preferred cluster, 40 % have two
            n_preferred = rng.choice([1, 2], p=[0.6, 0.4])
            preferred = rng.choice(train_clusters, size=n_preferred, replace=False)
            primary_clusters[uid] = preferred[0]

            weights = rng.dirichlet(np.ones(n_preferred))
            pref = sum(float(w) * cluster_centers[c] for w, c in zip(weights, preferred))

            # Small individual noise so no two users are identical
            pref = pref + rng.standard_normal(EMBED_DIM) * 0.05
            preferences[uid] = pref / np.linalg.norm(pref)

        self.preferences = preferences          # (200, 16)
        self.primary_clusters = primary_clusters

    def true_affinity(self, user_id: int, post_embedding: np.ndarray) -> float:
        """Dot product u·p — the ground-truth satisfaction signal."""
        return float(np.dot(self.preferences[user_id], post_embedding))
