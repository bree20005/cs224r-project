import numpy as np
from sim.config import (
    NUM_POSTS, NUM_CLUSTERS, POSTS_PER_CLUSTER,
    HELD_OUT_CLUSTER, EMBED_DIM, CLUSTER_COHESION, RANDOM_SEED,
)


class ContentLibrary:
    """
    500 posts organized into 5 topic clusters (analogous to subreddits).

    Cluster structure
    -----------------
    Each cluster k owns posts in index range [k*100, (k+1)*100).
    Post embeddings are unit vectors biased toward the k-th standard basis
    direction, with CLUSTER_COHESION controlling how tightly posts cluster.

    Cluster HELD_OUT_CLUSTER (4) is never shown during training; it is
    reserved for held-out evaluation of reward generalization.
    """

    def __init__(self, seed: int = RANDOM_SEED):
        rng = np.random.default_rng(seed)

        # Cluster centers: orthogonal unit vectors in the first NUM_CLUSTERS dims
        self.cluster_centers = np.eye(NUM_CLUSTERS, EMBED_DIM)   # (5, 16)

        embeddings = np.zeros((NUM_POSTS, EMBED_DIM))
        cluster_ids = np.zeros(NUM_POSTS, dtype=int)

        for k in range(NUM_CLUSTERS):
            start = k * POSTS_PER_CLUSTER
            end = start + POSTS_PER_CLUSTER
            cluster_ids[start:end] = k

            # Random unit vectors for each post
            noise = rng.standard_normal((POSTS_PER_CLUSTER, EMBED_DIM))
            noise /= np.linalg.norm(noise, axis=1, keepdims=True)

            # Mix cluster center with noise, then re-normalize
            raw = CLUSTER_COHESION * self.cluster_centers[k] + (1 - CLUSTER_COHESION) * noise
            raw /= np.linalg.norm(raw, axis=1, keepdims=True)
            embeddings[start:end] = raw

        self.embeddings = embeddings       # (500, 16)
        self.cluster_ids = cluster_ids     # (500,)

        self.train_post_ids = np.where(cluster_ids != HELD_OUT_CLUSTER)[0]  # 400 posts
        self.eval_post_ids = np.where(cluster_ids == HELD_OUT_CLUSTER)[0]   # 100 posts
