"""
Run PPO training on Modal with W&B logging.

Each (reward_type, seed) pair runs on its own T4 GPU in parallel.
For trex jobs, the reward network is trained first on the same container.

Run all 3 seeds for all policies (9 jobs in parallel)
------------------------------------------------------
    modal run --detach modal_train_ppo.py --seeds 0,1,2

Run a single policy × seed
---------------------------
    modal run --detach modal_train_ppo.py --reward-type trex --seeds 42

Run all seeds for one policy type
----------------------------------
    modal run --detach modal_train_ppo.py --reward-type engagement --seeds 0,1,2

Monitor after detaching
-----------------------
    modal app list
    modal app logs <app-id>

Download checkpoints + history when done
-----------------------------------------
    modal volume get cs224r-trex-results / ./checkpoints/
"""

from __future__ import annotations

import modal

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0", "numpy>=1.24", "wandb>=0.16")
    .add_local_python_source("sim", "models", "ppo", "pipeline")
    .add_local_file("train_ppo.py", "/root/train_ppo.py")
    .add_local_file("train_reward.py", "/root/train_reward.py")
    .add_local_dir("data", remote_path="/root/data")
)

output_vol = modal.Volume.from_name("cs224r-trex-results", create_if_missing=True)

app = modal.App("cs224r-trex-ppo", image=image)

# ---------------------------------------------------------------------------
# Per-(policy, seed) training function — one T4 per job
# ---------------------------------------------------------------------------

@app.function(
    secrets=[modal.Secret.from_name("wandb")],
    volumes={"/output": output_vol},
    timeout=10800,
    cpu=4,
    gpu="T4",
)
def train_policy(reward_type: str, seed: int, n_iterations: int = 200):
    import sys
    sys.path.insert(0, "/root")

    reward_ckpt = None  # derived from seed inside tp.train() for non-trex types

    # Train reward network for this seed before PPO (trex only)
    if reward_type == "trex":
        import train_reward as tr
        tr.train(data_dir="/root/data", checkpoint_dir="/output", seed=seed)
        reward_ckpt = f"/output/reward_net_frozen_seed{seed}.pt"

    import train_ppo as tp
    tp.N_ITERATIONS = n_iterations
    tp.N_EPISODES_PER_ITER = 64
    tp.EVAL_INTERVAL = 20

    history, base = tp.train(
        data_dir="/root/data",
        checkpoint_dir="/output",
        reward_ckpt=reward_ckpt,
        seed=seed,
        use_wandb=True,
        wandb_project="cs224r-trex",
        wandb_run_name=f"ppo-{reward_type}-seed{seed}",
        reward_type=reward_type,
    )

    output_vol.commit()

    last_eval = next(
        (r for r in reversed(history) if "eval_mean_true_sat" in r), {}
    )
    return {
        "reward_type": reward_type,
        "seed": seed,
        "baseline_true_sat": round(base["mean_true_sat"], 4),
        "final_eval_true_sat": round(last_eval.get("eval_mean_true_sat", 0.0), 4),
        "true_sat_delta": round(
            last_eval.get("eval_mean_true_sat", 0.0) - base["mean_true_sat"], 4
        ),
        "n_iterations": len(history),
    }


# ---------------------------------------------------------------------------
# Coordinator — runs ON Modal so it survives terminal disconnect (--detach)
# Spawns one container per (reward_type, seed) pair, all in parallel.
# ---------------------------------------------------------------------------

@app.function(
    volumes={"/output": output_vol},
    timeout=14400,
    cpu=1,
)
def run_all(reward_types: list, seeds: list, n_iterations: int = 200):
    jobs = [(rt, s) for rt in reward_types for s in seeds]
    print(f"Coordinator starting — {len(jobs)} jobs: {jobs}")
    futures = [train_policy.spawn(rt, s, n_iterations) for rt, s in jobs]
    results = [f.get() for f in futures]

    print("\n" + "=" * 60)
    print("ALL TRAINING COMPLETE")
    print("=" * 60)
    print(f"{'Policy':15s}  {'Seed':>6s}  {'True Sat':>10s}  {'Baseline':>10s}  {'Δ':>8s}")
    for r in sorted(results, key=lambda x: (x["reward_type"], x["seed"])):
        print(
            f"{r['reward_type']:15s}  "
            f"{r['seed']:6d}  "
            f"{r['final_eval_true_sat']:10.4f}  "
            f"{r['baseline_true_sat']:10.4f}  "
            f"{r['true_sat_delta']:+8.4f}"
        )
    return results


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(reward_type: str = "", seeds: str = "0,1,2", n_iterations: int = 200):
    """
    reward_type : engagement | trex | ground_truth, or empty for all three
    seeds       : comma-separated seed values (default: 0,1,2)
    n_iterations: training iterations per job (default: 200)
    """
    all_types = ["trex", "engagement", "ground_truth"]
    to_train = [reward_type] if reward_type else all_types
    seed_list = [int(s) for s in seeds.split(",")]

    n_jobs = len(to_train) * len(seed_list)
    print(f"\nSubmitting coordinator to Modal ({n_jobs} jobs in parallel)")
    print(f"  policies: {to_train}")
    print(f"  seeds:    {seed_list}")
    print("Pass --detach to disconnect your terminal safely.\n")

    results = run_all.remote(to_train, seed_list, n_iterations)

    print("\nDownload checkpoints:")
    print("  modal volume get cs224r-trex-results / ./checkpoints/")
    print("\nView W&B graphs:")
    print("  https://wandb.ai/home  →  project cs224r-trex")
