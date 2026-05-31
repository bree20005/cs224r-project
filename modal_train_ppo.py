"""
Run PPO training on Modal with W&B logging.

Trains engagement + ground_truth policies in parallel on separate T4 GPUs.
A coordinator function runs entirely on Modal, so closing your terminal
is safe — pass --detach to fully disconnect.

Run (detached — close terminal anytime)
----------------------------------------
    modal run --detach modal_train_ppo.py

Run a single policy
-------------------
    modal run --detach modal_train_ppo.py --reward-type engagement
    modal run --detach modal_train_ppo.py --reward-type ground_truth
    modal run --detach modal_train_ppo.py --reward-type trex

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
    .add_local_dir("data", remote_path="/root/data")
    .add_local_dir("checkpoints", remote_path="/root/ckpt")
)

output_vol = modal.Volume.from_name("cs224r-trex-results", create_if_missing=True)

app = modal.App("cs224r-trex-ppo", image=image)

# ---------------------------------------------------------------------------
# Per-policy training function (one T4 per policy)
# ---------------------------------------------------------------------------

@app.function(
    secrets=[modal.Secret.from_name("wandb")],
    volumes={"/output": output_vol},
    timeout=10800,
    cpu=4,
    gpu="T4",
)
def train_policy(reward_type: str, n_iterations: int = 200):
    import sys
    sys.path.insert(0, "/root")
    import train_ppo as tp

    tp.N_ITERATIONS = n_iterations
    tp.N_EPISODES_PER_ITER = 64
    tp.EVAL_INTERVAL = 20

    history, base = tp.train(
        data_dir="/root/data",
        checkpoint_dir="/output",
        reward_ckpt="/root/ckpt/reward_net_frozen.pt",
        use_wandb=True,
        wandb_project="cs224r-trex",
        wandb_run_name=f"ppo-{reward_type}",
        reward_type=reward_type,
    )

    output_vol.commit()

    last_eval = next(
        (r for r in reversed(history) if "eval_mean_true_sat" in r), {}
    )
    return {
        "reward_type": reward_type,
        "baseline_true_sat": round(base["mean_true_sat"], 4),
        "final_eval_true_sat": round(last_eval.get("eval_mean_true_sat", 0.0), 4),
        "true_sat_delta": round(
            last_eval.get("eval_mean_true_sat", 0.0) - base["mean_true_sat"], 4
        ),
        "n_iterations": len(history),
    }


# ---------------------------------------------------------------------------
# Coordinator — runs ON Modal so it survives terminal disconnect (--detach)
# Spawns one train_policy container per reward type in parallel.
# ---------------------------------------------------------------------------

@app.function(
    volumes={"/output": output_vol},
    timeout=14400,  # 4 h ceiling — well above any single training run
    cpu=1,
)
def run_all(reward_types: list, n_iterations: int = 200):
    print(f"Coordinator starting — training: {reward_types}")
    futures = [train_policy.spawn(rt, n_iterations) for rt in reward_types]
    results = [f.get() for f in futures]

    print("\n" + "=" * 60)
    print("ALL TRAINING COMPLETE")
    print("=" * 60)
    print(f"{'Policy':15s}  {'True Sat':>10s}  {'Baseline':>10s}  {'Δ':>8s}")
    for r in results:
        print(
            f"{r['reward_type']:15s}  "
            f"{r['final_eval_true_sat']:10.4f}  "
            f"{r['baseline_true_sat']:10.4f}  "
            f"{r['true_sat_delta']:+8.4f}"
        )
    return results


# ---------------------------------------------------------------------------
# Local entrypoint — just submits the coordinator and exits
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(reward_type: str = "", n_iterations: int = 200):
    """
    reward_type : engagement | trex | ground_truth, or empty for both missing ones
    n_iterations: training iterations (default 200)
    """
    missing = ["engagement", "ground_truth"]  # T-REX already trained
    to_train = [reward_type] if reward_type else missing

    print(f"\nSubmitting coordinator to Modal (trains: {to_train})")
    print("Pass --detach to disconnect your terminal safely.\n")

    # remote() call: blocks locally until done, but Modal keeps running if terminal closes
    results = run_all.remote(to_train, n_iterations)

    print("\nDownload checkpoints:")
    print("  modal volume get cs224r-trex-results / ./checkpoints/")
    print("\nView W&B graphs:")
    print("  https://wandb.ai/home  →  project cs224r-trex")
