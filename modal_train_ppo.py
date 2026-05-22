"""
Run PPO training on Modal with wandb logging.

Run
---
    modal run modal_train_ppo.py

Download outputs when done
--------------------------
    mkdir -p checkpoints
    modal volume get cs224r-trex-results / ./checkpoints/
"""

from __future__ import annotations

import modal

# ---------------------------------------------------------------------------
# Container image — bakes in local code + data at build time
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0", "numpy>=1.24", "wandb>=0.16")
    .add_local_python_source("sim", "models", "ppo")
    .add_local_file("train_ppo.py", "/root/train_ppo.py")
    .add_local_dir("data", remote_path="/root/data")
    .add_local_dir("checkpoints", remote_path="/root/ckpt")
)

# ---------------------------------------------------------------------------
# Volume for output artifacts (checkpoints + history json)
# ---------------------------------------------------------------------------
output_vol = modal.Volume.from_name("cs224r-trex-results", create_if_missing=True)

app = modal.App("cs224r-trex-ppo", image=image)

# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------
@app.function(
    secrets=[modal.Secret.from_name("wandb")],
    volumes={"/output": output_vol},
    timeout=10800,
    cpu=4,
    gpu="T4",  # Comment out this line to use CPU only
)
def train_ppo(
    n_iterations: int = 200,
    n_episodes: int = 64,
    eval_interval: int = 20,
    wandb_run_name: str = "ppo-user-conditioned",
):
    import shutil
    import sys

    sys.path.insert(0, "/root")

    import train_ppo as tp

    tp.N_ITERATIONS = n_iterations
    tp.N_EPISODES_PER_ITER = n_episodes
    tp.EVAL_INTERVAL = eval_interval

    history, base = tp.train(
        data_dir="/root/data",
        checkpoint_dir="/output",
        reward_ckpt="/root/ckpt/reward_net_frozen.pt",
        use_wandb=True,
        wandb_run_name=wandb_run_name,
    )

    output_vol.commit()

    last_eval = next(
        (r for r in reversed(history) if "eval_mean_true_sat" in r), {}
    )
    return {
        "baseline_true_sat": base["mean_true_sat"],
        "final_eval_true_sat": last_eval.get("eval_mean_true_sat"),
        "final_eval_reward": last_eval.get("eval_mean_reward"),
        "n_iterations": len(history),
    }


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    print("Submitting PPO training to Modal (detached)…")
    call = train_ppo.spawn()
    print(f"  function_id: {call.function_call_id}")
    print("\nCheck status:  modal function-call get", call.function_call_id)
    print("Download when done:")
    print("  modal volume get cs224r-trex-results / ./checkpoints/")
