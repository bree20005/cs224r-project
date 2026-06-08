#!/bin/bash
SRC=~/Downloads
DST=~/Downloads/Projs/cs224r-project/figures

cp "$SRC/Screenshot 2026-05-31 at 1.46.22 PM.png" "$DST/rollout_true_sat_engagement.png"
cp "$SRC/Screenshot 2026-05-31 at 1.46.39 PM.png" "$DST/eval_true_sat_engagement.png"
cp "$SRC/Screenshot 2026-05-31 at 1.47.35 PM.png" "$DST/rollout_reward_engagement.png"
cp "$SRC/Screenshot 2026-05-31 at 1.47.55 PM.png" "$DST/train_vf_loss_engagement.png"
cp "$SRC/Screenshot 2026-05-31 at 1.48.07 PM.png" "$DST/train_entropy_engagement.png"

echo "Done:"
ls "$DST"
