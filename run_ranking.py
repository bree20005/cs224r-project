#!/usr/bin/env python3
"""Generate ranked_pairs.json from data/sessions.json."""

import json
from pathlib import Path

from pipeline.ranking import generate_ranked_pairs


def main():
    data_dir = Path("data")
    sessions_path = data_dir / "sessions.json"
    if not sessions_path.exists():
        raise SystemExit(
            "Missing data/sessions.json — run: python generate_data.py"
        )

    summary = generate_ranked_pairs(
        sessions_path,
        data_dir / "ranked_pairs.json",
        verify_scores=True,
    )
    print("Ranking pipeline complete:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    with open(data_dir / "ranking_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
