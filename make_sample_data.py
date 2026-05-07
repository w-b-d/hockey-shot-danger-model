"""Generate a small synthetic shots CSV for local development.

The file matches the relevant subset of MoneyPuck columns used by the
pipeline. Goal probability is generated from a hand-crafted geometry rule plus
noise, so the trained model has a real signal to recover.

    python make_sample_data.py --n 20000 --out data/sample_shots.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SHOT_TYPES = ["WRIST", "SLAP", "SNAP", "BACKHAND", "TIP-IN", "DEFLECTED", "WRAP-AROUND"]
SHOT_TYPE_WEIGHTS = [0.55, 0.10, 0.13, 0.07, 0.08, 0.04, 0.03]
TYPE_BOOST = {  # log-odds bumps to the goal model
    "TIP-IN": 0.4, "DEFLECTED": 0.5, "WRAP-AROUND": -0.2,
    "BACKHAND": 0.1, "WRIST": 0.0, "SNAP": 0.05, "SLAP": -0.1,
}
TEAMS = ["BOS", "TOR", "MTL", "NYR", "PIT", "TBL", "COL", "VGK", "EDM", "CAR"]


def generate(n: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Most shots come from the slot, fewer from the perimeter
    distance = rng.gamma(shape=2.5, scale=10.0, size=n).clip(2, 80)
    # Angle slightly skews wide; mix beta with uniform
    angle = rng.beta(2, 2, size=n) * 90

    # Convert (distance, angle) back to (x, y) on the offensive half
    x = 89 - distance * np.cos(np.radians(angle))
    y = distance * np.sin(np.radians(angle)) * rng.choice([-1, 1], size=n)

    shot_type = rng.choice(SHOT_TYPES, size=n, p=SHOT_TYPE_WEIGHTS)
    is_rebound = rng.binomial(1, 0.05, size=n)
    is_rush = rng.binomial(1, 0.12, size=n)
    home = rng.binomial(1, 0.5, size=n)
    home_skaters = rng.choice([3, 4, 5, 6], size=n, p=[0.02, 0.10, 0.83, 0.05])
    away_skaters = rng.choice([3, 4, 5, 6], size=n, p=[0.02, 0.10, 0.83, 0.05])

    # Goal probability model (log-odds)
    log_odds = (
        -1.4
        - 0.06 * distance
        - 0.012 * angle
        + 1.1 * is_rebound
        + 0.45 * is_rush
        + np.array([TYPE_BOOST[t] for t in shot_type])
    )
    p = 1 / (1 + np.exp(-log_odds))
    goal = rng.binomial(1, p)

    return pd.DataFrame({
        "season": rng.choice([2022, 2023], size=n),
        "game_id": rng.integers(20001, 21300, size=n),
        "period": rng.choice([1, 2, 3], size=n),
        "team": np.where(home == 1, "HOME", "AWAY"),
        "teamCode": rng.choice(TEAMS, size=n),
        "shooterName": [f"Player{i % 250}" for i in range(n)],
        "shotType": shot_type,
        "xCordAdjusted": x.round(1),
        "yCordAdjusted": y.round(1),
        "shotDistance": distance.round(1),
        "shotAngleAdjusted": angle.round(1),
        "shotRebound": is_rebound,
        "shotRush": is_rush,
        "shotOnEmptyNet": rng.binomial(1, 0.01, size=n),
        "homeSkatersOnIce": home_skaters,
        "awaySkatersOnIce": away_skaters,
        "goal": goal,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20000, help="Number of shots")
    parser.add_argument("--out", type=str, default="data/sample_shots.csv")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    df = generate(args.n, seed=args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} shots to {out_path}")
    print(f"Goal rate: {df['goal'].mean()*100:.2f}%")


if __name__ == "__main__":
    main()
