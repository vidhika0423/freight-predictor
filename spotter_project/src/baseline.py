"""
baseline.py

Predicts rate_per_mile as the historical average for that exact pickup->delivery
lane (computed from training data only), falling back to the overall global average
rate_per_mile for any lane never seen in training.


Evaluated on the Sep-Oct holdout split,
using MAE and MAPE computed on the final dollar `posted_rate`
"""

import numpy as np
import pandas as pd

SPLIT_CUTOFF_DATE = "2025-09-01"  # train: Jan-Aug, holdout: Sep-Oct


def time_based_split(df: pd.DataFrame, cutoff: str = SPLIT_CUTOFF_DATE):
    dates = pd.to_datetime(df["date"])
    train_split = df[dates < cutoff].copy()
    holdout_split = df[dates >= cutoff].copy()
    return train_split, holdout_split


def fit_lane_baseline(train_split: pd.DataFrame) -> dict:
    """
    Learns the average rate_per_mile per (pickup, delivery) lane from training rows
    only, plus a global fallback average for lanes never seen in training.
    """
    df = train_split.copy()
    df["rate_per_mile"] = df["posted_rate"] / df["distance"]

    lane_avg = df.groupby(["pickup", "delivery"])["rate_per_mile"].mean().to_dict()
    global_avg = df["rate_per_mile"].mean()

    return {"lane_avg": lane_avg, "global_avg": global_avg}


def predict_lane_baseline(df: pd.DataFrame, baseline: dict) -> np.ndarray:
    """
    Predicts posted_rate for each row: looks up the lane's historical average
    rate_per_mile (or falls back to the global average for unseen lanes), then
    multiplies by that row's distance to get a dollar prediction.
    """
    lane_avg = baseline["lane_avg"]
    global_avg = baseline["global_avg"]

    predicted_rpm = df.apply(
        lambda row: lane_avg.get((row["pickup"], row["delivery"]), global_avg),
        axis=1,
    )
    predicted_rate = predicted_rpm.to_numpy() * df["distance"].to_numpy()
    return predicted_rate


def mae(actual, predicted) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def mape(actual, predicted) -> float:
    return float(np.mean(np.abs((actual - predicted) / actual)) * 100)


if __name__ == "__main__":
    raw = pd.read_csv("data/train-test.csv")
    train_split, holdout_split = time_based_split(raw)

    print(f"Train split (Jan-Aug): {len(train_split)} rows")
    print(f"Holdout split (Sep-Oct): {len(holdout_split)} rows")

    baseline = fit_lane_baseline(train_split)
    print(f"Learned {len(baseline['lane_avg'])} lane averages from training data.")
    print(f"Global fallback average rate_per_mile: {baseline['global_avg']:.4f}")

    predicted_rate = predict_lane_baseline(holdout_split, baseline)
    actual_rate = holdout_split["posted_rate"].to_numpy()

    baseline_mae = mae(actual_rate, predicted_rate)
    baseline_mape = mape(actual_rate, predicted_rate)

    print("=== Baseline performance on Sep-Oct holdout ===")
    print(f"MAE:  ${baseline_mae:,.2f}")
    print(f"MAPE: {baseline_mape:.2f}%")

    # how many holdout lanes were actually unseen 
    lane_avg = baseline["lane_avg"]
    unseen_mask = ~holdout_split.apply(
        lambda row: (row["pickup"], row["delivery"]) in lane_avg, axis=1
    )
    print(f"\nHoldout rows using fallback (unseen lane): {unseen_mask.sum()} / {len(holdout_split)} "
          f"({unseen_mask.mean()*100:.1f}%)")