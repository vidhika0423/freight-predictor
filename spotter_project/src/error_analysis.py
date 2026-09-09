"""
error_analysis.py

Error analysis on the Sep-Oct holdout set using the production EBM model.

Breaks down MAE/MAPE by:
  - equipment type
  - seen-vs-unseen lane (tests whether lat/lon + region-cluster actually
    generalizes to lanes never seen in training, as intended)
  - price range (cheap / mid / expensive loads)

Produces a predicted-vs-actual scatter plot.

Prerequisite: run src/train_ebm.py first to generate models/final_ebm_model.pkl
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from clean_features import fit_transform, transform
from baseline import time_based_split, mae, mape


def breakdown_by_group(actual, predicted, groups, label_name):
    df = pd.DataFrame({"actual": actual, "predicted": predicted, "group": groups})
    rows = []
    for group_value, sub in df.groupby("group"):
        rows.append({
            label_name: group_value,
            "n": len(sub),
            "MAE": mae(sub["actual"].to_numpy(), sub["predicted"].to_numpy()),
            "MAPE": mape(sub["actual"].to_numpy(), sub["predicted"].to_numpy()),
        })
    return pd.DataFrame(rows).sort_values(label_name)


if __name__ == "__main__":
    raw = pd.read_csv("data/train-test.csv")
    train_split, holdout_split = time_based_split(raw)

    # Fit feature artifacts on training split only (same discipline as production)
    _, _, _, train_artifacts = fit_transform(train_split)
    holdout_X, holdout_y, holdout_meta = transform(holdout_split, train_artifacts)

    holdout_distance = holdout_split["distance"]
    holdout_actual = holdout_split["posted_rate"].to_numpy()

    # Load the production EBM model (trained on full Jan-Oct data)
    with open("models/final_ebm_model.pkl", "rb") as f:
        model = pickle.load(f)

    predicted_rpm = model.predict(holdout_X)
    predicted_rate = predicted_rpm * holdout_distance.to_numpy()

    overall_mae = mae(holdout_actual, predicted_rate)
    overall_mape = mape(holdout_actual, predicted_rate)
    print(f"=== Overall holdout performance ===")
    print(f"MAE: ${overall_mae:,.2f}   MAPE: {overall_mape:.2f}%\n")

    print("=== By equipment type ")
    equip_breakdown = breakdown_by_group(holdout_actual, predicted_rate,
                                          holdout_split["equipment"].to_numpy(), "equipment")
    print(equip_breakdown.to_string(index=False))
    print()

    # seen vs unseen lane 
    train_lanes = set(zip(train_split["pickup"], train_split["delivery"]))
    is_unseen = [
        (p, d) not in train_lanes
        for p, d in zip(holdout_split["pickup"], holdout_split["delivery"])
    ]
    lane_status = np.where(is_unseen, "unseen_lane", "seen_lane")
    print(" By seen-vs-unseen lane ")
    lane_breakdown = breakdown_by_group(holdout_actual, predicted_rate, lane_status, "lane_status")
    print(lane_breakdown.to_string(index=False))
  

    #price range (terciles based on actual rate) 
    price_bucket = pd.qcut(holdout_actual, q=3, labels=["cheap", "mid", "expensive"])
    print("By price range (terciles) ")
    price_breakdown = breakdown_by_group(holdout_actual, predicted_rate, price_bucket, "price_range")
    print(price_breakdown.to_string(index=False))
   

    #  Scatter plot: predicted vs actual 
    plt.figure(figsize=(7, 7))
    plt.scatter(holdout_actual, predicted_rate, alpha=0.15, s=8, color="#2563eb")
    lims = [0, max(holdout_actual.max(), predicted_rate.max())]
    plt.plot(lims, lims, color="red", linestyle="--", linewidth=1, label="Perfect prediction")
    plt.xlabel("Actual posted_rate ($)")
    plt.ylabel("Predicted posted_rate ($)")
    plt.title("Predicted vs Actual — Sep-Oct Holdout (EBM)")
    plt.legend()
    plt.tight_layout()
    os.makedirs("reports/final", exist_ok=True)
    plt.savefig("reports/final/holdout_predicted_vs_actual.png", dpi=150)
    print("Saved scatter plot to reports/final/holdout_predicted_vs_actual.png")