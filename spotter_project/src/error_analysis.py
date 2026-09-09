"""
error_analysis.py

Error analysis on the Sep-Oct holdout set.

Breaks down MAE/MAPE by:
  - equipment type
  - seen-vs-unseen lane (tests whether lat/lon + region-cluster actually
    generalizes to lanes never seen in training, as intended)
  - price range (cheap / mid / expensive loads)

Produces a predicted-vs-actual scatter plot.

Prerequisite: run src/train_ebm.py first to generate models/final_ebm_model.pkl
(used here only for the labeled in-sample sanity check, not the headline metric)
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from interpret.glassbox import ExplainableBoostingRegressor

from clean_features import fit_transform, transform
from baseline import time_based_split, mae, mape
from train_ebm import EBM_PARAMS


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

    # Honest holdout model: same tuned params as production, fit on Jan-Aug ONLY
    train_X, train_y, train_meta, _ = fit_transform(train_split)
    print("Training holdout-only EBM (same EBM_PARAMS as production, fit on Jan-Aug only)...")
    model = ExplainableBoostingRegressor(**EBM_PARAMS)
    model.fit(train_X, train_y)

    predicted_rpm = model.predict(holdout_X)
    predicted_rate = predicted_rpm * holdout_distance.to_numpy()

    overall_mae = mae(holdout_actual, predicted_rate)
    overall_mape = mape(holdout_actual, predicted_rate)
    print(f"\n Overall holdout performance (HONEST - fit on Jan-Aug only)")
    print(f"MAE: ${overall_mae:,.2f}   MAPE: {overall_mape:.2f}%")

    # Labeled sanity check only: the full-data production model scored on the same rows it was partly trained on. Expected to look better. Not a metric
    try:
        with open("models/final_ebm_model.pkl", "rb") as f:
            production_model = pickle.load(f)
        production_pred_rate = production_model.predict(holdout_X) * holdout_distance.to_numpy()
        production_mae = mae(holdout_actual, production_pred_rate)
        print(f"[in-sample sanity check, NOT a holdout metric] production model on the same "
              f"rows: ${production_mae:,.2f}  this looks better only because those rows were "
              f"inside its training data.\n")
    except FileNotFoundError:
        print("(Skipping in-sample sanity check: models/final_ebm_model.pkl not found — "
              "run src/train_ebm.py first if you want to see it.)\n")

    printBy equipment type ")
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