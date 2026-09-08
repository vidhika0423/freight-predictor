"""
error_analysis.py

Error analysis on the Sep-Oct holdout set, using the final tuned XGBoost

Breaks down MAE/MAPE by:
  - equipment type
  - seen-vs-unseen lane (tests whether lat/lon + region-cluster actually
    generalizes to lanes never seen in training, as intended)
  - price range (cheap / mid / expensive loads)

produces a predicted-vs-actual scatter plot  ).

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

from clean_features import fit_transform, transform
from baseline import time_based_split, mae, mape

# Final tuned config locked in 
FINAL_XGB_PARAMS = dict(
    n_estimators=3000,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    eval_metric="mae",
    early_stopping_rounds=50,
)


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

    train_X, train_y, train_meta, artifacts = fit_transform(train_split)
    holdout_X, holdout_y, holdout_meta = transform(holdout_split, artifacts)

    holdout_distance = holdout_split["distance"]
    holdout_actual = holdout_split["posted_rate"].to_numpy()

    model = XGBRegressor(**FINAL_XGB_PARAMS)
    model.fit(train_X, train_y, eval_set=[(holdout_X, holdout_y)], verbose=False)

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
    plt.title("Predicted vs Actual — Sep-Oct Holdout (Tuned XGBoost)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/holdout_predicted_vs_actual.png", dpi=150)
    print("Saved scatter plot to reports/holdout_predicted_vs_actual.png")