"""

Systematic hyperparameter exploration for Explainable Boosting Machine (EBM).
Matches the rigor of train_tune.py (XGBoost tuning) by evaluating:
  - max_bins: granularity of shape functions (128, 256, 512)
  - interactions: number of pairwise interaction terms (0, 5, 10, 15)
  - learning_rate: boosting step size (0.005, 0.01, 0.02)

For each configuration, tracks:
  - Train MAE ($)
  - Holdout MAE ($) on Sep-Oct split
  - Holdout MAPE (%)
  - Train-Holdout Gap ($) to guard against overfitting
"""

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor

import sys
import os
sys.path.append(os.path.abspath("src"))

from clean_features import fit_transform, transform
from baseline import time_based_split, mae, mape


def dollar_metrics(model, X, distance, actual_dollar_rate):
    predicted_rpm = model.predict(X)
    predicted_rate = predicted_rpm * distance.to_numpy()
    return mae(actual_dollar_rate, predicted_rate), mape(actual_dollar_rate, predicted_rate)


if __name__ == "__main__":
    raw = pd.read_csv("data/train-test.csv")
    train_split, holdout_split = time_based_split(raw)

    train_X, train_y, train_meta, artifacts = fit_transform(train_split)
    holdout_X, holdout_y, holdout_meta = transform(holdout_split, artifacts)

    train_dist = train_split["distance"]
    train_act = train_split["posted_rate"].to_numpy()
    hold_dist = holdout_split["distance"]
    hold_act = holdout_split["posted_rate"].to_numpy()

    configs = [
        {"max_bins": 128, "interactions": 10, "learning_rate": 0.01, "name": "bins_128"},
        {"max_bins": 256, "interactions": 10, "learning_rate": 0.01, "name": "baseline_params"},
        {"max_bins": 512, "interactions": 10, "learning_rate": 0.01, "name": "bins_512"},
        {"max_bins": 256, "interactions": 0,  "learning_rate": 0.01, "name": "no_interactions_pure_additive"},
        {"max_bins": 256, "interactions": 5,  "learning_rate": 0.01, "name": "inter_5"},
        {"max_bins": 256, "interactions": 15, "learning_rate": 0.01, "name": "inter_15"},
        {"max_bins": 256, "interactions": 10, "learning_rate": 0.005,"name": "lr_0.005"},
        {"max_bins": 256, "interactions": 10, "learning_rate": 0.02, "name": "lr_0.02"},
    ]

    results = []
    print(f"{'Config':30s} {'Train MAE':>10s} {'Hold MAE':>10s} {'Hold MAPE':>10s} {'Gap':>8s}")
    print("-" * 75)

    for cfg in configs:
        model = ExplainableBoostingRegressor(
            max_bins=cfg["max_bins"],
            interactions=cfg["interactions"],
            learning_rate=cfg["learning_rate"],
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        model.fit(train_X, train_y)

        tr_mae, tr_mape = dollar_metrics(model, train_X, train_dist, train_act)
        ho_mae, ho_mape = dollar_metrics(model, holdout_X, hold_dist, hold_act)
        gap = ho_mae - tr_mae

        res = {
            "name": cfg["name"],
            "max_bins": cfg["max_bins"],
            "interactions": cfg["interactions"],
            "learning_rate": cfg["learning_rate"],
            "train_mae": tr_mae,
            "holdout_mae": ho_mae,
            "holdout_mape": ho_mape,
            "gap": gap
        }
        results.append(res)
        print(f"{cfg['name']:30s} ${tr_mae:9.2f} ${ho_mae:9.2f} {ho_mape:9.2f}% ${gap:7.2f}")

    df_res = pd.DataFrame(results).sort_values("holdout_mae")
    print("\n--- Summary Sorted by Holdout MAE ---")
    print(df_res.to_string(index=False))
