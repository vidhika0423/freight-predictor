"""
evaluate_ensemble.py

Compares tuned XGBoost, EBM, and a blended Ensemble on:
  1. The Sep-Oct holdout split (MAE, MAPE, Train-Holdout Gap)
  2. December extrapolation behaviour (smoothness of seasonal curve)

Generates:
  - outputs/december_ensemble.csv
  - reports/ensemble_holdout_comparison.png
  - reports/december_ensemble_chart.png
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from xgboost import XGBRegressor

import sys
sys.path.append(os.path.abspath("src"))

from clean_features import fit_transform, transform
from baseline import time_based_split, mae, mape


def dollar_metrics(model, X, distance, actual_dollar_rate):
    predicted_rpm = model.predict(X)
    predicted_rate = predicted_rpm * distance.to_numpy()
    return mae(actual_dollar_rate, predicted_rate), mape(actual_dollar_rate, predicted_rate), predicted_rate


if __name__ == "__main__":
    raw = pd.read_csv("data/train-test.csv")
    train_split, holdout_split = time_based_split(raw)

    train_X, train_y, train_meta, artifacts = fit_transform(train_split)
    holdout_X, holdout_y, holdout_meta = transform(holdout_split, artifacts)

    train_dist = train_split["distance"]
    train_act = train_split["posted_rate"].to_numpy()
    hold_dist = holdout_split["distance"]
    hold_act = holdout_split["posted_rate"].to_numpy()

    # 1. Load or fit tuned XGBoost
    xgb_model = XGBRegressor(
        n_estimators=141,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(train_X, train_y)
    xgb_train_mae, _, _ = dollar_metrics(xgb_model, train_X, train_dist, train_act)
    xgb_hold_mae, xgb_hold_mape, xgb_hold_pred = dollar_metrics(xgb_model, holdout_X, hold_dist, hold_act)

    # 2. Fit EBM on training split
    from interpret.glassbox import ExplainableBoostingRegressor
    ebm_model = ExplainableBoostingRegressor(
        max_bins=256,
        interactions=10,
        learning_rate=0.01,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    ebm_model.fit(train_X, train_y)
    ebm_train_mae, _, _ = dollar_metrics(ebm_model, train_X, train_dist, train_act)
    ebm_hold_mae, ebm_hold_mape, ebm_hold_pred = dollar_metrics(ebm_model, holdout_X, hold_dist, hold_act)

    # 3. Ensemble: 50/50 blend
    ens_train_pred = (xgb_model.predict(train_X) * 0.5 + ebm_model.predict(train_X) * 0.5) * train_dist.to_numpy()
    ens_train_mae = mae(train_act, ens_train_pred)

    ens_hold_pred = (xgb_hold_pred + ebm_hold_pred) / 2.0
    ens_hold_mae = mae(hold_act, ens_hold_pred)
    ens_hold_mape = mape(hold_act, ens_hold_pred)

    print("=== MODEL COMPARISON ON SEP-OCT HOLDOUT ===")
    print(f"{'Model':25s} {'Train MAE':>12s} {'Holdout MAE':>13s} {'Holdout MAPE':>13s} {'Gap':>10s}")
    print(f"{'XGBoost (tuned)':25s} ${xgb_train_mae:10.2f} ${xgb_hold_mae:11.2f} {xgb_hold_mape:12.2f}% ${xgb_hold_mae - xgb_train_mae:8.2f}")
    print(f"{'EBM (production config)':25s} ${ebm_train_mae:10.2f} ${ebm_hold_mae:11.2f} {ebm_hold_mape:12.2f}% ${ebm_hold_mae - ebm_train_mae:8.2f}")
    print(f"{'Ensemble (50/50 Blend)':25s} ${ens_train_mae:10.2f} ${ens_hold_mae:11.2f} {ens_hold_mape:12.2f}% ${ens_hold_mae - ens_train_mae:8.2f}")

    # 4. December extrapolation
    december_raw = pd.read_csv("data/december-chart-inputs.csv")
    dec_features, _, _ = transform(december_raw, artifacts)

    dec_xgb_pred = xgb_model.predict(dec_features) * december_raw["distance"].to_numpy()
    dec_ebm_pred = ebm_model.predict(dec_features) * december_raw["distance"].to_numpy()
    dec_ens_pred = (dec_xgb_pred + dec_ebm_pred) / 2.0

    dec_df = december_raw.copy()
    dec_df["predicted_rate"] = dec_ens_pred
    dec_df["xgb_predicted_rate"] = dec_xgb_pred
    dec_df["ebm_predicted_rate"] = dec_ebm_pred
    os.makedirs("outputs", exist_ok=True)
    dec_df.to_csv("outputs/december_ensemble.csv", index=False)
    print("\nSaved outputs/december_ensemble.csv")

    # Plot comparisons
    os.makedirs("reports", exist_ok=True)
    dates = pd.to_datetime(december_raw["date"])
    plt.figure(figsize=(11, 5))
    plt.plot(dates, dec_xgb_pred, marker="s", label="XGBoost (flat staircase)", color="#f59e0b", linestyle="--")
    plt.plot(dates, dec_ebm_pred, marker="o", label="EBM (smooth shape)", color="#6366f1", linewidth=2)
    plt.plot(dates, dec_ens_pred, marker="^", label="Ensemble 50/50", color="#10b981", linewidth=1.8)
    plt.xlabel("Date")
    plt.ylabel("Predicted rate ($)")
    plt.title("December Extrapolation: XGBoost vs EBM vs Ensemble")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("reports/december_ensemble_chart.png", dpi=150)
    print("Saved reports/december_ensemble_chart.png")
