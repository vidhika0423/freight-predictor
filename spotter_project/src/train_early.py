"""
train_early.py

Step 3: Early model testing (quick, rough pass).

Purpose: confirm the end-to-end pipeline (clean_features.py -> model -> metrics)
actually works, and get a first read on whether XGBoost/Random Forest/Linear
Regression can beat the baseline. This is NOT the tuning step - default-ish
hyperparameters are used here on purpose.

Trained on Jan-Aug, evaluated on the Sep-Oct holdout - same split as baseline.py,
so results are directly comparable.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from clean_features import fit_transform, transform
from baseline import time_based_split, mae, mape, fit_lane_baseline, predict_lane_baseline


def evaluate_model(name, model, train_X, train_y, holdout_X, holdout_distance, holdout_actual_rate):
    model.fit(train_X, train_y)
    predicted_rpm = model.predict(holdout_X)
    predicted_rate = predicted_rpm * holdout_distance.to_numpy()

    model_mae = mae(holdout_actual_rate, predicted_rate)
    model_mape = mape(holdout_actual_rate, predicted_rate)

    print(f"{name:22s}  MAE: ${model_mae:8,.2f}   MAPE: {model_mape:6.2f}%")
    return model, model_mae, model_mape


if __name__ == "__main__":
    raw = pd.read_csv("data/train-test.csv")
    train_split, holdout_split = time_based_split(raw)

    # fit on train, transform holdout using same artifacts 
    train_X, train_y, train_meta, artifacts = fit_transform(train_split)
    holdout_X, holdout_y, holdout_meta = transform(holdout_split, artifacts)

    holdout_distance = holdout_split["distance"]
    holdout_actual_rate = holdout_split["posted_rate"].to_numpy()

    print(f"Train: {train_X.shape}, Holdout: {holdout_X.shape}\n")

    # Baseline, for comparison on the same split 
    baseline = fit_lane_baseline(train_split)
    baseline_pred = predict_lane_baseline(holdout_split, baseline)
    baseline_mae = mae(holdout_actual_rate, baseline_pred)
    baseline_mape = mape(holdout_actual_rate, baseline_pred)

    print("=== Results on Sep-Oct holdout (predicting posted_rate, in dollars) ===")
    print(f"{'Baseline (lane avg)':22s}  MAE: ${baseline_mae:8,.2f}   MAPE: {baseline_mape:6.2f}%")

    # Linear Regression
    evaluate_model(
        "Linear Regression", LinearRegression(),
        train_X, train_y, holdout_X, holdout_distance, holdout_actual_rate,
    )

    # Random Forest (default-ish, light cap on depth for speed) 
    evaluate_model(
        "Random Forest", RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        train_X, train_y, holdout_X, holdout_distance, holdout_actual_rate,
    )

    # XGBoost (default-ish, this is the quick pass, not the tuned version) 
    evaluate_model(
        "XGBoost (default)", XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                                           random_state=42, n_jobs=-1),
        train_X, train_y, holdout_X, holdout_distance, holdout_actual_rate,
    )