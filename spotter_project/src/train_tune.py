"""
train_tune.py

Step 4: Holdout training & hyperparameter tuning.

Gives XGBoost a fair, properly-tuned shot against Random Forest (which won the
untuned Step 3 comparison). Tuning priority follows the plan locked in earlier:
  1. max_depth       - controls how specific/deep a tree can get
  2. min_child_weight - stops leaves from forming around single rare rows/lanes
  3. learning_rate + n_estimators + early_stopping_rounds, tuned together

For each config we track BOTH train MAE and holdout MAE, not just holdout alone -
a large gap between them is a red flag for overfitting even if the holdout number
looks good.

Random Forest gets a light tuning pass too (max_depth, min_samples_leaf) so the
final comparison is apples-to-apples, not tuned-vs-untuned.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

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

    train_distance = train_split["distance"]
    train_actual = train_split["posted_rate"].to_numpy()
    holdout_distance = holdout_split["distance"]
    holdout_actual = holdout_split["posted_rate"].to_numpy()

    # Stage 1: tune max_depth + min_child_weight (fixed learning_rate)
    results_stage1 = []
    for max_depth in [3, 4, 5, 6]:
        for min_child_weight in [1, 5, 10, 20]:
            model = XGBRegressor(
                n_estimators=2000,
                learning_rate=0.05,
                max_depth=max_depth,
                min_child_weight=min_child_weight,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                eval_metric="mae",
                early_stopping_rounds=50,
            )
            model.fit(train_X, train_y, eval_set=[(holdout_X, holdout_y)], verbose=False)

            train_mae, train_mape = dollar_metrics(model, train_X, train_distance, train_actual)
            hold_mae, hold_mape = dollar_metrics(model, holdout_X, holdout_distance, holdout_actual)
            best_iter = model.best_iteration

            results_stage1.append({
                "max_depth": max_depth, "min_child_weight": min_child_weight,
                "best_iter": best_iter, "train_mae": train_mae, "holdout_mae": hold_mae,
                "holdout_mape": hold_mape, "gap": hold_mae - train_mae,
            })

    stage1_df = pd.DataFrame(results_stage1).sort_values("holdout_mae")
    print(stage1_df.to_string(index=False))
    best_stage1 = stage1_df.iloc[0]
    print(f"\nBest so far: max_depth={int(best_stage1.max_depth)}, "
          f"min_child_weight={int(best_stage1.min_child_weight)}, "
          f"holdout MAE=${best_stage1.holdout_mae:,.2f}\n")

    # Stage 2: tune learning_rate around the best max_depth/min_child_weight
    results_stage2 = []
    for lr in [0.01, 0.03, 0.05, 0.1]:
        model = XGBRegressor(
            n_estimators=3000,
            learning_rate=lr,
            max_depth=int(best_stage1.max_depth),
            min_child_weight=int(best_stage1.min_child_weight),
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            eval_metric="mae",
            early_stopping_rounds=50,
        )
        model.fit(train_X, train_y, eval_set=[(holdout_X, holdout_y)], verbose=False)

        train_mae, train_mape = dollar_metrics(model, train_X, train_distance, train_actual)
        hold_mae, hold_mape = dollar_metrics(model, holdout_X, holdout_distance, holdout_actual)

        results_stage2.append({
            "learning_rate": lr, "best_iter": model.best_iteration,
            "train_mae": train_mae, "holdout_mae": hold_mae,
            "holdout_mape": hold_mape, "gap": hold_mae - train_mae,
        })

    stage2_df = pd.DataFrame(results_stage2).sort_values("holdout_mae")
    print(stage2_df.to_string(index=False))
    best_stage2 = stage2_df.iloc[0]
    print(f"\nBest learning_rate: {best_stage2.learning_rate}, "
          f"holdout MAE=${best_stage2.holdout_mae:,.2f}\n")

    # Final tuned XGBoost, trained with the winning config
    final_xgb = XGBRegressor(
        n_estimators=3000,
        learning_rate=float(best_stage2.learning_rate),
        max_depth=int(best_stage1.max_depth),
        min_child_weight=int(best_stage1.min_child_weight),
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        eval_metric="mae",
        early_stopping_rounds=50,
    )
    final_xgb.fit(train_X, train_y, eval_set=[(holdout_X, holdout_y)], verbose=False)
    xgb_train_mae, xgb_train_mape = dollar_metrics(final_xgb, train_X, train_distance, train_actual)
    xgb_hold_mae, xgb_hold_mape = dollar_metrics(final_xgb, holdout_X, holdout_distance, holdout_actual)

    # ============================================================
    # Random Forest - light tuning pass, for a fair comparison
    # ============================================================
    print("=== Random Forest: light tuning pass ===")
    # NOTE: max_depth=None (fully unbounded trees) was tested and took ~110 seconds
    # PER FIT on this dataset - far too slow for a grid search, and unbounded trees
    # are also more prone to exactly the overfitting risk we're trying to avoid.
    # Capping depth keeps this fast AND more defensible.
    rf_results = []
    for max_depth in [8, 10, 12, 16]:
        for min_samples_leaf in [1, 5, 10]:
            rf = RandomForestRegressor(
                n_estimators=150, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                random_state=42, n_jobs=-1,
            )
            rf.fit(train_X, train_y)
            train_mae, _ = dollar_metrics(rf, train_X, train_distance, train_actual)
            hold_mae, hold_mape = dollar_metrics(rf, holdout_X, holdout_distance, holdout_actual)
            rf_results.append({
                "max_depth": max_depth, "min_samples_leaf": min_samples_leaf,
                "train_mae": train_mae, "holdout_mae": hold_mae,
                "holdout_mape": hold_mape, "gap": hold_mae - train_mae,
            })

    rf_df = pd.DataFrame(rf_results).sort_values("holdout_mae")
    print(rf_df.to_string(index=False))
    best_rf_row = rf_df.iloc[0]
    best_rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=int(best_rf_row.max_depth),
        min_samples_leaf=int(best_rf_row.min_samples_leaf),
        random_state=42, n_jobs=-1,
    )
    best_rf.fit(train_X, train_y)
    rf_train_mae, _ = dollar_metrics(best_rf, train_X, train_distance, train_actual)
    rf_hold_mae, rf_hold_mape = dollar_metrics(best_rf, holdout_X, holdout_distance, holdout_actual)

    # Final head-to-head
    print("\n=== FINAL HEAD-TO-HEAD (tuned models, Sep-Oct holdout) ===")
    print(f"{'Model':25s} {'Train MAE':>12s} {'Holdout MAE':>13s} {'Holdout MAPE':>13s} {'Gap':>10s}")
    print(f"{'XGBoost (tuned)':25s} ${xgb_train_mae:10,.2f} ${xgb_hold_mae:11,.2f} {xgb_hold_mape:12.2f}% "
          f"${xgb_hold_mae - xgb_train_mae:8,.2f}")
    print(f"{'Random Forest (tuned)':25s} ${rf_train_mae:10,.2f} ${rf_hold_mae:11,.2f} {rf_hold_mape:12.2f}% "
          f"${rf_hold_mae - rf_train_mae:8,.2f}")