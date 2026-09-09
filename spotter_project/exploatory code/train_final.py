"""
train_final.py

Step 6: Final retraining on the full dataset.

The Jan-Aug/Sep-Oct split was only for HONEST TESTING during development (Steps
3-5). Now that XGBoost's hyperparameters and approach are validated, we retrain on
ALL available labeled data (Jan-Oct, 48,000 rows) before generating real
predictions - we don't want to throw away two good months of real signal.

Saves:
  - models/final_xgb_model.json       (the trained model)
  - models/artifacts.pkl              (weight median, city lookup, cluster model -
                                        needed to transform validation/december data
                                        the exact same way as training data)
"""

import pickle
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from clean_features import fit_transform

# Final tuned config, locked in during Step 4
FINAL_XGB_PARAMS = dict(
    n_estimators=141,          # best_iteration found via early stopping in Step 4
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
)

if __name__ == "__main__":
    raw = pd.read_csv("data/train-test.csv")

    print(f"Training on full dataset: {len(raw)} rows (Jan-Oct)")

    # fit_transform learns the weight median / city lookup / cluster model from
    # the FULL dataset this time (previously it only learned from Jan-Aug).
    features, target, meta, artifacts = fit_transform(raw)

    print(f"Feature matrix shape: {features.shape}")
    print(f"Training final XGBoost with locked-in tuned hyperparameters:")
    for k, v in FINAL_XGB_PARAMS.items():
        print(f"  {k}: {v}")

    # NOTE: no early stopping / eval_set here - there's no holdout left to watch,
    # since we deliberately used ALL data. n_estimators is fixed at the exact
    # best_iteration number found during Step 4's early-stopping run, rather than
    # guessing a new number - this avoids re-introducing the overfitting risk that
    # early stopping was protecting against.
    final_model = XGBRegressor(**FINAL_XGB_PARAMS)
    final_model.fit(features, target)

    # Sanity check: predict on training data itself, just to confirm nothing is
    # obviously broken (not a real evaluation - we already did that in Steps 4-5).
    train_pred_rpm = final_model.predict(features)
    train_pred_rate = train_pred_rpm * meta["distance"].to_numpy()
    train_actual_rate = raw["posted_rate"].to_numpy()
    train_mae = np.mean(np.abs(train_actual_rate - train_pred_rate))
    print(f"\nSanity check - MAE on full training data (not a real metric, just a smoke test): ${train_mae:,.2f}")

    # --- Save model + artifacts ---
    final_model.save_model("models/final_xgb_model.json")
    with open("models/artifacts.pkl", "wb") as f:
        pickle.dump(artifacts, f)

    print("\nSaved:")
    print("  models/final_xgb_model.json")
    print("  models/artifacts.pkl")