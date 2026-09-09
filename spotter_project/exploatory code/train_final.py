"""
Final retraining XGBoost on the full dataset 


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


FINAL_XGB_PARAMS = dict(
    n_estimators=141,         
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


    features, target, meta, artifacts = fit_transform(raw)

    print(f"Feature matrix shape: {features.shape}")
    print(f"Training final XGBoost with locked-in tuned hyperparameters:")
    for k, v in FINAL_XGB_PARAMS.items():
        print(f"  {k}: {v}")

   
    final_model = XGBRegressor(**FINAL_XGB_PARAMS)
    final_model.fit(features, target)

    train_pred_rpm = final_model.predict(features)
    train_pred_rate = train_pred_rpm * meta["distance"].to_numpy()
    train_actual_rate = raw["posted_rate"].to_numpy()
    train_mae = np.mean(np.abs(train_actual_rate - train_pred_rate))
    print(f"\nSanity check - MAE on full training data (not a real metric, just a smoke test): ${train_mae:,.2f}")

    # Save model + artifacts 
    final_model.save_model("models/final_xgb_model.json")
    with open("models/artifacts.pkl", "wb") as f:
        pickle.dump(artifacts, f)

    print("\nSaved:")
    print("  models/final_xgb_model.json")
    print("  models/artifacts.pkl")