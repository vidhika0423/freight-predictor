"""
predict_validation.py

Step 7: Generate predictions for validation.csv (the 12,000 loads requiring final
predictions) using the final model trained on the full Jan-Oct dataset.

Loads the saved model + artifacts from Step 6, runs validation.csv through the SAME
cleaning/feature pipeline (via transform(), never re-fit), predicts rate_per_mile,
converts back to a dollar predicted_rate, and fills the official template.

Output: validation_predictions.csv (matches validation_predictions_template.csv's
load_id order/schema, as required).
"""

import pickle
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from clean_features import transform

if __name__ == "__main__":
    # --- Load saved model + artifacts from Step 6 ---
    model = XGBRegressor()
    model.load_model("models/final_xgb_model.json")

    with open("models/artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)

    # --- Load validation data + the official template ---
    validation_raw = pd.read_csv("data/validation.csv")
    template = pd.read_csv("data/validation-predictions-template.csv")

    print(f"validation.csv rows: {len(validation_raw)}")
    print(f"template rows: {len(template)}")

    # --- Run through the exact same feature pipeline (transform, not fit_transform) ---
    features, target_unused, meta = transform(validation_raw, artifacts)

    # --- Predict rate_per_mile, convert back to dollar predicted_rate ---
    predicted_rpm = model.predict(features)
    predicted_rate = predicted_rpm * validation_raw["distance"].to_numpy()

    # --- Build output, matched by load_id to the template's exact row order ---
    predictions_df = pd.DataFrame({
        "load_id": validation_raw["load_id"],
        "predicted_rate": predicted_rate,
    })

    output = template[["load_id"]].merge(predictions_df, on="load_id", how="left")

    # --- Sanity checks before saving ---
    assert len(output) == len(template), "Row count mismatch with template!"
    assert output["predicted_rate"].isna().sum() == 0, "Some load_ids got no prediction!"
    assert (output["predicted_rate"] > 0).all(), "Found non-positive predicted_rate!"
    assert set(output["load_id"]) == set(template["load_id"]), "load_id sets don't match!"

    print(f"\nPrediction distribution:")
    print(output["predicted_rate"].describe())

    print(f"\nTraining target (posted_rate) distribution, for comparison:")
    train_raw = pd.read_csv("data/train-test.csv")
    print(train_raw["posted_rate"].describe())

    output.to_csv("outputs/validation_predictions.csv", index=False)
    print(f"\nSaved outputs/validation_predictions.csv ({len(output)} rows)")