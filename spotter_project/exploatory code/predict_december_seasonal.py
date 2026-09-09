"""
predict_december_seasonal.py

Generates December chart predictions using XGBoost + the seasonal adjustment
layer. Used ONLY for this file - tested and confirmed (via the Sep-Oct holdout)
that applying this same layer to validation.csv makes accuracy worse there, since
that file already has full feature variation (distance/equipment/geography) doing
most of the explanatory work.

Why this layer exists: in this file, distance, equipment, weight, and lat/lon are
all frozen (same lane every day), so date is the ONLY thing that can move the
prediction. XGBoost only relies on doy_sin/doy_cos for ~5.8% of its decisions
overall (correct behavior given the full dataset), so without this layer the
resulting chart is a flat "staircase" rather than a real curve.

How it works:
  1. Fit rate_per_mile ~ doy_sin + doy_cos on the full training data - isolates
     the real seasonal cycle (the ~15% Jan-June-Oct swing found during EDA),
     independent of distance/equipment's much larger influence.
  2. Keep XGBoost's own overall price LEVEL for this lane/equipment/weight.
  3. Reshape the day-to-day SHAPE across December using the seasonal curve's
     relative pattern, anchored so December's average still matches XGBoost's
     base level.

Output:
  - outputs/december_with_seasonal.csv
  - reports/december_with_seasonal_chart.png
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression

from clean_features import transform


def fit_seasonal_curve(train_raw: pd.DataFrame):
    df = train_raw.copy()
    dates = pd.to_datetime(df["date"])
    doy = dates.dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365)
    df["rate_per_mile"] = df["posted_rate"] / df["distance"]

    X = df[["doy_sin", "doy_cos"]].to_numpy()
    y = df["rate_per_mile"].to_numpy()
    reg = LinearRegression().fit(X, y)
    mean_level = reg.predict(X).mean()
    return reg, mean_level


def seasonal_index(reg, mean_level, doy_sin, doy_cos):
    X = np.column_stack([doy_sin, doy_cos])
    return reg.predict(X) / mean_level


if __name__ == "__main__":
    model = XGBRegressor()
    model.load_model("models/final_xgb_model.json")

    with open("models/artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)

    december_raw = pd.read_csv("data/december-chart-inputs.csv")
    features, target_unused, meta = transform(december_raw, artifacts)

    predicted_rpm = model.predict(features)

    train_raw = pd.read_csv("data/train-test.csv")
    seasonal_reg, seasonal_mean_level = fit_seasonal_curve(train_raw)

    dec_seasonal_idx = seasonal_index(seasonal_reg, seasonal_mean_level,
                                       features["doy_sin"].to_numpy(), features["doy_cos"].to_numpy())

    # Anchor to XGBoost's own average level for this lane, so we're reshaping the
    # SHAPE across days, not overriding the model's overall price estimate.
    base_level = predicted_rpm.mean()
    december_idx_mean = dec_seasonal_idx.mean()
    adjusted_rpm = base_level * dec_seasonal_idx / december_idx_mean
    predicted_rate = adjusted_rpm * december_raw["distance"].to_numpy()

    output = december_raw.drop(columns=["predicted_rate"], errors="ignore").copy()
    output["predicted_rate"] = predicted_rate

    assert output["predicted_rate"].isna().sum() == 0, "Some rows got no prediction!"
    assert (output["predicted_rate"] > 0).all(), "Found non-positive predicted_rate!"

    output.to_csv("outputs/december_with_seasonal.csv", index=False)
    print("Saved outputs/december_with_seasonal.csv")
    print(output[["date", "predicted_rate"]].to_string(index=False))
    print(f"\nRange: ${output['predicted_rate'].min():,.2f} - ${output['predicted_rate'].max():,.2f}")
    print(f"Std deviation across the month: ${output['predicted_rate'].std():,.2f}")

    # --- Plot ---
    dates = pd.to_datetime(output["date"])
    plt.figure(figsize=(10, 5))
    plt.plot(dates, output["predicted_rate"], marker="o", color="#2563eb", linewidth=1.5)
    plt.xlabel("Date")
    plt.ylabel("Predicted rate ($)")
    plt.title("December Predicted Rate — With Seasonal Adjustment")
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/december_with_seasonal_chart.png", dpi=150)
    print("Saved reports/december_with_seasonal_chart.png")