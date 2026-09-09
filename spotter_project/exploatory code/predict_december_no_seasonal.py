"""
predict_december_no_seasonal.py

Generates December chart predictions using RAW XGBoost only - no seasonal
adjustment layer. This is what the model produces natively.

Known limitation (see error analysis / write-up): since distance, equipment,
weight, and lat/lon are all frozen in this file (same lane every day), and
XGBoost only relies on doy_sin/doy_cos for ~5.8% of its decisions overall, the
resulting chart comes out as a "staircase" - long flat stretches with sudden
jumps - rather than a smooth day-to-day curve. This file exists so that
comparison is possible against the seasonally-adjusted version.

Output:
  - outputs/december_no_seasonal.csv
  - reports/december_no_seasonal_chart.png
"""

import pickle
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

from clean_features import transform

if __name__ == "__main__":
    model = XGBRegressor()
    model.load_model("models/final_xgb_model.json")

    with open("models/artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)

    december_raw = pd.read_csv("data/december-chart-inputs.csv")
    features, target_unused, meta = transform(december_raw, artifacts)

    predicted_rpm = model.predict(features)
    predicted_rate = predicted_rpm * december_raw["distance"].to_numpy()

    output = december_raw.drop(columns=["predicted_rate"], errors="ignore").copy()
    output["predicted_rate"] = predicted_rate

    assert output["predicted_rate"].isna().sum() == 0, "Some rows got no prediction!"
    assert (output["predicted_rate"] > 0).all(), "Found non-positive predicted_rate!"

    output.to_csv("outputs/december_no_seasonal.csv", index=False)
    print("Saved outputs/december_no_seasonal.csv")
    print(output[["date", "predicted_rate"]].to_string(index=False))
    print(f"\nRange: ${output['predicted_rate'].min():,.2f} - ${output['predicted_rate'].max():,.2f}")
    print(f"Std deviation across the month: ${output['predicted_rate'].std():,.2f}")

    # Plot 
    dates = pd.to_datetime(output["date"])
    plt.figure(figsize=(10, 5))
    plt.plot(dates, output["predicted_rate"], marker="o", color="#94a3b8", linewidth=1.5)
    plt.xlabel("Date")
    plt.ylabel("Predicted rate ($)")
    plt.title("December Predicted Rate — No Seasonal Adjustment (Raw XGBoost)")
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/december_no_seasonal_chart.png", dpi=150)
    print("Saved reports/december_no_seasonal_chart.png")