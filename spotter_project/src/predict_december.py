"""
predict_december_ebm.py

Generates December chart predictions using the EBM (Explainable Boosting Machine)
trained on the full Jan-Oct dataset.

Output:
  - outputs/december_ebm.csv
  - reports/final/december_ebm_chart.png
"""

import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from clean_features import transform

if __name__ == "__main__":
    # Load EBM model + shared artifacts
    with open("models/final_ebm_model.pkl", "rb") as f:
        ebm_model = pickle.load(f)

    with open("models/artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)

    december_raw = pd.read_csv("data/december-chart-inputs.csv")
    features, target_unused, meta = transform(december_raw, artifacts)

    # EBM prediction: rate_per_mile * distance - dollar rate
    predicted_rpm  = ebm_model.predict(features)
    predicted_rate = predicted_rpm * december_raw["distance"].to_numpy()

    output = december_raw.drop(columns=["predicted_rate"], errors="ignore").copy()
    output["predicted_rate"] = predicted_rate

    assert output["predicted_rate"].isna().sum() == 0, "Some rows got no prediction!"
    assert (output["predicted_rate"] > 0).all(), "Found non-positive predicted_rate!"

    output.to_csv("outputs/december_ebm.csv", index=False)
    print("Saved outputs/december_ebm.csv")
    print(output[["date", "predicted_rate"]].to_string(index=False))
    print(f"\nRange:          ${output['predicted_rate'].min():,.2f} – ${output['predicted_rate'].max():,.2f}")
    print(f"Std deviation:  ${output['predicted_rate'].std():,.2f}")

    # Plot
    dates = pd.to_datetime(output["date"])
    rates = output["predicted_rate"]

    # Zoom y-axis tight to the actual data range so the seasonal swing is visible
    y_min, y_max = rates.min(), rates.max()
    y_pad = (y_max - y_min) * 0.5          # 50% padding either side of the swing
    y_lo  = y_min - y_pad
    y_hi  = y_max + y_pad

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.fill_between(dates, rates, y_lo,    # shade between curve and bottom of axis
                    alpha=0.10, color="#6366f1")
    ax.plot(dates, rates, marker="o", markersize=5,
            color="#6366f1", linewidth=2.2, label="EBM prediction", zorder=3)

    # Annotate min and max points
    idx_min = rates.idxmin()
    idx_max = rates.idxmax()
    ax.annotate(f"${rates[idx_min]:,.0f}",
                xy=(dates[idx_min], rates[idx_min]),
                xytext=(0, -18), textcoords="offset points",
                ha="center", fontsize=9, color="#6366f1", fontweight="bold")
    ax.annotate(f"${rates[idx_max]:,.0f}",
                xy=(dates[idx_max], rates[idx_max]),
                xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=9, color="#6366f1", fontweight="bold")

    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Predicted rate ($)", fontsize=11)
    ax.set_title("December Predicted Rate — EBM", fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    os.makedirs("reports/final", exist_ok=True)
    plt.savefig("reports/final/december_ebm_chart.png", dpi=150)
    print("Saved reports/final/december_ebm_chart.png")
    plt.show()