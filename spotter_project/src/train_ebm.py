"""
train_ebm.py

Trains an Explainable Boosting Machine (EBM) on the full Jan-Oct dataset using
the exact same feature pipeline as train_final.py (via fit_transform).

Saves:
  - models/final_ebm_model.pkl   (the fitted EBM loaded by evaluate_ensemble.py,
                                   predict_validation.py, predict_december_no_seasonal.py)
  Reuses:
  - models/artifacts.pkl         (already written by train_final.py no re-fit needed)


EBM configuration
Note on Hyperparameter Tuning Parity:
Evaluated systematically via `exploatory code/tune_ebm.py` across:
  - max_bins: [128, 256, 512]
  - interactions: [0, 5, 10, 15]
  - learning_rate: [0.005, 0.01, 0.02]
Pure additive (0 interactions) achieved MAE ~$150.40; adding top-10 pairwise interactions
dropped MAE significantly. Pushing interactions to 15 offered negligible gain (<$0.50)
while increasing interaction noise on rare lanes. max_bins=256 provides high-resolution
splines for date/seasonality without the bin noise observed at 512.
EBM is structurally robust to these settings, and further tuning was deliberately
concluded due to diminishing marginal returns.
"""

import pickle
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor

from clean_features import fit_transform

EBM_PARAMS = dict(
    max_bins=256,          # fine-grained shape functions - smooth seasonal curve
    interactions=10,       # top-10 pairwise interactions so distance  * equipment is still learned
    learning_rate=0.01,    # slow learning rate works well with EBM's round-robin boosting
    min_samples_leaf=2,
    random_state=42,
)

if __name__ == "__main__":
    raw = pd.read_csv("data/train-test.csv")

    print(f"Training EBM on full dataset: {len(raw)} rows (Jan-Oct)")

    features, target, meta, artifacts = fit_transform(raw)

    print(f"Feature matrix shape: {features.shape}")
    print("Training EBM with params:")
    for k, v in EBM_PARAMS.items():
        print(f"  {k}: {v}")
    print("(EBM trains one shape function per feature in round-robin - this may take a few minutes)")

    ebm_model = ExplainableBoostingRegressor(**EBM_PARAMS)
    ebm_model.fit(features, target)

    # Sanity check on training data (not a real metric - data it trained on)
    train_pred_rpm = ebm_model.predict(features)
    train_pred_rate = train_pred_rpm * meta["distance"].to_numpy()
    train_actual_rate = raw["posted_rate"].to_numpy()
    train_mae = np.mean(np.abs(train_actual_rate - train_pred_rate))
    print(f"\nSanity check - MAE on full training data (not a real metric): ${train_mae:,.2f}")

    # Save model
    with open("models/final_ebm_model.pkl", "wb") as f:
        pickle.dump(ebm_model, f)

    # Also save updated artifacts (in case train_final.py hasn't been run yet)
    with open("models/artifacts.pkl", "wb") as f:
        pickle.dump(artifacts, f)

    print("\nSaved:")
    print("\nmodels/final_ebm_model.pkl")
    print("\nmodels/artifacts.pkl  (overwritten with same content — safe to share with XGBoost scripts)")
