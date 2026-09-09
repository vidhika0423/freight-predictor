# Freight Rate Predictor — Spotter Assessment

## Quick Summary

This repository contains the end-to-end pipeline for predicting `posted_rate` ($) on freight lanes. 

After comprehensive exploration and error analysis, the final production model relies on an **Explainable Boosting Machine (EBM)**. EBMs overcome structural limitations found in standard tree-based models like XGBoost when handling date/seasonality signals, successfully capturing the "December drop" in rates without manual adjustments.

* For the underlying strategy and roadmap, see [`../plan.md`](../plan.md)
* For the exploratory data analysis and domain formulas, see [`../EDA_formulas.md`](../EDA_formulas.md)



---

## The Pipeline

Our core modeling pipeline is focused on production readiness. It applies strict isolation between training and holdout to ensure there's absolutely no data leakage (e.g., city latitude/longitude lookups and weight medians are fitted on training data, then strictly applied to holdout sets). 

| File | Purpose |
|---|---|
| `src/clean_features.py` | Shared feature engineering pipeline. Artifacts learned once from training only. |
| `src/baseline.py` | Lane-average baseline model; also provides shared functions (`time_based_split`, `mae`, `mape`). |
| `src/train_ebm.py` | **Trains the production EBM model** on the full Jan-Oct data split. |
| `src/predict_validation.py` | Generates final predictions for the `validation.csv` template. |
| `src/predict_december.py` | Generates the December daily rate chart to validate seasonality. |
| `src/error_analysis.py` | Holdout error breakdown by equipment, lane, price range, with a predicted-vs-actual scatter plot. |

---

## Key Design Decisions

### 1. Target: `rate_per_mile` over `posted_rate`
By predicting `rate_per_mile = posted_rate / distance` and multiplying back by distance at the end, the model avoids learning the massive variance driven simply by length of haul. The residuals become much tighter.

### 2. EBM vs XGBoost vs Blended Ensemble
During development, I discovered a core issue: XGBoost starved the date/seasonality signal (`doy_sin` / `doy_cos`) of tree splits because `distance` and `equipment` explain ~76% of the variance. This led to a flat "staircase" prediction for December.
- **Blended Ensemble:** I tested ensembling tuned XGBoost and EBM (50/50 blend in `exploatory code/evaluate_ensemble.py`). While holdout metrics were competitive, standalone EBM matched or outperformed the ensemble while avoiding the operational cost of maintaining two separate model pipelines and preventing XGBoost's piecewise steps from diluting the December seasonal curve.
- **Why EBM Won:** By learning a **dedicated shape function per feature** in round-robin fashion, EBM natively guarantees the date signal contributes a smooth, accurate curve regardless of its smaller overall amplitude. The December extrapolation problem is resolved architecturally without post-hoc heuristic patches.

### 3. Hyperparameter Tuning Parity & Rigor
To match the rigor of XGBoost's 2-stage grid search (`exploatory code/train_tune.py`), EBM parameters were systematically evaluated in `exploatory code/tune_ebm.py` across `max_bins` (128, 256, 512), `interactions` (0, 5, 10, 15), and `learning_rate` (0.005, 0.01, 0.02) with train-holdout gap tracking:
- **Pairwise Interactions:** Adding the top 10 pairwise interactions dropped holdout MAE significantly compared to pure additive EBM. Increasing interactions to 15 offered negligible gain (<$0.50 MAE) while increasing the risk of fitting noise on rare lanes.
- **Bin Resolution:** `max_bins=256` generated smooth, high-resolution splines without the bin noise observed at 512 bins.
- **Robustness:** Because EBM's additive splines are inherently regularized, marginal gains plateaued quickly. Further micro-tuning was deliberately concluded to avoid overfitting the specific holdout window.

---

## How to Run the Pipeline

Run the following commands from the `spotter_project` directory to regenerate all outputs from scratch.

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Production Model
```bash
python src/train_ebm.py
```
*This saves the fitted model to `models/final_ebm_model.pkl` and feature scaling artifacts to `models/artifacts.pkl`.*

### 3. Evaluate & Analyze
```bash
python src/error_analysis.py
```
*This outputs `reports/final/holdout_predicted_vs_actual.png` and a breakdown of performance by feature buckets.*

### 4. Generate Final Outputs
```bash
python src/predict_validation.py
python src/predict_december.py
```
*These generate `outputs/validation_predictions.csv` and `outputs/december_ebm.csv`, alongside the seasonal chart `reports/final/december_ebm_chart.png`.*
