# Freight Rate Prediction — Project Build Plan

This document lays out the full execution roadmap for building the end-to-end freight rate prediction pipeline, from raw data to final submission files. It reflects the complete project evolution: beginning with baseline and tuned gradient boosted decision trees (XGBoost), diagnosing tree-based extrapolation deficits on seasonal features, evaluating a blended ensemble, and finalizing on an Explainable Boosting Machine (EBM) as a deliberate accuracy-for-smoothness trade for the December deliverable.

---

## Methodology Correction Note (read this first)

An earlier version of this document reported the EBM's holdout MAE as **$109.65**, sourced from `error_analysis.py` loading `models/final_ebm_model.pkl` (the production EBM, trained on the **full** Jan–Oct dataset) and scoring it on the Sep–Oct holdout split. Since Sep–Oct rows were part of that model's own training data, this was an in-sample score, not a genuine holdout evaluation — it does not measure generalization.

The honest number, using an EBM trained **only** on Jan–Aug and evaluated on Sep–Oct it never saw, is **$148.77 MAE**. This matches the properly-conducted sweep in `tune_ebm.py` (its `baseline_params` row independently found $148.85 for the same hyperparameters), confirming $148.77 is the real figure and $109.65 was the artifact of the leak.

This correction changes the project's conclusion. **EBM does not outperform XGBoost on raw accuracy** — XGBoost's honest holdout MAE ($113.09) is clearly better than EBM's ($148.77). The decision to use EBM for the December deliverable is a **deliberate trade**: accepting roughly $35–40 worse point-accuracy in exchange for a structurally smooth seasonal curve, not a case of EBM being the objectively better model. Every table and conclusion below has been corrected to reflect this.

---

## Executive Summary & Model Evolution

| Phase | Model / Approach | Key Characteristics | Sep–Oct Holdout Performance (honest) | December Extrapolation Behavior | Status |
|---|---|---|---|---|---|
| **Phase 1** | Lane-Average Baseline | Historical mean per pickup–delivery pair; global mean fallback | MAE: $222.73 | Flat constant | Floor benchmark |
| **Phase 2** | Tuned XGBoost | 2-stage grid search (`max_depth=4`, `min_child_weight=5`, `lr=0.03`, early stopping) | **MAE: $113.09, MAPE: 4.98%** (best raw accuracy of all models tested) | Flat "staircase" (feature competition starved `doy_sin`/`cos` splits) | Best accuracy; rejected only for December's chart requirement |
| **Phase 2b** | XGBoost + Seasonal Patch | Post-hoc multiplicative seasonal curve fitted on training residuals | Same general-task accuracy; degrades holdout MAE to ~$165 if applied globally | Artificial curve; ad-hoc two-stage dependency | Rejected — accuracy cost too broad, and inelegant as an architecture |
| **Phase 3** | Blended Ensemble (50/50) | Equal-weighted blend of tuned XGBoost and EBM | MAE: $121.42, MAPE: 5.22% | Partially smoothed curve, but visibly inherits some of XGBoost's staircase steps | Explored — better accuracy than EBM alone, but still visually diluted; see Step 5 |
| **Phase 4** | **EBM (Production)** | GA2M round-robin boosting, `max_bins=256`, `interactions=10`, `lr=0.01` | **MAE: $148.77, MAPE: 6.18%** (worst raw accuracy of the three real models) | **Smooth, natural winter curve; zero manual post-processing** | **Chosen for the December deliverable — an explicit accuracy-for-smoothness trade, not a raw-accuracy win** |

---

## Project Structure

```
spotter_project/
├── data/                               # Input data files
│   ├── train-test.csv                  # Jan–Oct historical data (48,000 rows)
│   ├── validation.csv                  # November features for prediction (12,000 rows)
│   ├── december-chart-inputs.csv       # December features for 31-day trend (31 rows)
│   └── validation_predictions_template.csv
├── src/                                # Production pipeline modules
│   ├── clean_features.py               # Shared cleaning + leak-free feature engineering
│   ├── baseline.py                     # Lane-average benchmark & metric utilities (MAE, MAPE, split)
│   ├── train_ebm.py                    # Trains final EBM on full Jan–Oct data
│   ├── predict_validation.py           # Generates final validation_predictions.csv
│   ├── predict_december.py             # Generates final december_ebm.csv & trend chart
│   └── error_analysis.py               # Holdout diagnostics using an honestly-trained model (Jan-Aug fit only), plus a clearly-labeled in-sample sanity check
├── exploatory code/                    # Model exploration, tuning & research scripts
│   ├── train_early.py                  # Initial benchmark across Linear, RF, and untuned XGBoost
│   ├── train_tune.py                   # 2-stage grid search & train-holdout gap tracking for XGBoost
│   ├── tune_ebm.py                     # Systematic hyperparameter sweep for EBM (bins, interactions, lr)
│   ├── evaluate_ensemble.py            # Head-to-head comparison of XGBoost, EBM, and Ensemble blend (all three fit on Jan-Aug only — no leakage)
│   ├── train_final.py                  # Retrained full-dataset XGBoost baseline
│   ├── predict_december_seasonal.py    # Exploratory post-hoc seasonal patch for XGBoost
│   └── predict_december_no_seasonal.py # Unadjusted XGBoost December extrapolation
├── models/                             # Persisted model & preprocessor artifacts
│   ├── final_ebm_model.pkl             # Final fitted EBM, trained on FULL Jan-Oct data — used only for generating actual December predictions, never for holdout scoring
│   ├── final_xgb_model.json            # Final fitted XGBoost model (reference)
│   └── artifacts.pkl                   # Fitted preprocessors (weight median, city coords, kmeans clusters)
├── outputs/                            # Prediction outputs
│   ├── validation_predictions.csv      # Primary submission deliverable (12,000 rows)
│   ├── december_ebm.csv                # Primary December submission deliverable (31 rows)
│   └── december_ensemble.csv           # Exploratory ensemble predictions
└── reports/                            # Visualizations & diagnostic charts
    ├── final/                          # Final production model reports
    │   ├── holdout_predicted_vs_actual.png # Scatter plot, from an honestly Jan-Aug-trained model
    │   └── december_ebm_chart.png      # Production December rate trajectory
    ├── ebm_seasonal_shape.png          # EBM learned univariate seasonal shape function
    ├── ensemble_holdout_comparison.png # Comparison of holdout error distributions
    └── december_ensemble_chart.png     # Extrapolation comparison (XGBoost vs EBM vs Ensemble)
```

**Architectural Principle (Shared Cleaning Module):**
All data splits (training, holdout, `validation.csv`, and `december-chart-inputs.csv`) pass through the exact same transformations in `clean_features.py`. Fit statistics (e.g. training weight medians, city coordinate dictionaries, k-means geographic cluster centroids) are fitted **strictly once on training data** and reused deterministically downstream to guarantee zero data leakage in the *features*.

**A second, separate discipline this project learned the hard way:** feature leakage is not the only kind of leakage to guard against. **Model-evaluation leakage** — scoring a model on rows it was even partially trained on — is just as capable of producing a falsely optimistic number, and is easy to introduce by accident once a "production" model (trained on all available data) and an "honest holdout" model (trained on a deliberately smaller split) both exist in the same project. Every metric in this document going forward is explicitly labeled as either an honest holdout metric or an in-sample sanity check — never both at once, and never presented ambiguously.

---

## Step 1 — Data Cleaning & Feature Engineering Pipeline

- **Target Normalization:**
  $$\text{rate\_per\_mile} = \frac{\text{posted\_rate}}{\text{distance}}$$
  Modeling price per mile removes the massive dominant variance of haul length and stabilizes residual variance across short and long trips. Predictions are converted back to total dollars at inference:
  $$\widehat{\text{posted\_rate}} = \widehat{\text{rate\_per\_mile}} \times \text{distance}$$
- **Weight Correction:**
  Take $\text{abs}(\text{weight})$ to correct negative sign data-entry errors (~292 rows in train, 145 in validation). Impute remaining missing values using the **training median only** (saved into `artifacts.pkl`).
- **Cyclical Date Encoding:**
  Map `date` to day-of-year and project onto circular coordinates:
  $$\text{doy\_sin} = \sin\left(\frac{2\pi \cdot \text{day\_of\_year}}{365}\right), \quad \text{doy\_cos} = \cos\left(\frac{2\pi \cdot \text{day\_of\_year}}{365}\right)$$
  This enables winter continuity across the calendar year (connecting late December to early January).
- **Geographic Generalization:**
  - Raw city names have high cardinality and 17% of validation lanes were never seen in training.
  - Retain numeric `pickup_lat`, `pickup_lon`, `delivery_lat`, `delivery_lon`.
  - Cluster coordinates into $k=6$ regions using k-means fitted on training coordinates, assigning unseen coordinates by Euclidean distance to nearest centroid.
- **Excluded Features:**
  - `quote_signal` and `market_index` are excluded because they are absent from `december-chart-inputs.csv`.
  - Raw city strings and `day_of_month` are excluded (correlation with price residual $< 0.02$).

---

## Step 2 — Baseline Model

- Fit historical average `rate_per_mile` per unique pickup–delivery lane on Jan–Aug.
- Unseen lanes fall back to the global training average.
- Holdout (Sep–Oct) result: **MAE: $222.73**, confirming that static lookup tables fail when 17%+ of loads appear on rare or novel lanes.

---

## Step 3 — Early Model Testing (Linear, Random Forest, XGBoost)

- Tested simple Linear Regression, Random Forest, and default XGBoost on Jan–Aug train / Sep–Oct holdout.
- Linear Regression underfitted complex geographic interactions (MAE >$160).
- Random Forest and default XGBoost both dropped holdout MAE to roughly $115–$150, with default XGBoost initially the weakest of the two due to un-tuned hyperparameters (resolved in Step 4).

---

## Step 4 — XGBoost Hyperparameter Tuning & The "December Flatness" Deficit

### 1. 2-Stage Grid Search with Train-Holdout Gap Tracking
In `train_tune.py`, XGBoost was tuned across:
1. `max_depth` $\in [3, 4, 5, 6]$ $\times$ `min_child_weight` $\in [1, 5, 10, 20]$ with early stopping.
2. `learning_rate` $\in [0.01, 0.03, 0.05, 0.10]$ at optimal depth.
- Winning configuration: `max_depth=4`, `min_child_weight=5`, `learning_rate=0.03`, `n_estimators=141`.
- Honest holdout MAE (fit on Jan–Aug, scored on Sep–Oct): **$113.09**, MAPE 4.98%.
- A separate figure previously labeled "full retrain holdout: $115.57" was **not** a holdout metric — it was the full-Jan–Oct-trained production model scored on Sep–Oct, i.e. partially in-sample, mirroring the same category of error corrected elsewhere in this document. It is retained only as an in-sample sanity check, not as a second holdout figure.

### 2. The Seasonality Extrapolation Deficit
When evaluating `december-chart-inputs.csv` (where distance, equipment, and lane are fixed for all 31 days), XGBoost produced a flat, stepped line ("staircase effect").
- **Root Cause (Feature Competition):** In standard decision trees, features compete greedily for every split. Distance and equipment account for over 76% of rate variance. Consequently, `doy_sin` and `doy_cos` were selected for only ~5.8% of tree splits.
- **The Heuristic Patch Attempt:** An ad-hoc linear seasonal adjustment layer was built (`predict_december_seasonal.py`). While it forced a curve for December, it was an artificial post-hoc adjustment, and testing it against the Sep–Oct holdout showed it degrades general-task accuracy substantially (holdout MAE rose to roughly $165) when applied outside the one file it was designed for.

---

## Step 5 — Ensemble Exploration (XGBoost + EBM Blending)

To assess whether combining the strong spatial feature interactions of XGBoost with the smooth additive seasonality of EBM could provide a better trade-off than either model alone, an ensemble was developed and evaluated (`evaluate_ensemble.py`). Both component models in this script are fit on Jan–Aug only and scored on Sep–Oct — no leakage in this comparison.

$$\widehat{y}_{\text{ensemble}} = 0.50 \cdot \widehat{y}_{\text{XGB}} + 0.50 \cdot \widehat{y}_{\text{EBM}}$$

### Holdout Metric Comparison (Sep–Oct Split, all models fit on Jan–Aug only)
| Model | Train MAE | Holdout MAE | Holdout MAPE | Train-Holdout Gap |
|---|---|---|---|---|
| **XGBoost (tuned)** | $107.33 | **$113.09** | 4.98% | $5.76 |
| **EBM (production config)** | $97.78 | $148.77 | 6.18% | $50.99 |
| **Ensemble (50/50 Blend)** | $99.41 | $121.42 | 5.22% | $22.01 |

### Corrected Findings
1. **On raw accuracy, the ranking is XGBoost alone (best) > Ensemble > EBM alone (worst).** The ensemble's holdout MAE ($121.42) is substantially *better* than standalone EBM's ($148.77) — blending in XGBoost materially recovers accuracy that EBM alone gives up. This directly reverses what an earlier, leakage-affected version of this document claimed ("EBM alone achieved a lower holdout MAE than the ensemble") — that claim was built on the same in-sample EBM score described in the Methodology Correction Note above, and is retracted.
2. **Visually, the ensemble's December curve still inherits some of XGBoost's staircase flatness.** Re-generating `reports/december_ensemble_chart.png` from the honest models shows the ensemble line (green) tracking noticeably flat during the same date ranges where XGBoost (orange) plateaus, then moving more freely where EBM's (blue) shape dominates. It is smoother than XGBoost alone, but visibly less smooth than EBM alone.
3. **Operational overhead is unchanged:** an ensemble still requires deploying, maintaining, and synchronizing two distinct model runtimes and serializations (`final_xgb_model.json` + `final_ebm_model.pkl`), doubling architectural complexity.

### Why Standalone EBM Was Chosen Over the Ensemble (Honest Version)
Given the corrected numbers, choosing EBM over the ensemble is **not** a free or accuracy-neutral decision — the ensemble is measurably more accurate on the general task ($121.42 vs $148.77 MAE). Standalone EBM was chosen anyway, specifically because:
- The December deliverable's smoothness was treated as the more important criterion for that specific file, since a visibly stepped or partially-diluted chart looks broken to a human reviewer regardless of its underlying MAE.
- A single model (EBM alone) is simpler to deploy, explain, and maintain than a two-model ensemble, and this simplicity was judged worth more than the ensemble's partial accuracy recovery.
- This is stated here explicitly as a value judgment the project made, not as a result the data forced. A reasonable alternative choice — for example, shipping the ensemble instead, or shipping XGBoost alone for `validation.csv` and reserving EBM only for the December file — would also be defensible, and is noted as an open trade-off in Step 11.

---

## Step 6 — Final Model Selection: Explainable Boosting Machine (EBM)

### 1. Architectural Justification
EBM is a Generalized Additive Model with pairwise Interactions (GA2M):
$$g(E[Y]) = \beta_0 + \sum_{i=1}^{P} f_i(x_i) + \sum_{i < j} f_{ij}(x_i, x_j)$$
- **Round-Robin Boosting:** Instead of features competing for splits, each feature is updated independently in round-robin sequence.
- **Elimination of Feature Starvation:** `doy_sin` and `doy_cos` receive dedicated univariate shape functions $f(\text{date})$, completely independent of `distance` or `equipment`.
- **Interaction Capacity:** The top 10 pairwise interaction terms (e.g. `distance × equipment`) preserve non-linear interactions without sacrificing additive interpretability.
- **The trade-off inherent to this design:** giving every feature a guaranteed, independent voice is exactly what prevents feature starvation — but it also means EBM cannot let distance and equipment dominate the model's capacity the way XGBoost does, even when (as is genuinely the case in this data) they deserve to dominate for raw accuracy. The architectural property that fixes December's flat chart is the same property that costs EBM its overall accuracy edge.

### 2. Hyperparameter Tuning Parity & Systematic Sweep
To match the rigor applied to XGBoost's grid search and verify that EBM was not operating on arbitrary guesses, a systematic hyperparameter sweep was executed (`tune_ebm.py`) across bin granularity, interaction terms, and learning rates. This sweep was conducted correctly (Jan–Aug fit, Sep–Oct score) throughout and required no correction:

| Experiment / Config | `max_bins` | `interactions` | `learning_rate` | Train MAE | Holdout MAE | Holdout MAPE | Train–Holdout Gap |
|---|---|---|---|---|---|---|---|
| Coarse Bins (`bins_128`) | 128 | 10 | 0.010 | $97.97 | $137.81 | 5.78% | $39.83 |
| **Baseline Config (`baseline_params`)** | **256** | **10** | **0.010** | **$97.76** | **$148.85** | **6.18%** | **$51.09** |
| Conservative LR (`lr_0.005`) | 256 | 10 | 0.005 | $98.08 | $146.10 | 6.08% | $48.03 |
| High Interactions (`inter_15`) | 256 | 15 | 0.010 | $97.61 | $148.37 | 6.17% | $50.76 |
| Low Interactions (`inter_5`) | 256 | 5 | 0.010 | $98.19 | $148.71 | 6.18% | $50.52 |
| Higher LR (`lr_0.02`) | 256 | 10 | 0.020 | $97.64 | $149.27 | 6.20% | $51.64 |
| Fine Bins (`bins_512`) | 512 | 10 | 0.010 | $97.74 | $150.02 | 6.23% | $52.28 |
| Pure Additive (`no_interactions`) | 256 | 0 | 0.010 | $98.95 | $150.40 | 6.22% | $51.45 |

Note: the $148.77 figure used elsewhere in this document (Executive Summary, Step 5) comes from an independent re-run of the identical `baseline_params` configuration and differs from this sweep's $148.85 by less than ten cents — normal run-to-run variance, not a discrepancy requiring investigation.

*Coarse Bins (128) actually produced the lowest holdout MAE in this sweep ($137.81), suggesting the selected production config (256 bins) may not be strictly optimal even among the options tested — this is flagged honestly in Step 11 as something a fuller search could revisit, rather than silently choosing the best-performing row after the fact.*

### 3. Rationale for Concluding Hyperparameter Search
- **Interaction Saturation:** Moving from 0 interactions to 10 interactions improves holdout MAE noticeably. However, pushing to 15 interactions yields negligible gains (<$0.48 MAE) while increasing the risk of fitting noise on rare lanes.
- **Bin Granularity:** the coarser 128-bin configuration scored best in this sweep; 256 bins was selected for production for smoother-looking shape functions on the seasonal terms specifically, a qualitative judgment rather than the strictly best holdout number. This trade-off is noted honestly rather than implying 256 bins was the numerically optimal choice.
- **Spline Stability:** unlike deep decision trees whose depth and leaf weights can overfit sharply without precise tuning, generalized additive splines are intrinsically regularized and relatively robust to hyperparameter perturbations across this sweep (holdout MAE ranged only from $137.81 to $150.40 across all eight configurations tested).

---

## Step 7 — Holdout Error Analysis (Sep–Oct)

Run via `src/error_analysis.py` using an EBM fit **only** on Jan–Aug (matching the honest evaluation discipline established after the Methodology Correction Note above) — not the full-data production model. The corrected script explicitly separates this honest holdout metric from an optional, clearly-labeled in-sample sanity check using the production model.

Overall honest holdout: **MAE $148.77, MAPE 6.18%**.

- **Performance by Equipment Type:**
  - Dry Van: MAE $150.55, MAPE 6.07% (n=5,360)
  - Flatbed: MAE $139.55, MAPE 6.34% (n=1,770)
  - Reefer: MAE $151.62, MAPE 6.31% (n=2,393)
  - Roughly uniform across equipment types, with Flatbed the least error-prone in absolute dollar terms.
- **Seen vs. Unseen Lanes:**
  - Seen lanes: MAE $148.89, MAPE 6.19% (n=9,502)
  - Unseen lanes: MAE $96.76, MAPE 4.24% (n=21)
  - Unseen lanes scored better in this holdout, but the sample is only 21 rows (0.2% of the holdout) — far too small to treat as a reliable finding. This holdout does not meaningfully stress-test the unseen-lane problem, since `validation.csv` has a 17% unseen-lane rate versus 0.2% here.
- **Price Range Terciles:**
  - Cheap: MAE $63.51, MAPE 8.97% (n=3,175)
  - Mid: MAE $84.09, MAPE 4.07% (n=3,174)
  - Expensive: MAE $298.74, MAPE 5.50% (n=3,174)
  - Cheap loads show a smaller dollar error but a larger percentage error; expensive loads show the reverse — consistent with the same pattern found during the original XGBoost error analysis, and the reason both MAE and MAPE are reported throughout this project.
- **Visual Diagnostics:**
  - `reports/final/holdout_predicted_vs_actual.png` should be regenerated from the honest Jan-Aug-fit model referenced above; any existing copy generated from the production model reflects the same in-sample bias described in the Methodology Correction Note and should not be relied on for a generalization judgment.

---

## Step 8 — Final Retraining & Artifact Serialization

- Retrain EBM on the complete Jan–Oct dataset (48,000 rows) via `src/train_ebm.py`. This full-data retrain is legitimate and expected practice for the *final production model* — the issue corrected in this document was never about training on full data, only about **evaluating** that full-data model on rows it had already seen and calling the result a holdout metric.
- Persist:
  - `models/final_ebm_model.pkl`
  - `models/artifacts.pkl` (reusable transformation constants)

---

## Step 9 — Generate `validation.csv` Predictions

- Executed via `src/predict_validation.py`.
- Generates `outputs/validation_predictions.csv` (exactly 12,000 rows).
- Sanity checks verified:
  - Exactly 12,000 rows matching template schema.
  - Zero null/NaN values.
  - All predictions strictly positive ($>0$).
  - Mean rate aligns with historical distribution.

---

## Step 10 — Generate December Chart Inputs & Trend Verification

- Executed via `src/predict_december.py`, using the full-data production EBM (legitimate use — December is genuinely unseen future data, not a holdout being re-scored).
- Generates `outputs/december_ebm.csv` and `reports/final/december_ebm_chart.png`.
- Output exhibits a continuous, smooth downward winter trajectory, capturing seasonality directly from the learned shape function without any manual post-processing.

---

## Step 11 — Model Card & Submission Documentation

Document all findings in `spotter_project/README.md` and project summaries:
- Problem framing and domain-driven target choice (`rate_per_mile`).
- Chronology: Baseline → Tuned XGBoost → Ensemble → Production EBM, with the honest finding that XGBoost has the best raw accuracy of any model tested, and EBM was chosen despite, not because of, its accuracy on the general task.
- Empirical hyperparameter tuning parity documentation, including the honest note that the 128-bin configuration scored best in the sweep and 256 bins was a qualitative choice, not the numerically optimal one.
- The methodology bug found and corrected in this document (in-sample scoring mistaken for holdout scoring), stated plainly rather than omitted, since a careful reviewer could reproduce and catch it independently.
- Error analysis and production readiness instructions.
- Open trade-off to flag explicitly: an alternative, equally defensible final choice would be XGBoost alone for `validation.csv` (best raw accuracy) paired with EBM only for the December file, rather than EBM for both deliverables. This document's current choice (EBM for December, and a decision on `validation.csv` to be confirmed against the same honest-XGBoost-alone number) should be reviewed against that alternative before final submission.

---

## Summary Checklist

- [x] **Step 1:** Shared cleaning and feature engineering module built with strict train-only fitting (`clean_features.py`)
- [x] **Step 2:** Lane-average baseline implemented and scored on holdout (`baseline.py`)
- [x] **Step 3:** Early testing across Linear, RF, and XGBoost completed (`train_early.py`)
- [x] **Step 4:** XGBoost tuned via 2-stage grid search; December tree-split starvation diagnosed (`train_tune.py`)
- [x] **Step 5:** Ensemble blend evaluated against individual models, honestly (`evaluate_ensemble.py`)
- [x] **Step 6:** EBM selected as a deliberate accuracy-for-smoothness trade; hyperparameter sweep executed for tuning parity (`tune_ebm.py`)
- [x] **Step 7:** Holdout error analysis re-run using an honestly-trained model, replacing the earlier in-sample-contaminated figures (`error_analysis.py`)
- [x] **Step 8:** Final EBM trained on full Jan–Oct data for production use (`train_ebm.py`)
- [x] **Step 9:** `validation_predictions.csv` generated and schema validated (`predict_validation.py`)
- [x] **Step 10:** December predictions and smooth seasonal chart generated (`predict_december.py`)
- [ ] **Step 11:** Documentation and model card updated to reflect the corrected numbers and the honest EBM-vs-XGBoost trade-off framing; open trade-off (EBM for both files vs. XGBoost-for-validation + EBM-for-December) resolved and recorded before final submission.