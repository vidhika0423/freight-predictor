# EDA & Feature Engineering — Mathematical & Formula Reference

Note: The formulas in this document are intentionally written in plain text rather than LaTeX. This makes the file render reliably on GitHub without requiring MathJax/LaTeX support. The descriptions and numerical results are preserved from the original EDA document.



This document provides a comprehensive, rigorous reference for every mathematical formula, statistical test, and feature transformation used across Exploratory Data Analysis (EDA), feature engineering, baseline modeling, ensemble experimentation, and final model evaluation for the Spotter.ai Freight Rate Prediction project.

## 1. Target Modeling & Reconstruction

Instead of predicting raw total dollars `posted_rate` directly, the modeling pipeline normalizes by trip distance.

### 1.1 Target Normalization (Rate per Mile)

Formula (plain text): Formula: textrate_per_mile_i = fractextposted_rate_itextdistance_i

### 1.2 Dollar Rate Reconstruction at Inference

Formula (plain text): Formula: widehattextposted_rate_i = widehattextrate_per_mile_i times textdistance_i

* Rationale: Haul distance alone accounts for $>70%$ of raw dollar variance ($R^2 \approx 0.76$). Regressing directly on total dollars causes the learning algorithm to expend its capacity rediscovering "longer distance = higher dollar price." Normalizing isolates the unit pricing signal (market conditions, equipment type, lane demand, seasonality) and stabilizes residual variance across 70-mile hauls and 2,500-mile hauls.

## 2. Temporal & Cyclical Feature Engineering

Training data spans January through October. Raw sequential dates or linear day counts cannot be extrapolated by tree-based or spline models beyond October (they would predict late-October values flat across December).

### 2.1 Angular Day-of-Year Projection

Formula (plain text): Formula: theta_i = frac2pi cdot textday_of_year_i365

### 2.2 Cyclical Sine / Cosine Decomposition

Formula (plain text): Formula: textdoy_sin_i = sinleft(frac2pi cdot textday_of_year_i365right)



Formula (plain text): Formula: textdoy_cos_i = cosleft(frac2pi cdot textday_of_year_i365right)

* Continuous Cycle Boundary:

 

Formula (plain text): Formula: lim_textdate to textDec 31 (textdoy_sin, textdoy_cos) approx (0, 1) = textJan 1 coordinates

  December coordinates $(\approx -0.49, 0.87)$ land directly adjacent in Euclidean feature space to January coordinates $(\approx 0.02, 1.00)$, allowing the model to naturally generalize winter freight dynamics learned in January to December without discontinuous extrapolation.

## 3. Data Cleaning & Imputation

### 3.1 Weight Sign Correction

Formula (plain text): Formula: w_textclean, i = |w_i|

* Diagnostic Check: ~292 rows in training and 145 rows in validation contained negative weights (e.g., $-47,500\text{ lbs}$). Taking the absolute value preserved the correlation with rate ($r \approx 0.09$), confirming data-entry sign inversion rather than corrupted records.

### 3.2 Leak-Free Median Imputation

Formula (plain text): Formula: tildew_texttrain = textmedianleft(|w_i| : i in mathcalD_texttrain, w_i text is not nullright)



Formula (plain text): Formula: w_textfinal, i = begincases |w_i| & textif  w_i neq textnull  tildew_texttrain & textif  w_i = textnull endcases

* Data Isolation Rule: $\tilde{w}_{\text{train}}$ is computed strictly once on training data and saved to `artifacts.pkl`. It is applied as a fixed constant to holdout, validation, and December datasets to prevent lookahead leakage.

### 3.3 Coordinate Fallback (Unseen Cities)

For rare rows missing coordinate metadata:

Formula (plain text): Formula: (barmu_textlat, barmu_textlon) = left(frac1N_texttrain sum_i in mathcalD_texttrain textpickup_lat_i, ; frac1N_texttrain sum_i in mathcalD_texttrain textpickup_lon_iright)

## 4. Spatial Engineering & Geographic Clustering

### 4.1 Great-Circle (Haversine) Distance Verification

Used to validate road distance integrity against physical straight-line distance:

Formula (plain text): Formula: a = sin^2left(fracDelta textlat2right) + cos(textlat_1) cos(textlat_2) sin^2left(fracDelta textlon2right)



Formula (plain text): Formula: d_texthaversine = 2 R arcsinleft(sqrtaright), quad R = 3958.8text miles

### 4.2 Circuity / Routing Tortuosity Ratio

Formula (plain text): Formula: tau_i = fractextdistance_id_texthaversine, i

* Finding: $\tau$ sits consistently in the $[1.06, 1.20]$ band across the dataset, validating the `distance` column as authentic driving mileage. A small cluster of rows at $\text{distance} = 70$ with low $d_{\text{haversine}}$ represents standard commercial freight minimum billing floors, not corruptions.

### 4.3 K-Means Regional Clustering Objective

To group coordinates into coarse geographic market hubs:

Formula (plain text): Formula: argmin_mathcalS sum_j=1^k sum_x in S_j |x - boldsymbolmu_j|_2^2

### 4.4 Heuristic Derivation of Optimal Number of Clusters ($k$)

The dataset contains $48,000$ rows, but only $N_{\text{cities}} = 64$ unique geographic nodes. Applying the spatial rule of thumb:

Formula (plain text): Formula: k approx sqrtfracN_textcities2 = sqrtfrac642 = sqrt32 approx 5.66 implies k = 6

### 4.5 Nearest Centroid Assignment for Unseen Cities

Unseen coordinates $x^*$ are assigned deterministically without re-fitting:

Formula (plain text): Formula: textcluster(x^) = argmin_j in 1,dots,6 |x^ - boldsymbolmu_j|_2

## 5. Statistical Screening & Feature Selection

### 5.1 Pearson Correlation Coefficient

Used for pairwise linear association tests:

Formula (plain text): Formula: r_xy = fracsum_i=1^n (x_i - barx)(y_i - bary)sqrtsum_i=1^n (x_i - barx)^2 sqrtsum_i=1^n (y_i - bary)^2

* Key Findings:

  - $r(\text{weight}, \text{rate\_per\_mile}) \approx 0.09$ (very weak)

  - $r(\text{day\_of\_year}, \text{daily\_avg\_rpm}) \approx 0.44$ (moderate seasonal wave)

  - $r(\text{day\_of\_month}, \text{rate\_per\_mile}) \approx 0.02$ (no statistical signal)

### 5.2 Quote Signal Investigation & Coefficient of Determination ($R^2$)

Tested whether `quote_signal` represents an early quoted rate-per-mile:

Formula (plain text): Formula: textimplied_rate_i = textquote_signal_i times textdistance_i



Formula (plain text): Formula: R^2 = 1 - fracsum_i=1^n (textposted_rate_i - textimplied_rate_i)^2sum_i=1^n (textposted_rate_i - overlinetextposted_rate)^2 approx 0.81

* Exclusion Decision: While `quote_signal` explains $81%$ of rate variance, it is excluded from all production models because it is absent from `december-chart-inputs.csv`. Retaining it would cause catastrophic missing-feature failure during December inference.

### 5.3 Within-Month Residual De-trending Formula

To test if specific days of the month (e.g. 1st vs 15th vs 31st) have pricing premiums independent of the monthly macro-trend:

Formula (plain text): Formula: textresidual_i = textrate_per_mile_i - overlinetextrate_per_mile_textmonth(i)



Formula (plain text): Correlation(day_of_month, residual) = 0.02

* Exclusion Decision: `day_of_month` is excluded to prevent tree and spline models from memorizing spurious calendar noise.

## 6. Baseline Benchmark Model

### 6.1 Historical Lane-Average Lookup with Global Fallback

Formula (plain text): Formula: widehattextrpm_(p, d) = begincases

frac1|S_(p,d)| sum_i in S_(p,d) textrate_per_mile_i & textif  (p, d) in mathcalL_texttrain [8pt]

frac1N_texttrain sum_i=1^N_texttrain textrate_per_mile_i & textif  (p, d) notin mathcalL_texttrain

endcases



Formula (plain text): Formula: widehattextposted_rate_i = widehattextrpm_(p_i, d_i) times textdistance_i

* Evaluation: Scores $\text{MAE} > \$350+$ on the Sep–Oct holdout because $17%$ of validation lanes are unobserved in training, demonstrating why coordinate-based regression is mandatory.

## 7. Model Formulations & Ensembling

### 7.1 Explainable Boosting Machine (GA2M Model)

The production model utilizes a Generalized Additive Model with pairwise Interactions:

Formula (plain text): Formula: g(E[Y]) = beta_0 + sum_j=1^P f_j(x_j) + sum_j < k f_jk(x_j, x_k)

Where $g(\cdot)$ is the identity link for regression:

Formula (plain text): Formula: widehattextrate_per_mile = beta_0 + sum_j=1^P f_j(x_j) + sum_(j, k) in mathcalI f_jk(x_j, x_k)

### 7.2 Round-Robin Boosting Update Rule

Unlike decision trees where features compete for splits, EBM updates univariate shape functions sequentially in a cyclic round-robin loop:

Formula (plain text): Formula: f_j^(t+1)(x_j) leftarrow f_j^(t)(x_j) + eta cdot h_j^(t)(x_j)

Where $\eta = 0.01$ is the learning rate and $h_j$ is a shallow spline/tree fit to the current residual on feature $x_j$.

* Why this resolves December Extrapolation:

  In XGBoost, distance and equipment explain $76%$ of variance, starving date features of splits ($\le 5.8%$ split weight). In EBM, the cyclical date shape function $f_{\text{date}}(\text{doy\_sin}, \text{doy\_cos})$ is trained in its own dedicated round-robin step, guaranteeing a smooth, non-trivial seasonal curve.

### 7.3 Ensemble Blending Formula

Evaluated during Step 5 on the Sep–Oct holdout:

Formula (plain text): Formula: widehaty_textensemble, i = w_1 cdot widehaty_textXGB, i + w_2 cdot widehaty_textEBM, i, quad w_1 = w_2 = 0.50

### 7.4 Exploratory Post-Hoc Seasonal Scaling (Heuristic Baseline)

During early XGBoost debugging, a multiplicative seasonal adjustment was tested:

Formula (plain text): Formula: textidx_textseasonal(t) = fracbeta_0 + beta_1 sin(theta_t) + beta_2 cos(theta_t)overlinetextrate_per_mile_texttrain



Formula (plain text): Formula: widehattextrpm_textadjusted(t) = overlinewidehattextrpm_textXGB times fractextidx_textseasonal(t)overlinetextidx_textDec

*Rejected for production because it is an ad-hoc two-stage hack that degraded general validation accuracy.*

## 8. Evaluation Metrics & Error Diagnostics

### 8.1 Mean Absolute Error (MAE)

Formula (plain text): Formula: textMAE = frac1n sum_i=1^n |textposted_rate_i - widehattextposted_rate_i|

### 8.2 Mean Absolute Percentage Error (MAPE)

Formula (plain text): Formula: textMAPE = frac100%n sum_i=1^n left| fractextposted_rate_i - widehattextposted_rate_itextposted_rate_i right|

### 8.3 Overfitting Generalization Gap

Formula (plain text): Formula: Delta_textgen = textMAE_textholdout - textMAE_texttrain

* Role: Tracked across every candidate configuration in `train_tune.py` and `tune_ebm.py` to prevent selecting overfitted models with deceptively low training errors.

### 8.4 Price Range Stratification (Terciles)

For holdout error diagnosis in `error_analysis.py`:

Formula (plain text): Formula: q_1 = textquantile(y, 0.333), quad q_2 = textquantile(y, 0.667)



Formula (plain text): Formula: textTier(y_i) = begincases

textCheap & y_i le q_1

textMid-range & q_1 < y_i le q_2

textExpensive & y_i > q_2

endcases

## 9. Master Feature & Formula Decision Table

| Feature Name | Type / Transform | Mathematical Formula | Pipeline Status | Rationale |

|---|---|---|---|---|

| `posted_rate` | Target | $\text{rpm} = \text{rate} / \text{distance}$ | Transformed | Isolates unit pricing; eliminates length-of-haul variance |

| `distance` | Continuous | Used directly; verified via Haversine $\tau$ | Retained | Primary pricing driver; clean road distance |

| `date` | Cyclical | $\sin(2\pi \cdot \text{doy} / 365), \; \cos(2\pi \cdot \text{doy} / 365)$ | Retained | Continuous calendar cycle; enables winter extrapolation |

| `day_of_month` | Discrete | Excluded ($r_{\text{residual}} = 0.02$) | Dropped | No intra-month signal; prevents overfitting noise |

| `weight` | Continuous | $\text{abs}(w)$ then fillna with $\tilde{w}_{\text{train}}$ | Retained | Corrects sign-entry error; weak linear driver |

| `pickup_lat/lon` | Geographic | Continuous coordinates + K-Means ($k=6$) | Retained | Generalizes to unseen cities and novel lanes |

| `delivery_lat/lon` | Geographic | Continuous coordinates + K-Means ($k=6$) | Retained | Generalizes to unseen cities and novel lanes |

| `equipment` | Categorical | One-hot encoded across 3 fixed categories | Retained | Fixed schema across train, validation, and December |

| `quote_signal` | Continuous | Implied rate: $\text{quote} \times \text{distance}$ ($R^2 \approx 0.81$) | Dropped | Powerful signal, but missing from December file |

| `market_index` | Continuous | Excluded | Dropped | Missing from December input template |

| `pickup/delivery` | Categorical | High cardinality strings | Dropped | Replaced by continuous lat/lon coordinates |