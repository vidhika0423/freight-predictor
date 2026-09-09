# EDA & Feature Engineering — Mathematical & Formula Reference

This document provides a comprehensive, rigorous reference for every mathematical formula, statistical test, and feature transformation used across Exploratory Data Analysis (EDA), feature engineering, baseline modeling, ensemble experimentation, and final model evaluation for the Spotter.ai Freight Rate Prediction project.

---

## 1. Target Modeling & Reconstruction

Instead of predicting raw total dollars `posted_rate` directly, the modeling pipeline normalizes by trip distance.

### 1.1 Target Normalization (Rate per Mile)
$$\text{rate\_per\_mile}_i = \frac{\text{posted\_rate}_i}{\text{distance}_i}$$

### 1.2 Dollar Rate Reconstruction at Inference
$$\widehat{\text{posted\_rate}}_i = \widehat{\text{rate\_per\_mile}}_i \times \text{distance}_i$$

* **Rationale:** Haul distance alone accounts for $>70\%$ of raw dollar variance ($R^2 \approx 0.76$). Regressing directly on total dollars causes the learning algorithm to expend its capacity rediscovering "longer distance = higher dollar price." Normalizing isolates the unit pricing signal (market conditions, equipment type, lane demand, seasonality) and stabilizes residual variance across 70-mile hauls and 2,500-mile hauls.

---

## 2. Temporal & Cyclical Feature Engineering

Training data spans January through October. Raw sequential dates or linear day counts cannot be extrapolated by tree-based or spline models beyond October (they would predict late-October values flat across December).

### 2.1 Angular Day-of-Year Projection
$$\theta_i = \frac{2\pi \cdot \text{day\_of\_year}_i}{365}$$

### 2.2 Cyclical Sine / Cosine Decomposition
$$\text{doy\_sin}_i = \sin\left(\frac{2\pi \cdot \text{day\_of\_year}_i}{365}\right)$$
$$\text{doy\_cos}_i = \cos\left(\frac{2\pi \cdot \text{day\_of\_year}_i}{365}\right)$$

* **Continuous Cycle Boundary:** 
  $$\lim_{\text{date} \to \text{Dec 31}} (\text{doy\_sin}, \text{doy\_cos}) \approx (0, 1) = \text{Jan 1 coordinates}$$
  December coordinates $(\approx -0.49, 0.87)$ land directly adjacent in Euclidean feature space to January coordinates $(\approx 0.02, 1.00)$, allowing the model to naturally generalize winter freight dynamics learned in January to December without discontinuous extrapolation.

---

## 3. Data Cleaning & Imputation

### 3.1 Weight Sign Correction
$$w_{\text{clean}, i} = |w_i|$$

* **Diagnostic Check:** ~292 rows in training and 145 rows in validation contained negative weights (e.g., $-47,500\text{ lbs}$). Taking the absolute value preserved the correlation with rate ($r \approx 0.09$), confirming data-entry sign inversion rather than corrupted records.

### 3.2 Leak-Free Median Imputation
$$\tilde{w}_{\text{train}} = \text{median}\left(\{|w_i| : i \in \mathcal{D}_{\text{train}}, w_i \text{ is not null}\}\right)$$
$$w_{\text{final}, i} = \begin{cases} |w_i| & \text{if } w_i \neq \text{null} \\ \tilde{w}_{\text{train}} & \text{if } w_i = \text{null} \end{cases}$$

* **Data Isolation Rule:** $\tilde{w}_{\text{train}}$ is computed **strictly once** on training data and saved to `artifacts.pkl`. It is applied as a fixed constant to holdout, validation, and December datasets to prevent lookahead leakage.

### 3.3 Coordinate Fallback (Unseen Cities)
For rare rows missing coordinate metadata:
$$(\bar{\mu}_{\text{lat}}, \bar{\mu}_{\text{lon}}) = \left(\frac{1}{N_{\text{train}}} \sum_{i \in \mathcal{D}_{\text{train}}} \text{pickup\_lat}_i, \; \frac{1}{N_{\text{train}}} \sum_{i \in \mathcal{D}_{\text{train}}} \text{pickup\_lon}_i\right)$$

---

## 4. Spatial Engineering & Geographic Clustering

### 4.1 Great-Circle (Haversine) Distance Verification
Used to validate road distance integrity against physical straight-line distance:

$$a = \sin^2\left(\frac{\Delta \text{lat}}{2}\right) + \cos(\text{lat}_1) \cos(\text{lat}_2) \sin^2\left(\frac{\Delta \text{lon}}{2}\right)$$
$$d_{\text{haversine}} = 2 R \arcsin\left(\sqrt{a}\right), \quad R = 3958.8\text{ miles}$$

### 4.2 Circuity / Routing Tortuosity Ratio
$$\tau_i = \frac{\text{distance}_i}{d_{\text{haversine}, i}}$$

* **Finding:** $\tau$ sits consistently in the $[1.06, 1.20]$ band across the dataset, validating the `distance` column as authentic driving mileage. A small cluster of rows at $\text{distance} = 70$ with low $d_{\text{haversine}}$ represents standard commercial freight minimum billing floors, not corruptions.

### 4.3 K-Means Regional Clustering Objective
To group coordinates into coarse geographic market hubs:
$$\arg\min_{\mathcal{S}} \sum_{j=1}^k \sum_{x \in S_j} \|x - \boldsymbol{\mu}_j\|_2^2$$

### 4.4 Heuristic Derivation of Optimal Number of Clusters ($k$)
The dataset contains $48,000$ rows, but only $N_{\text{cities}} = 64$ unique geographic nodes. Applying the spatial rule of thumb:
$$k \approx \sqrt{\frac{N_{\text{cities}}}{2}} = \sqrt{\frac{64}{2}} = \sqrt{32} \approx 5.66 \implies k = 6$$

### 4.5 Nearest Centroid Assignment for Unseen Cities
Unseen coordinates $x^*$ are assigned deterministically without re-fitting:
$$\text{cluster}(x^*) = \arg\min_{j \in \{1,\dots,6\}} \|x^* - \boldsymbol{\mu}_j\|_2$$

---

## 5. Statistical Screening & Feature Selection

### 5.1 Pearson Correlation Coefficient
Used for pairwise linear association tests:
$$r_{xy} = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^n (y_i - \bar{y})^2}}$$

* **Key Findings:**
  - $r(\text{weight}, \text{rate\_per\_mile}) \approx 0.09$ (very weak)
  - $r(\text{day\_of\_year}, \text{daily\_avg\_rpm}) \approx 0.44$ (moderate seasonal wave)
  - $r(\text{day\_of\_month}, \text{rate\_per\_mile}) \approx 0.02$ (no statistical signal)

### 5.2 Quote Signal Investigation & Coefficient of Determination ($R^2$)
Tested whether `quote_signal` represents an early quoted rate-per-mile:
$$\text{implied\_rate}_i = \text{quote\_signal}_i \times \text{distance}_i$$
$$R^2 = 1 - \frac{\sum_{i=1}^n (\text{posted\_rate}_i - \text{implied\_rate}_i)^2}{\sum_{i=1}^n (\text{posted\_rate}_i - \overline{\text{posted\_rate}})^2} \approx 0.81$$

* **Exclusion Decision:** While `quote_signal` explains $81\%$ of rate variance, it is **excluded from all production models** because it is absent from `december-chart-inputs.csv`. Retaining it would cause catastrophic missing-feature failure during December inference.

### 5.3 Within-Month Residual De-trending Formula
To test if specific days of the month (e.g. 1st vs 15th vs 31st) have pricing premiums independent of the monthly macro-trend:
$$\text{residual}_i = \text{rate\_per\_mile}_i - \overline{\text{rate\_per\_mile}}_{\text{month}(i)}$$
$$r(\text{day\_of\_month}, \text{residual}) = 0.02$$

* **Exclusion Decision:** `day_of_month` is excluded to prevent tree and spline models from memorizing spurious calendar noise.

---

## 6. Baseline Benchmark Model

### 6.1 Historical Lane-Average Lookup with Global Fallback
$$\widehat{\text{rpm}}_{(p, d)} = \begin{cases} 
\frac{1}{|S_{(p,d)}|} \sum_{i \in S_{(p,d)}} \text{rate\_per\_mile}_i & \text{if } (p, d) \in \mathcal{L}_{\text{train}} \\[8pt]
\frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} \text{rate\_per\_mile}_i & \text{if } (p, d) \notin \mathcal{L}_{\text{train}} 
\end{cases}$$

$$\widehat{\text{posted\_rate}}_i = \widehat{\text{rpm}}_{(p_i, d_i)} \times \text{distance}_i$$

* **Evaluation:** Scores **$\text{MAE} > \$350+$** on the Sep–Oct holdout because $17\%$ of validation lanes are unobserved in training, demonstrating why coordinate-based regression is mandatory.

---

## 7. Model Formulations & Ensembling

### 7.1 Explainable Boosting Machine (GA2M Model)
The production model utilizes a Generalized Additive Model with pairwise Interactions:
$$g(E[Y]) = \beta_0 + \sum_{j=1}^{P} f_j(x_j) + \sum_{j < k} f_{jk}(x_j, x_k)$$

Where $g(\cdot)$ is the identity link for regression:
$$\widehat{\text{rate\_per\_mile}} = \beta_0 + \sum_{j=1}^{P} f_j(x_j) + \sum_{(j, k) \in \mathcal{I}} f_{jk}(x_j, x_k)$$

### 7.2 Round-Robin Boosting Update Rule
Unlike decision trees where features compete for splits, EBM updates univariate shape functions sequentially in a cyclic round-robin loop:
$$f_j^{(t+1)}(x_j) \leftarrow f_j^{(t)}(x_j) + \eta \cdot h_j^{(t)}(x_j)$$
Where $\eta = 0.01$ is the learning rate and $h_j$ is a shallow spline/tree fit to the current residual on feature $x_j$.

* **Why this resolves December Extrapolation:**
  In XGBoost, distance and equipment explain $76\%$ of variance, starving date features of splits ($\le 5.8\%$ split weight). In EBM, the cyclical date shape function $f_{\text{date}}(\text{doy\_sin}, \text{doy\_cos})$ is trained in its own dedicated round-robin step, guaranteeing a smooth, non-trivial seasonal curve.

### 7.3 Ensemble Blending Formula
Evaluated during Step 5 on the Sep–Oct holdout:
$$\widehat{y}_{\text{ensemble}, i} = w_1 \cdot \widehat{y}_{\text{XGB}, i} + w_2 \cdot \widehat{y}_{\text{EBM}, i}, \quad w_1 = w_2 = 0.50$$

### 7.4 Exploratory Post-Hoc Seasonal Scaling (Heuristic Baseline)
During early XGBoost debugging, a multiplicative seasonal adjustment was tested:
$$\text{idx}_{\text{seasonal}}(t) = \frac{\beta_0 + \beta_1 \sin(\theta_t) + \beta_2 \cos(\theta_t)}{\overline{\text{rate\_per\_mile}}_{\text{train}}}$$
$$\widehat{\text{rpm}}_{\text{adjusted}}(t) = \overline{\widehat{\text{rpm}}}_{\text{XGB}} \times \frac{\text{idx}_{\text{seasonal}}(t)}{\overline{\text{idx}}_{\text{Dec}}}$$
*Rejected for production because it is an ad-hoc two-stage hack that degraded general validation accuracy.*

---

## 8. Evaluation Metrics & Error Diagnostics

### 8.1 Mean Absolute Error (MAE)
$$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |\text{posted\_rate}_i - \widehat{\text{posted\_rate}}_i|$$

### 8.2 Mean Absolute Percentage Error (MAPE)
$$\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^n \left| \frac{\text{posted\_rate}_i - \widehat{\text{posted\_rate}}_i}{\text{posted\_rate}_i} \right|$$

### 8.3 Overfitting Generalization Gap
$$\Delta_{\text{gen}} = \text{MAE}_{\text{holdout}} - \text{MAE}_{\text{train}}$$

* **Role:** Tracked across every candidate configuration in `train_tune.py` and `tune_ebm.py` to prevent selecting overfitted models with deceptively low training errors.

### 8.4 Price Range Stratification (Terciles)
For holdout error diagnosis in `error_analysis.py`:
$$q_1 = \text{quantile}(y, 0.333), \quad q_2 = \text{quantile}(y, 0.667)$$
$$\text{Tier}(y_i) = \begin{cases} 
\text{Cheap} & y_i \le q_1 \\
\text{Mid-range} & q_1 < y_i \le q_2 \\
\text{Expensive} & y_i > q_2 
\end{cases}$$

---

## 9. Master Feature & Formula Decision Table

| Feature Name | Type / Transform | Mathematical Formula | Pipeline Status | Rationale |
|---|---|---|---|---|
| `posted_rate` | Target | $\text{rpm} = \text{rate} / \text{distance}$ | **Transformed** | Isolates unit pricing; eliminates length-of-haul variance |
| `distance` | Continuous | Used directly; verified via Haversine $\tau$ | **Retained** | Primary pricing driver; clean road distance |
| `date` | Cyclical | $\sin(2\pi \cdot \text{doy} / 365), \; \cos(2\pi \cdot \text{doy} / 365)$ | **Retained** | Continuous calendar cycle; enables winter extrapolation |
| `day_of_month` | Discrete | Excluded ($r_{\text{residual}} = 0.02$) | **Dropped** | No intra-month signal; prevents overfitting noise |
| `weight` | Continuous | $\text{abs}(w)$ then fillna with $\tilde{w}_{\text{train}}$ | **Retained** | Corrects sign-entry error; weak linear driver |
| `pickup_lat/lon` | Geographic | Continuous coordinates + K-Means ($k=6$) | **Retained** | Generalizes to unseen cities and novel lanes |
| `delivery_lat/lon` | Geographic | Continuous coordinates + K-Means ($k=6$) | **Retained** | Generalizes to unseen cities and novel lanes |
| `equipment` | Categorical | One-hot encoded across 3 fixed categories | **Retained** | Fixed schema across train, validation, and December |
| `quote_signal` | Continuous | Implied rate: $\text{quote} \times \text{distance}$ ($R^2 \approx 0.81$) | **Dropped** | Powerful signal, but missing from December file |
| `market_index` | Continuous | Excluded | **Dropped** | Missing from December input template |
| `pickup/delivery` | Categorical | High cardinality strings | **Dropped** | Replaced by continuous lat/lon coordinates |