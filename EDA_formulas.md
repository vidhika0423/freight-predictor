# EDA & Feature Engineering — Formula Reference

This document captures every formula used during exploratory data analysis and feature
engineering for the Spotter.ai freight rate prediction assessment, along with the
reasoning behind each one.

---

## 1. Target Transformation — Rate per Mile

Instead of predicting raw `posted_rate` directly, the model predicts price normalized
by distance. This isolates the *pricing* signal (market conditions, seasonality,
equipment type) from the dominant, mostly-linear effect of distance.

```
rate_per_mile = posted_rate / distance
```

To reconstruct the final submission value:

```
predicted_rate = predicted_rate_per_mile * distance
```

**Why:** `distance` explains most of the raw price by itself. Modeling `rate_per_mile`
lets the model focus its capacity on the more subtle drivers of price (date, equipment,
market conditions) rather than re-learning "longer distance = more money" over and over.

---

## 2. Date — Cyclical (sin/cos) Encoding

```
day_of_year = date.dayofyear                      # Jan 1 = 1, Dec 31 = 365

doy_sin = sin( 2 * pi * day_of_year / 365 )
doy_cos = cos( 2 * pi * day_of_year / 365 )
```

**Why:** Training data only covers January–October. Raw/linear date features
(e.g. day-of-year as a plain number, or "days since start") cannot be extrapolated
by tree-based models (XGBoost/CatBoost) beyond the range they were trained on — the
model would just repeat late-October behavior for all of December, producing a flat,
unrealistic trend line.

Cyclical encoding treats the calendar as a circle instead of a line. December's
sin/cos coordinates land close to January's (both winter months), letting the model
reuse learned winter patterns instead of extrapolating blindly off the edge of its
training range.

**Rejected alternative:** raw linear day-of-year / day count — confirmed to extrapolate
poorly for tree models past the training range.

---

## 3. Weight Cleanup

```
weight_clean = abs(weight)
```

Missing values (after cleanup) are filled with the **median** weight from training data.

**Why:** ~300 rows (train) / ~145 rows (validation) had negative weight values
(e.g. -36,559 lbs), which is physically impossible. Checked whether `abs(weight)`
correlates with price similarly to the positive-weight rows — it does (correlation
~0.09 either way), supporting a simple sign-error rather than a separate broken data
category. Since weight has very weak correlation with price overall, this is a
low-stakes fix — the goal is just to avoid feeding clearly-wrong values into the model.

---

## 4. Distance Sanity Check — Haversine (Great-Circle) Distance

Used only for validation of the `distance` column, not as a model feature itself.

```
a = sin²(Δlat / 2) + cos(lat1) * cos(lat2) * sin²(Δlon / 2)
haversine_distance = 2 * R * arcsin( sqrt(a) )        # R = 3958.8 miles (Earth's radius)

diff_ratio = distance / haversine_distance
```

**Why:** Confirms `distance` represents real road distance rather than corrupted data.
`diff_ratio` consistently fell between ~1.06–1.2, consistent with real driving routes
being longer than straight-line distance — this validated the column as trustworthy.
(One exception found: 48 rows fixed at `distance = 70` despite short straight-line
distance — consistent with a real-world minimum-distance billing floor, not an error.)

---

## 5. Quote Signal Investigation

Tested whether `quote_signal` is effectively an implied price-per-mile figure:

```
implied_rate = quote_signal * distance
```

Compared `implied_rate` against actual `posted_rate` using R² (coefficient of
determination):

```
R² = 1 - ( Σ(actual - predicted)² / Σ(actual - mean(actual))² )
```

**Result:** R² ≈ 0.81 — `quote_signal * distance` alone explains ~80% of price
variation.

**Why it matters:** This confirmed `quote_signal` is a legitimate, powerful business
feature (an early quoted price-per-mile) rather than random leakage computed backward
from the final price. However, it is **excluded from the final model** because it does
not exist in `december_chart_inputs.csv` — using it would create a model that silently
fails on the December prediction task.

---

## 6. Pearson Correlation (general-purpose relationship test)

Used throughout to test "is there a real linear relationship between X and Y?"

```
corr(X, Y) = Σ[(Xi - mean(X)) * (Yi - mean(Y))]
             ─────────────────────────────────────
             sqrt(Σ(Xi - mean(X))²) * sqrt(Σ(Yi - mean(Y))²)
```

Ranges from **-1 to +1**. Values near 0 indicate no meaningful linear relationship.

**Applied to:**
- `weight` vs `rate_per_mile` → ~0.03–0.09 (very weak, low-priority feature)
- `quote_signal` vs `posted_rate` (raw, whole dataset) → ~-0.04 (misleading on its own,
  see section 5 for the real relationship once distance is accounted for)
- `day_of_month` vs `rate_per_mile` → ~0.02 (no relationship, feature rejected)
- `day_of_year` vs daily avg `rate_per_mile` → ~0.44 (real, moderate seasonal signal)

---

## 7. Day-of-Month Check (Month Effect Removed)

To avoid a busy month (e.g. June) falsely making it look like "certain days of the
month" are pricier, the month-level average was subtracted out first:

```
residual = rate_per_mile - avg_rate_per_mile_for_that_row's_month
```

`day_of_month` was then correlated against `residual` instead of raw `rate_per_mile`,
isolating any *within-month* pattern separate from the *which-month* pattern.

**Result:** correlation ≈ 0.02, no meaningful within-month pattern found across the
data (values stayed in a tight 2.18–2.25 band for every day 1–31).

**Decision:** `day_of_month` is **excluded** as a feature. Since it carries no real
signal, including it risks the model latching onto coincidental noise (overfitting)
rather than adding predictive value.

---

## Summary of Feature Decisions Backed by This Analysis

| Feature | Decision | Backed by |
|---|---|---|
| `posted_rate` | Model `rate_per_mile` instead; reconstruct via `× distance` | Section 1 |
| `date` | Drop raw date; use `doy_sin` / `doy_cos` | Section 2 |
| `day_of_month` | Excluded — no signal found | Section 7 |
| `weight` | `abs()` + median-impute missing | Section 3 |
| `distance` | Used as-is, verified clean | Section 4 |
| `quote_signal`, `market_index` | Excluded from final model (real signal, but unavailable in December file) | Section 5 |
| `pickup_lat/lon`, `delivery_lat/lon` | Used directly (generalizes to unseen cities) | Cardinality check (see conversation) |