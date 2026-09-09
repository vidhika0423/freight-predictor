# EDA & Feature Engineering 


## Master Feature & Formula Decision Table

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