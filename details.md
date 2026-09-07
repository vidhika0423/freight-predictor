- FEATURE ENGINEERING
1. distance 
-Real driving distance vs straight-line distance checks out. I compared the given distance column against the actual great-circle (straight-line) distance calculated from lat/lon. The ratio sits consistently around 1.06–1.2x the straight-line distance — which is exactly what you'd expect in real life, since trucks drive on roads, not in straight lines.
-8 rows have distance fixed at exactly 70, even though their real straight-line distance is much shorter (as low as ~7 miles). This isn't a data error — it looks like a minimum-distance floor, which is completely normal in real freight pricing
2. pickup_lat/lon, delivery_lat/lon
-use directly (generalizes to unseen cities)
-training data has 64 unique cities forming 4,014 unique lanes
-Most lanes appear 6–16 times, but 67 lanes appear only once
-validation.csv contains 8 cities that never appear anywhere in your training data — Norfolk, Allentown, Jackson, Chicago, Charlotte, Laredo, Knoxville, San Diego. And 736 of the 4,214 lanes in validation (17%) were never seen in training at all.
-encoding that treats city names as a fixed lookup table will simply have no answer for those new cities
-every city has one fixed lat/lon coordinate, have a numeric backbone that works for any city, seen or not.
-Add a coarse "region" feature derived from lat/lon — cluster all cities into a handful of geographic regions (e.g., 5–6 clusters using something like k-means on lat/lon). A brand-new city just gets assigned to whichever region its coordinates are closest to
-Skip using raw city name or lane (pickup+delivery combo) as a plain categorical/target-encoded feature
   
4. weight
-Missing values: 300 missing in train, 165 in validation, 0 in December.
-negative weights: 292 rows in training (and 145 in validation) : {-47,500 lbs} :clearly bad data, not a real signal
-correlates with price anyway (correlation ≈ 0.03–0.07) with original data 
-correlation is still weak (~0.09) with abs(weight) data 
-but taking abs() is more defensible than deleting ~600 rows of otherwise-fine data, and safer than leaving negative numbers in
5. date
-price-per-mile has a mild seasonal wave : rises from about 2.06 in January up to a peak around 2.35–2.40 in June, then eases back down and plateaus around 2.20–2.30 through August–October. That's roughly a 15% swing across the year — real, but not huge.
-Day-of-week has almost no effect (2.19 vs 2.25 — basically flat). Not worth much engineering effort there.
-tree-based models (XGBoost/CatBoost, which we're planning to use): they cannot extrapolate.
-for December it will just repeat whatever it learned for late October — meaning your December chart would come out as a flat, boring line, showing no seasonality at all.
-cyclical date encoding: Instead of feeding the model a raw increasing number for date, encode day-of-year (or month) as a cycle — using sine and cosine of the date, so the calendar wraps around like a clock instead of a straight line.
-Why this actually solves the problem: December is, seasonally, right next door to January — both are winter months. With cyclical encoding, December's sin/cos values land very close to January's values in feature space — values the model has already seen and learned from. So instead of the model going "I've never seen this, I'll just repeat October," it effectively goes "this looks like winter, similar to January" — which is a much more sensible guess, and it's also just true in the real world.
-not holiday-specific features (like "is it near Thanksgiving/Christmas") — you have zero November/December historical data to confirm those effects actually exist in this dataset.
-grouped every row in the training data by "which day of the month it fell on" (1st, 2nd, 3rd... up to 31st) and looked at the average price-per-mile for each.
Prices across every day of the month sit in a tight band between about 2.18 and 2.25 — no meaningful highs or lows tied to a specific day. The statistical correlation between day-of-month and price came out to 0.02, which is close enough to zero to call "no relationship."
-
6. market_index, quote_signal: exclude from final model (not available in December file)
7. target: rate_per_mile = posted_rate / distance, multiply back by distance for final prediction


- SPLIT STRATEGY 
time-based split, not random

We already established this earlier (no shuffling), but here's the concrete reasoning restated simply: your real task is to predict November (validation.csv) and December (the chart) — both are future months your model has never trained on. So the fairest way to test yourself is to recreate that exact situation during development: hide the most recent months, train only on what came before, and see how well you predict the "future" you already secretly know the answer to.

A random shuffle-split would let the model "practice" on some September rows while also training on other September rows from the same month — that's easier than the real task, so it would make your validation score look better than it'll actually perform in reality.
It lands almost exactly on a clean 80/20 split, which is a well-understood, standard ratio — nothing exotic to justify.
Using two full months (Sept + Oct) as holdout, rather than just one, gives a big enough sample (9,523 rows) to trust the validation numbers — not just a handful of rows where luck could swing the score.
More importantly: this setup mimics the real task closely. October is your last known month, and you're predicting November next (1 month ahead) and December after that (2 months ahead). Holding out Sept–Oct and predicting them from Jan–Aug is structurally the same kind of stretch — "predict 1–2 months past what you've seen" — so your holdout score should be a realistic preview of how you'll do on the real November/December tasks.

- MODEL
XGBoost, predicting rate_per_mile, on the cleaned feature set, trained Jan–Aug / validated Sep–Oct, then retrained on full Jan–Oct for final predictions
1. Baseline (mean-per-lane) — build first, takes 5 minutes, gives you a floor to beat.
2. Gradient Boosting (LightGBM or XGBoost) — this is your main model. Best fit for this data shape, industry-standard choice, handles nonlinear interactions well.
3. Linear Regression — keep as a secondary reference, and specifically re-check it against the December chart output as a sanity check on extrapolation behavior.
4. Depth-wise growth (XGBoost) is more restrained — because it builds evenly across the whole tree instead of chasing the single best split, it's naturally less likely to build an ultra-specific branch just to capture one quirky group of rows.  67 lanes appear only once in the entire training set.
17% of validation lanes were never seen in training at all.
If a lane only shows up once, its price is basically a coin flip — it could be a completely normal price, or it could be a one-off fluke (maybe that specific day had unusual conditions). A model that's too eager to build a very specific, deep branch just for that one rare lane will end up "memorizing" that single data point as if it were a real, generalizable pattern. That's overfitting — it looks great on training data, but falls apart the moment it sees a genuinely new situation, which is exactly what 17% of your validation set is.

k in kmeans
You plugged in n = 12,000 (the size of validation.csv). But we're not clustering 12,000 individual loads — we're clustering geographic locations (cities), and a city's lat/lon repeats every time that city shows up as a pickup or delivery point. We only have 64 unique cities total. The other ~47,900+ rows are just the same 64 coordinate points repeated over and over.

What happens if you actually use the right n

If we plug in the real number of distinct things being clustered:


𝐾
≈
64
2
=
32
≈
5.7
≈
6
K≈
2
64
	​

	​

=
32
	​

≈5.7≈6

That lines up almost exactly with the 6 clusters we already chose in the code. So this heuristic, applied correctly, actually confirms our earlier decision rather than contradicting it — nice coincidence to double check, but not a reason to change anything.


(venv) C:\Users\HP\Documents\projects\fright_rate_preditor\spotter_project>python src/clean_features.py
[train-test.csv] OK — shape=(48000, 12), no NaNs, columns match schema.
   distance  pickup_lat  pickup_lon  ...   weight   doy_sin   doy_cos
0     274.3    38.09122   -76.78906  ...  30658.0  0.017213  0.999852
1     280.5    38.09122   -76.78906  ...  17555.0  0.017213  0.999852
2     967.8    39.22317   -72.96710  ...  31721.0  0.017213  0.999852

[3 rows x 12 columns]

[validation.csv] OK — shape=(12000, 12), no NaNs, columns match schema.
[december-chart-inputs.csv] OK — shape=(31, 12), no NaNs, columns match schema.
   distance  pickup_lat  pickup_lon  ...  weight   doy_sin   doy_cos
0       360    36.99152   -84.99876  ...   32000 -0.493776  0.869589
1       360    36.99152   -84.99876  ...   32000 -0.478734  0.877960
2       360    36.99152   -84.99876  ...   32000 -0.463550  0.886071

[3 rows x 12 columns]
