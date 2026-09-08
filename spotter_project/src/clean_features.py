"""
Shared data cleaning + feature engineering module 

Design principle: every file (training data, holdout split, validation.csv,
december_chart_inputs.csv) MUST go through the exact same transformations, using
values (weight median, cluster centers, city lat/lon lookup) that were learned ONCE
from training data and reused everywhere else.

"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

EQUIPMENT_CATEGORIES = ["Dry Van", "Reefer", "Flatbed"]
N_REGION_CLUSTERS = 6
RANDOM_STATE = 42

# Final feature columns fed into the model, in a fixed order.
FEATURE_COLUMNS = [
    "distance",
    "pickup_lat", "pickup_lon",
    "delivery_lat", "delivery_lon",
    "region_cluster",
    "equipment_Dry Van", "equipment_Reefer", "equipment_Flatbed",
    "weight",
    "doy_sin", "doy_cos",
]


def _build_city_lookup(df: pd.DataFrame) -> dict:
    """
    Build a city name -> (lat, lon) lookup from any file that has coordinate columns.
    """
    lookup = {}
    if {"pickup", "pickup_lat", "pickup_lon"}.issubset(df.columns):
        for city, lat, lon in df[["pickup", "pickup_lat", "pickup_lon"]].dropna().itertuples(index=False):
            lookup.setdefault(city, (lat, lon))
    if {"delivery", "delivery_lat", "delivery_lon"}.issubset(df.columns):
        for city, lat, lon in df[["delivery", "delivery_lat", "delivery_lon"]].dropna().itertuples(index=False):
            lookup.setdefault(city, (lat, lon))
    return lookup


def _fill_missing_coords(df: pd.DataFrame, city_lookup: dict, fallback_latlon: tuple) -> pd.DataFrame:
    """
    Ensures pickup_lat/lon and delivery_lat/lon exist and are filled, using the city
    lookup table when a file doesn't include coordinate columns at all. 
    Falls back to the overall average lat/lon for any city 
    that's genuinely never been seen before
    """
    df = df.copy()

    for role in ["pickup", "delivery"]:
        lat_col, lon_col = f"{role}_lat", f"{role}_lon"

        if lat_col not in df.columns or lon_col not in df.columns:
            # File has no coordinate columns at all 
            df[lat_col] = df[role].map(lambda c: city_lookup.get(c, fallback_latlon)[0])
            df[lon_col] = df[role].map(lambda c: city_lookup.get(c, fallback_latlon)[1])
        else:
            # File has the columns but maybe some rows are missing values.
            missing = df[lat_col].isna() | df[lon_col].isna()
            if missing.any():
                filled_lat = df.loc[missing, role].map(lambda c: city_lookup.get(c, fallback_latlon)[0])
                filled_lon = df.loc[missing, role].map(lambda c: city_lookup.get(c, fallback_latlon)[1])
                df.loc[missing, lat_col] = filled_lat
                df.loc[missing, lon_col] = filled_lon

    return df


def _add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dates = pd.to_datetime(df["date"])
    day_of_year = dates.dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365)
    df["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365)
    return df


def _clean_weight(df: pd.DataFrame, weight_median: float) -> pd.DataFrame:
    df = df.copy()
    df["weight"] = df["weight"].abs()
    df["weight"] = df["weight"].fillna(weight_median)
    return df


def _one_hot_equipment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for cat in EQUIPMENT_CATEGORIES:
        df[f"equipment_{cat}"] = (df["equipment"] == cat).astype(int)
    return df


def fit_transform(train_df: pd.DataFrame):
    """
    Learns all reusable artifacts (weight median, city lookup, cluster model) from
    training data, applies the full feature pipeline, and returns both the
    engineered features + target, and the artifacts needed to transform other files
    consistently.
    """
    df = train_df.copy()

    #learn artifacts 
    weight_median = df["weight"].abs().median()
    city_lookup = _build_city_lookup(df)
    fallback_latlon = (df["pickup_lat"].mean(), df["pickup_lon"].mean())

    df = _clean_weight(df, weight_median)
    df = _fill_missing_coords(df, city_lookup, fallback_latlon)
    df = _add_date_features(df)
    df = _one_hot_equipment(df)

    # fit region clustering on training coordinates
    coords = pd.concat([
        df[["pickup_lat", "pickup_lon"]].rename(columns={"pickup_lat": "lat", "pickup_lon": "lon"}),
        df[["delivery_lat", "delivery_lon"]].rename(columns={"delivery_lat": "lat", "delivery_lon": "lon"}),
    ], ignore_index=True)
    cluster_model = KMeans(n_clusters=N_REGION_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    cluster_model.fit(coords.to_numpy())

    df["region_cluster"] = cluster_model.predict(df[["pickup_lat", "pickup_lon"]].to_numpy())

    # target 
    df["rate_per_mile"] = df["posted_rate"] / df["distance"]

    artifacts = {
        "weight_median": weight_median,
        "city_lookup": city_lookup,
        "fallback_latlon": fallback_latlon,
        "cluster_model": cluster_model,
    }

    features = df[FEATURE_COLUMNS].copy()
    target = df["rate_per_mile"].copy()
    meta = df[["load_id", "distance"]].copy() if "load_id" in df.columns else df[["distance"]].copy()

    return features, target, meta, artifacts


def transform(df: pd.DataFrame, artifacts: dict):
    """
    Applies the already-learned artifacts to a new file (holdout split, validation,
    or december chart inputs). 
    """
    df = df.copy()

    df = _clean_weight(df, artifacts["weight_median"])
    df = _fill_missing_coords(df, artifacts["city_lookup"], artifacts["fallback_latlon"])
    df = _add_date_features(df)
    df = _one_hot_equipment(df)
    df["region_cluster"] = artifacts["cluster_model"].predict(df[["pickup_lat", "pickup_lon"]].to_numpy())

    features = df[FEATURE_COLUMNS].copy()

    meta_cols = [c for c in ["load_id", "distance", "date"] if c in df.columns]
    meta = df[meta_cols].copy()

    target = None
    if "posted_rate" in df.columns:
        target = df["posted_rate"] / df["distance"]

    return features, target, meta


def _sanity_check(features: pd.DataFrame, name: str):
    """Basic assertions run after transforming any file."""
    assert features.isna().sum().sum() == 0, f"[{name}] NaNs remain in features!"
    assert (features["distance"] >= 0).all(), f"[{name}] negative distance found!"
    assert (features["weight"] >= 0).all(), f"[{name}] negative weight found!"
    assert list(features.columns) == FEATURE_COLUMNS, f"[{name}] column mismatch!"
    print(f"[{name}] OK — shape={features.shape}, no NaNs, columns match schema.")


if __name__ == "__main__":
    # Quick self-test
    train_raw = pd.read_csv("data/train-test.csv")
    val_raw = pd.read_csv("data/validation.csv")
    dec_raw = pd.read_csv("data/december-chart-inputs.csv")

    train_features, train_target, train_meta, artifacts = fit_transform(train_raw)
    _sanity_check(train_features, "train-test.csv")
    print(train_features.head(3))
    print()

    val_features, val_target, val_meta = transform(val_raw, artifacts)
    _sanity_check(val_features, "validation.csv")

    dec_features, dec_target, dec_meta = transform(dec_raw, artifacts)
    _sanity_check(dec_features, "december-chart-inputs.csv")
    print(dec_features.head(3))