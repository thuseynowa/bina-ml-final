"""
data_prep.py — loading, cleaning, splitting, and the price-tier label.

All data wrangling lives here; the rest of the code just calls these
functions. Every cleaning decision is recorded in a `CleaningLog` so the
report can cite exact row counts.

Cleaning decisions (documented, rule-based — no statistics leaked from
val/test):
  1. Deduplicate: the dump contains the same listing scraped repeatedly
     (same `estate_rel_url`); we keep the latest scrape per listing.
  2. Scope: keep the three residential categories (`Yeni tikili`,
     `Köhnə tikili`, `Həyət evi/Bağ evi`). Offices, land plots, garages
     and commercial objects use different units (sot vs m²) and form a
     different market.
  3. Parse Azerbaijani string columns into numbers:
     `Sahə` ("145 m²") → area_m2, `Mərtəbə` ("7 / 9") → floor and
     building_floors, `Torpaq sahəsi` ("1.3 sot") → land_sot.
  4. Drop target-leakage columns: `unit_price` (= price / m²) and
     `total_price` (= price restated), plus identifiers and free text.
  5. Outliers via fixed, documented rules (not data-dependent quantiles):
     10k ≤ price ≤ 3M AZN, 15 ≤ area ≤ 800 m², rooms ≤ 20,
     building_floors ≤ 40, land ≤ 100 sot, coordinates inside the
     Azerbaijan bounding box (38.3–42.0 N, 44.5–51.0 E).
  6. Missing values are encoded with a constant + an explicit indicator
     column (no mean/median imputation → nothing to leak across splits).
"""

from __future__ import annotations

import os
import unicodedata

import numpy as np
import pandas as pd

# --- Config -----------------------------------------------------------------
SEED = 42
DATA_PATH = os.path.join("data", "bina_az_sale.csv")

# Columns that LEAK the target (derived from price). Dropped before training.
LEAKAGE_COLUMNS: list[str] = [
    "unit_price",    # price per m² — arithmetic function of the target
    "total_price",   # the sale price restated (identical to `price`)
]

# Identifiers / free text / scrape metadata — useless or unfair as features.
DROP_COLUMNS: list[str] = [
    "id_x", "id_y", "estate_id", "estate_details_id_x", "estate_details_id_y",
    "rel_url", "estate_rel_url_x", "estate_rel_url_y", "estate_rel_url",
    "img_url", "owner_name", "shop_name", "shop_title", "address",
    "description", "attributes", "extra_info", "city_when",
    "datetime_scrape_x", "datetime_scrape_y", "day_x", "day_y",
    "hour_x", "hour_y", "updated", "currency_x", "currency_y",
    "vip", "featured", "views",  # views/vip/featured are post-listing popularity, not property traits
    "repair",        # redundant duplicate of `Təmir`
    "bill_of_sale",  # redundant duplicate of `Çıxarış`
    "mortgage",      # redundant duplicate of `İpoteka`
    "Binanın növü",  # 99.8% missing
]

RESIDENTIAL_CATEGORIES = ["Yeni tikili", "Köhnə tikili", "Həyət evi/Bağ evi"]

# Fixed, documented outlier rules (AZN / m² / counts / degrees).
PRICE_MIN, PRICE_MAX = 10_000.0, 3_000_000.0
AREA_MIN, AREA_MAX = 15.0, 800.0
ROOMS_MAX = 20
FLOORS_MAX = 40
LAND_MAX_SOT = 100.0
LAT_MIN, LAT_MAX = 38.3, 42.0
LNG_MIN, LNG_MAX = 44.5, 51.0

# Baku city centre (Fountains Square) — for the distance feature.
BAKU_CENTER = (40.3725, 49.8390)

TOP_LOCATIONS = 30  # one-hot the 30 most frequent districts, rest → "other"


class CleaningLog(dict):
    """Ordered record of {step: rows_remaining} for the report."""

    def note(self, step: str, df: pd.DataFrame) -> None:
        self[step] = len(df)


def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV into a DataFrame (UTF-8). Return it unmodified."""
    return pd.read_csv(path, low_memory=False, encoding="utf-8")


# --- parsing helpers ---------------------------------------------------------
def _parse_area(s: pd.Series) -> pd.Series:
    """'145 m²' → 145.0 (handles thin spaces and comma decimals)."""
    txt = s.astype(str).str.replace(" ", " ", regex=False)
    num = txt.str.extract(r"([\d\s]+(?:[.,]\d+)?)", expand=False)
    num = num.str.replace(" ", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(num, errors="coerce")


def _parse_floor(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """'7 / 9' → (7, 9)."""
    parts = s.astype(str).str.extract(r"(\d+)\s*/\s*(\d+)")
    floor = pd.to_numeric(parts[0], errors="coerce")
    total = pd.to_numeric(parts[1], errors="coerce")
    return floor, total


def _norm_text(s: pd.Series) -> pd.Series:
    """NFC-normalise + strip, so 'Səbail r.' variants collapse."""
    return s.astype(str).map(lambda x: unicodedata.normalize("NFC", x).strip())


def clean(df: pd.DataFrame, log: CleaningLog | None = None) -> pd.DataFrame:
    """Apply the documented cleaning pipeline. Returns the cleaned frame."""
    log = log if log is not None else CleaningLog()
    log.note("raw rows", df)

    # 1. one row per listing: keep the latest scrape of each estate URL
    df = df.sort_values("datetime_scrape_y").drop_duplicates(
        subset=["estate_rel_url_x"], keep="last"
    )
    log.note("after deduplication (latest scrape per listing)", df)

    # 2. residential scope
    df = df[df["Kateqoriya"].isin(RESIDENTIAL_CATEGORIES)].copy()
    log.note("after keeping residential categories", df)

    # 3. parse numeric columns that arrive as Azerbaijani strings
    df["area_m2"] = _parse_area(df["Sahə"])
    floor, total_floors = _parse_floor(df["Mərtəbə"])
    df["floor"] = floor
    df["building_floors"] = total_floors
    df["land_sot"] = _parse_area(df["Torpaq sahəsi"])
    df["rooms"] = pd.to_numeric(df["Otaq sayı"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # categorical → canonical text
    for col in ("location", "city", "Kateqoriya", "Təmir", "Çıxarış",
                "İpoteka", "owner_title", "products_label"):
        df[col] = _norm_text(df[col])

    # 4. drop leakage + identifier columns
    df = df.drop(columns=[c for c in LEAKAGE_COLUMNS + DROP_COLUMNS
                          if c in df.columns])

    # 5. validity + outlier rules (fixed thresholds, documented above)
    df = df.dropna(subset=["price", "area_m2", "lat", "lng"])
    log.note("after dropping rows without price/area/coords", df)

    mask = (
        df["price"].between(PRICE_MIN, PRICE_MAX)
        & df["area_m2"].between(AREA_MIN, AREA_MAX)
        & df["lat"].between(LAT_MIN, LAT_MAX)
        & df["lng"].between(LNG_MIN, LNG_MAX)
        & (df["rooms"].isna() | (df["rooms"] <= ROOMS_MAX))
        & (df["building_floors"].isna() | (df["building_floors"] <= FLOORS_MAX))
        & (df["land_sot"].isna() | (df["land_sot"] <= LAND_MAX_SOT))
    )
    df = df[mask]
    log.note("after outlier rules", df)

    # exact duplicates of the modelling-relevant view (reposted listings)
    df = df.drop_duplicates(
        subset=["price", "area_m2", "rooms", "floor", "building_floors",
                "location", "Kateqoriya", "lat", "lng"]
    )
    log.note("after dropping exact repost duplicates", df)

    df.attrs["cleaning_log"] = dict(log)
    return df.reset_index(drop=True)


def _haversine_km(lat1, lng1, lat2, lng2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lng2) - np.radians(lng1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def make_features(df: pd.DataFrame):
    """
    Cleaned frame → numeric design matrix.

    Missing numerics get a constant 0 plus an explicit `*_missing`
    indicator (no statistics involved → leakage-free by construction).
    Categoricals are one-hot encoded; `location` keeps the TOP_LOCATIONS
    most frequent districts and folds the rest into 'other'.

    Returns (X, y, feature_names) with X float64, y = price in AZN.
    """
    out = pd.DataFrame(index=df.index)

    out["area_m2"] = df["area_m2"]
    out["rooms"] = df["rooms"].fillna(0)
    out["rooms_missing"] = df["rooms"].isna().astype(float)
    out["floor"] = df["floor"].fillna(0)
    out["building_floors"] = df["building_floors"].fillna(0)
    out["floor_missing"] = df["floor"].isna().astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = df["floor"] / df["building_floors"]
    out["floor_ratio"] = ratio.replace([np.inf, -np.inf], np.nan).fillna(0)
    out["is_top_floor"] = (df["floor"] == df["building_floors"]).astype(float)
    out["is_ground_floor"] = (df["floor"] == 1).astype(float)
    out["land_sot"] = df["land_sot"].fillna(0)
    out["has_land"] = df["land_sot"].notna().astype(float)

    out["lat"] = df["lat"]
    out["lng"] = df["lng"]
    out["dist_center_km"] = _haversine_km(
        df["lat"].values, df["lng"].values, BAKU_CENTER[0], BAKU_CENTER[1]
    )

    out["is_repaired"] = (df["Təmir"] == "var").astype(float)
    out["repair_missing"] = df["Təmir"].isin(["nan", ""]).astype(float)
    out["has_bill_of_sale"] = (df["Çıxarış"] == "var").astype(float)
    out["mortgage_available"] = (df["İpoteka"] == "var").astype(float)
    out["owner_is_agent"] = (df["owner_title"] == "vasitəçi (agent)").astype(float)
    out["listed_by_agency"] = (df["products_label"] == "Agentlik").astype(float)
    out["is_baku"] = (df["city"] == "bakı").astype(float)

    # building category (3 values after cleaning scope)
    for cat in RESIDENTIAL_CATEGORIES:
        safe = (cat.lower()
                .replace(" ", "_").replace("/", "_")
                .replace("ə", "e").replace("ö", "o").replace("ı", "i"))
        out[f"cat_{safe}"] = (df["Kateqoriya"] == cat).astype(float)

    # district: top-K one-hot + 'other'
    top = df["location"].value_counts().head(TOP_LOCATIONS).index.tolist()
    for loc in top:
        safe = (unicodedata.normalize("NFD", loc).encode("ascii", "ignore")
                .decode().lower().replace(" ", "_").replace(".", ""))
        out[f"loc_{safe or 'unk'}"] = (df["location"] == loc).astype(float)
    out["loc_other"] = (~df["location"].isin(top)).astype(float)

    X = out.to_numpy(dtype=np.float64)
    y = df["price"].to_numpy(dtype=np.float64)
    return X, y, list(out.columns)


def make_tier_label(y_price, threshold: float | None = None):
    """
    Price TIER: premium (1) if price > threshold else standard (0).
    The threshold MUST be the median of the TRAINING prices only — pass
    threshold=None with the training y once, then reuse the returned
    threshold for val/test.
    """
    y_price = np.asarray(y_price, dtype=float)
    if threshold is None:
        threshold = float(np.median(y_price))
    return (y_price > threshold).astype(int), threshold


def train_val_test_split(X, y, val_size=0.15, test_size=0.15, seed=SEED):
    """
    Deterministic shuffled split into train / validation / test
    (70/15/15 by default). Returns (X_tr, y_tr, X_val, y_val, X_te, y_te,
    idx_tr, idx_val, idx_te) — the indices let callers map rows back to
    the cleaned frame for error analysis.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(X)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = int(round(n * test_size))
    n_val = int(round(n * val_size))
    idx_te = perm[:n_test]
    idx_val = perm[n_test:n_test + n_val]
    idx_tr = perm[n_test + n_val:]
    return (X[idx_tr], y[idx_tr], X[idx_val], y[idx_val], X[idx_te], y[idx_te],
            idx_tr, idx_val, idx_te)


def standardize(X_tr, *others):
    """
    Standardize using TRAIN statistics only. Returns (X_tr_s, *others_s,
    mean, std). Constant columns get std=1 so they map to 0.
    """
    X_tr = np.asarray(X_tr, dtype=float)
    mean = X_tr.mean(axis=0)
    std = X_tr.std(axis=0)
    std[std == 0] = 1.0
    scaled = [(np.asarray(o, dtype=float) - mean) / std for o in others]
    return ((X_tr - mean) / std, *scaled, mean, std)
