"""
Pure data-processing pipeline for the expansion-decision dashboard. No Streamlit dependency,
so this can be (and is) tested directly with plain Python before being wrapped with caching in app.py.
Mirrors the validated methodology from the companion analysis notebook.
"""

import io
import json
import gzip
import base64
import zipfile
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

RANDOM_STATE = 42

WEIGHT_PRESETS = {
    "Balanced":      {"scale_n": 0.34, "growth_n": 0.33, "fit_n": 0.33},
    "Growth-tilted": {"scale_n": 0.20, "growth_n": 0.50, "fit_n": 0.30},
    "Scale-tilted":  {"scale_n": 0.50, "growth_n": 0.20, "fit_n": 0.30},
    "Fit-tilted":    {"scale_n": 0.25, "growth_n": 0.25, "fit_n": 0.50},
}


def load_and_clean(file_bytes: bytes, filename: str):
    if filename.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            members = [m for m in zf.namelist() if m.lower().endswith(".csv") and "metadata" not in m.lower()]
            if not members:
                raise ValueError("No data CSV found inside the uploaded zip.")
            with zf.open(members[0]) as f:
                raw = pd.read_csv(f)
    else:
        raw = pd.read_csv(io.BytesIO(file_bytes))

    df = raw.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"startups_ recognized": "startups_recognized"})
    if "startups_recognized" not in df.columns:
        raise ValueError("Expected a 'startups_recognized' column (after cleaning) -- is this the right file?")

    if df.duplicated(subset=["year", "state", "industry"]).any():
        raise ValueError("Duplicate (year, state, industry) rows found -- unexpected data shape.")

    provisional_years = sorted(df.loc[df["note"].notna(), "year"].unique().tolist()) if "note" in df.columns else []
    all_years = sorted(df["year"].unique().tolist())
    stable_years = [y for y in all_years if y not in provisional_years]

    df = df[["year", "state", "industry", "startups_recognized"]].copy()
    df["state"] = df["state"].str.strip()
    df["industry"] = df["industry"].str.strip()

    growth_start_year = stable_years[0] + 1 if len(stable_years) > 1 else stable_years[0]
    growth_years = [y for y in stable_years if y >= growth_start_year]

    return df, stable_years, provisional_years, growth_years


def get_geojson(path="india_states_map.b64"):
    try:
        with open(path) as f:
            b64 = f.read().strip()
        return json.loads(gzip.decompress(base64.b64decode(b64)))
    except Exception:
        return None


def compute_eligibility(df: pd.DataFrame, stable_years: list, min_threshold: int):
    stable = df[df["year"].isin(stable_years)].copy()
    state_totals = (stable.groupby("state")["startups_recognized"].sum()
                     .sort_values(ascending=False).reset_index())
    state_totals.columns = ["state", "total_startups"]
    eligible_states = set(state_totals.loc[state_totals["total_startups"] >= min_threshold, "state"])
    return stable, state_totals, eligible_states


def compute_growth(stable: pd.DataFrame, state_totals: pd.DataFrame, eligible_states: set,
                    growth_years: list, n_bootstrap: int = 300):
    year_state = (stable.pivot_table(index="state", columns="year", values="startups_recognized",
                                      aggfunc="sum", fill_value=0)
                  .reindex(columns=growth_years, fill_value=0))

    def trend_fit(row):
        y = np.log1p(np.asarray(row, dtype=float))
        x = np.arange(len(y))
        if y.std() == 0:
            return pd.Series({"trend_growth_pct": 0.0, "r2": np.nan})
        slope, intercept = np.polyfit(x, y, 1)
        pred = slope * x + intercept
        ss_res, ss_tot = ((y - pred) ** 2).sum(), ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return pd.Series({"trend_growth_pct": (np.exp(slope) - 1) * 100, "r2": r2})

    growth_df = year_state.apply(trend_fit, axis=1)

    elig = [s for s in growth_df.index if s in eligible_states]
    elig_growth = growth_df.loc[elig, "trend_growth_pct"]
    elig_totals = state_totals.set_index("state").loc[elig, "total_startups"]
    log_scale = np.log1p(elig_totals)
    slope, intercept = np.polyfit(log_scale, elig_growth, 1)
    predicted = slope * log_scale + intercept
    growth_df.loc[elig, "growth_resid"] = elig_growth - predicted
    growth_df.loc[elig, "growth_predicted"] = predicted

    rng = np.random.default_rng(RANDOM_STATE)

    def bootstrap_ci(counts):
        y = np.log1p(np.asarray(counts, dtype=float))
        n = len(y)
        x = np.arange(n)
        slopes = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            if len(set(idx)) < 2:
                continue
            s, _ = np.polyfit(x[idx], y[idx], 1)
            slopes.append((np.exp(s) - 1) * 100)
        return pd.Series(np.percentile(slopes, [5, 95]), index=["growth_p5", "growth_p95"])

    ci = year_state.loc[elig].apply(bootstrap_ci, axis=1)
    growth_df = growth_df.join(ci)
    return growth_df


def compute_lq(stable: pd.DataFrame):
    state_industry = stable.pivot_table(index="state", columns="industry", values="startups_recognized",
                                         aggfunc="sum", fill_value=0)
    state_share = state_industry.div(state_industry.sum(axis=1), axis=0)
    national_share = state_industry.sum(axis=0) / state_industry.values.sum()
    lq = state_share.div(national_share, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0)
    return lq, state_share


def compute_sector_growth(stable: pd.DataFrame, eligible_states: set, growth_years: list, target_sector: str):
    sub = stable[(stable["industry"] == target_sector) & (stable["state"].isin(eligible_states))]
    pivot = (sub.pivot_table(index="state", columns="year", values="startups_recognized", aggfunc="sum", fill_value=0)
             .reindex(columns=growth_years, fill_value=0))

    def trend(row):
        y = np.log1p(np.asarray(row, dtype=float))
        if y.std() == 0:
            return 0.0
        x = np.arange(len(y))
        slope, _ = np.polyfit(x, y, 1)
        return (np.exp(slope) - 1) * 100

    return pivot.apply(trend, axis=1)


def compute_clusters(state_totals: pd.DataFrame, growth_df: pd.DataFrame, state_share: pd.DataFrame,
                      eligible_states: set):
    def shannon_diversity(p):
        p = p[p > 0]
        return float(-(p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 0.0

    diversity = state_share.apply(shannon_diversity, axis=1).rename("diversity")
    features = pd.DataFrame({
        "log_scale": np.log1p(state_totals.set_index("state")["total_startups"]),
        "growth": growth_df["trend_growth_pct"],
        "diversity": diversity,
    }).dropna()
    features.index.name = "state"
    features = features.loc[features.index.isin(eligible_states)]

    X = StandardScaler().fit_transform(features)
    best_k, best_score = 2, -1
    for k in range(2, min(7, len(features))):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X)
        sc = silhouette_score(X, km.labels_)
        if sc > best_score:
            best_k, best_score = k, sc

    km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10).fit(X)
    features["cluster"] = km.labels_
    centroid = features.groupby("cluster")[["log_scale", "growth"]].mean()

    def safe_z(s):
        std = s.std(ddof=0)
        return (s - s.mean()) / std if std > 0 else s * 0

    sz, gz = safe_z(centroid["log_scale"]), safe_z(centroid["growth"])

    def label(s, g):
        if s >= 0 and g >= 0: return "Larger & Faster-Growing"
        if s >= 0 and g < 0:  return "Larger & Slower-Growing"
        if s < 0 and g >= 0:  return "Smaller & Faster-Growing"
        return "Smaller & Slower-Growing"

    raw = {c: label(sz[c], gz[c]) for c in centroid.index}
    counts, seen, labels_map = Counter(raw.values()), Counter(), {}
    for c, lab in raw.items():
        if counts[lab] > 1:
            seen[lab] += 1
            labels_map[c] = f"{lab} ({seen[lab]})"
        else:
            labels_map[c] = lab
    features["cluster_label"] = features["cluster"].map(labels_map)
    return features


def compute_scores(features: pd.DataFrame, growth_df: pd.DataFrame, lq: pd.DataFrame,
                    sector_growth: pd.Series, state_totals: pd.DataFrame, stable: pd.DataFrame,
                    recent_years: list, target_sector: str, weight_scenarios: dict = WEIGHT_PRESETS):
    recent_scale = (stable[stable["year"].isin(recent_years)].groupby("state")["startups_recognized"].sum()
                    .rename("recent_scale"))

    sector_growth_aligned = sector_growth.reindex(features.index)
    fallback = sector_growth_aligned.min() - 1 if sector_growth_aligned.notna().any() else -100
    sector_growth_aligned = sector_growth_aligned.fillna(fallback)
    fit_combined = (0.5 * lq[target_sector].reindex(features.index).rank(pct=True)
                     + 0.5 * sector_growth_aligned.rank(pct=True))

    score_inputs = features.copy()
    score_inputs["recent_scale"] = recent_scale
    score_inputs["log_recent_scale"] = np.log1p(score_inputs["recent_scale"])
    score_inputs["growth_for_score"] = growth_df["growth_resid"]
    score_inputs["fit_combined"] = fit_combined
    score_inputs = score_inputs.dropna(subset=["log_recent_scale", "growth_for_score", "fit_combined"])

    norm = pd.DataFrame({
        "scale_n": score_inputs["log_recent_scale"].rank(pct=True),
        "growth_n": score_inputs["growth_for_score"].rank(pct=True),
        "fit_n": score_inputs["fit_combined"].rank(pct=True),
    }, index=score_inputs.index)

    out = pd.DataFrame(index=norm.index)
    for name, w in weight_scenarios.items():
        out[name] = (norm["scale_n"].clip(lower=0.01) ** w["scale_n"]
                      * norm["growth_n"].clip(lower=0.01) ** w["growth_n"]
                      * norm["fit_n"].clip(lower=0.01) ** w["fit_n"]) * 100
    out["avg_score"] = out[list(weight_scenarios.keys())].mean(axis=1)
    return out.sort_values("avg_score", ascending=False), norm, score_inputs


def score_with_weights(norm: pd.DataFrame, weights: dict):
    """Composite score under one specific weight vector (e.g., a manager's custom weighting),
    as opposed to compute_scores' fixed 4-preset robustness ensemble."""
    return (norm["scale_n"].clip(lower=0.01) ** weights["scale_n"]
            * norm["growth_n"].clip(lower=0.01) ** weights["growth_n"]
            * norm["fit_n"].clip(lower=0.01) ** weights["fit_n"]) * 100


def monte_carlo_stability(norm: pd.DataFrame, home_state: str, top_n: int, n_sims: int = 500):
    rng = np.random.default_rng(RANDOM_STATE)
    counts = pd.Series(0, index=norm.index)
    for _ in range(n_sims):
        w = rng.dirichlet([1, 1, 1])
        sim = (norm["scale_n"].clip(lower=0.01) ** w[0] * norm["growth_n"].clip(lower=0.01) ** w[1]
               * norm["fit_n"].clip(lower=0.01) ** w[2])
        top = sim.drop(index=home_state, errors="ignore").nlargest(top_n).index
        counts.loc[top] += 1
    return (counts / n_sims * 100).sort_values(ascending=False).rename("pct_in_top")
