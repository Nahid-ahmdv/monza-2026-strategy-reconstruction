"""Race-pace estimation from FP2 and qualifying timing, following Ruth's stated method.

Ruth's article defines three quantities:

  FP2 long run   'Average of clean laps in each driver's best long run in FP2, as a gap to
                  the fastest ... Clean laps are within two per cent of the stint median;
                  out laps and in laps removed', stints of five clean laps or more.
  Friday-weighted  'Half the FP2 long run ... and half qualifying', both as gaps to the best.
                  Drivers without a long run use qualifying only.
  Saturday-rebased 'Take Saturday's qualifying gap and add only each car's own Friday
                  quali-to-race drop-off relative to the field', where the drop-off is the
                  FP2 long-run average minus the FP2 push lap.

It also provides a fuel-corrected tyre-degradation estimate for a single long run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def t975(df: int) -> float:
    """97.5 % quantile of Student's t with `df` degrees of freedom (two-sided 95 % intervals), from SciPy."""
    return float(stats.t.ppf(0.975, df))


def clean_laps(g: pd.DataFrame, tolerance: float = 0.02, symmetric: bool = True,
               green_only: bool = True) -> pd.DataFrame:
    """Clean laps of one stint: no out/in laps, (optionally) green-flag only, within `tolerance` of the median.

    symmetric=True is the literal reading of 'within two per cent of the stint median'; symmetric=False is the
    one-sided reading (t <= 1.02 median). Both select the same runs on this data set (verified in the notebook).
    """
    timed = g[~g["PitIn"] & ~g["PitOut"] & g["LapTime_s"].notna()]
    if green_only:
        timed = timed[timed["TrackStatus"] == 1]
    if timed.empty:
        return timed
    median = timed["LapTime_s"].median()
    if symmetric:
        keep = (timed["LapTime_s"] - median).abs() <= tolerance * median
    else:
        keep = timed["LapTime_s"] <= (1 + tolerance) * median
    return timed[keep]


def stint_summary(laps: pd.DataFrame, **filter_kw) -> pd.DataFrame:
    """Clean-lap statistics for every (driver, stint) in a practice session."""
    rows = []
    for (driver, stint), g in laps.groupby(["Driver", "Stint"]):
        clean = clean_laps(g.sort_values("LapNumber"), **filter_kw)
        if clean.empty:
            continue
        rows.append({"driver": driver, "stint": int(stint), "compound": g["Compound"].iloc[0],
                     "clean_laps": len(clean), "mean_lap_s": clean["LapTime_s"].mean(),
                     "lap_numbers": list(clean["LapNumber"].astype(int))})
    return pd.DataFrame(rows)


def fp2_long_runs(laps: pd.DataFrame, min_clean_laps: int = 5, exclude: tuple[str, ...] = (),
                  **filter_kw) -> pd.DataFrame:
    """Each driver's best (fastest-average) long run, as a gap to the fastest long run.

    `exclude` is used ONLY for the source-matching exception (Bearman), never for independent extraction.
    """
    s = stint_summary(laps, **filter_kw)
    s = s[(s["clean_laps"] >= min_clean_laps) & ~s["driver"].isin(exclude)]
    best = s.loc[s.groupby("driver")["mean_lap_s"].idxmin()].sort_values("mean_lap_s")
    best["gap_s"] = best["mean_lap_s"] - best["mean_lap_s"].min()
    return best.set_index("driver")


def quali_gaps(quali: pd.DataFrame) -> pd.Series:
    """Best lap from any qualifying segment, as a gap to the pole lap."""
    best = quali.set_index("Abbreviation")[["Q1_s", "Q2_s", "Q3_s"]].min(axis=1)
    return (best - best.min()).rename("quali_gap_s")


def push_laps(laps: pd.DataFrame) -> pd.Series:
    """Fastest valid single lap of each driver (green flag, accurately timed): Ruth's 'FP2 push lap'."""
    valid = laps[(laps["TrackStatus"] == 1) & laps["IsAccurate"].astype(bool) & laps["LapTime_s"].notna()]
    return valid.groupby("Driver")["LapTime_s"].min().rename("push_lap_s")


def friday_weighted(long_run_gap: pd.Series, quali_gap: pd.Series) -> pd.Series:
    """0.5 x long-run gap + 0.5 x quali gap; quali gap alone where there is no long run."""
    lr = long_run_gap.reindex(quali_gap.index)
    return (0.5 * lr + 0.5 * quali_gap).fillna(quali_gap).rename("friday_weighted_s")


def saturday_rebased(quali_gap: pd.Series, long_run_mean: pd.Series, push_lap: pd.Series) -> pd.DataFrame:
    """Quali gap + own (long-run minus push-lap) drop-off relative to the field median.

    For drivers WITH a drop-off, the centring constant (median) cancels in the final min-rebasing, so their
    values do not identify which centring Ruth used. It matters only for drivers WITHOUT a long run, who get
    the field-median drop-off (relative drop 0): an explicit assumption, flagged in 'drop_source'.
    """
    drop = (long_run_mean - push_lap.reindex(long_run_mean.index)).rename("drop_s")
    rel = (drop - drop.median()).reindex(quali_gap.index)
    out = pd.DataFrame({"quali_gap_s": quali_gap, "drop_s": drop.reindex(quali_gap.index),
                        "relative_drop_s": rel.fillna(0.0),
                        "drop_source": np.where(rel.isna(), "assumed field median", "FP2 timing")})
    raw = out["quali_gap_s"] + out["relative_drop_s"]
    out["saturday_rebased_s"] = raw - raw.min()
    return out


def fuel_corrected_degradation(lap_numbers: np.ndarray, lap_times: np.ndarray, fuel_s_per_lap: float) -> dict:
    """Linear fit of lap time on lap number within ONE stint, converted to a tyre-degradation rate.

    Within a stint, tyre age and fuel burnt both rise by one per lap, so the fitted slope b equals
    d - f (degradation minus fuel effect) and the two cannot be separated from the stint alone:
    d_hat = b_hat + f depends entirely on the ASSUMED fuel effect f. The confidence interval reflects only the
    lap-to-lap scatter (OLS, independent errors); uncertainty in f shifts the whole interval.
    """
    x = np.asarray(lap_numbers, dtype=float)
    y = np.asarray(lap_times, dtype=float)
    n = len(x)
    X = np.column_stack([np.ones(n), x - x.mean()])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = resid @ resid / (n - 2)
    se_b = float(np.sqrt(s2 / ((x - x.mean()) ** 2).sum()))
    half = t975(n - 2) * se_b
    lag1 = float(np.corrcoef(resid[:-1], resid[1:])[0, 1]) if n > 3 else float("nan")
    return {"n_laps": n, "slope_s_per_lap": float(beta[1]), "slope_se": se_b,
            "deg_hat": float(beta[1]) + fuel_s_per_lap, "deg_ci_low": float(beta[1]) + fuel_s_per_lap - half,
            "deg_ci_high": float(beta[1]) + fuel_s_per_lap + half, "resid_sd_s": float(np.sqrt(s2)),
            "resid_lag1_corr": lag1}
