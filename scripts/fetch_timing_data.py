"""Fetch the PRE-RACE timing data (FP2 laps, qualifying results) for the 2026 Italian GP.

This script is run ONCE with internet access; the notebook only reads the CSV files it
writes, so the research notebook itself executes offline.

Data-integrity rule: this script deliberately never loads the Race session. Anything
about the actual race lives in `scripts/fetch_post_race_data.py` and `data/post_race/`,
which are only used in the clearly labelled post-race epilogue of the notebook.

Usage:
    .venv/bin/python scripts/fetch_timing_data.py
"""
from __future__ import annotations

from pathlib import Path

import fastf1
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "timing"
CACHE = ROOT / "data" / ".fastf1_cache"

YEAR, EVENT = 2026, "Italian Grand Prix"


def td_seconds(series: pd.Series) -> pd.Series:
    """Convert a timedelta column to float seconds (NaN preserved)."""
    return series.dt.total_seconds()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE))

    # ---- FP2: every lap, with compound / stint / pit flags --------------------------
    fp2 = fastf1.get_session(YEAR, EVENT, "FP2")
    fp2.load(laps=True, telemetry=False, weather=True, messages=False)
    laps = fp2.laps.copy()
    keep = ["Driver", "Team", "LapNumber", "Stint", "Compound", "TyreLife", "FreshTyre",
            "LapTime", "PitInTime", "PitOutTime", "TrackStatus", "Deleted", "IsAccurate"]
    laps = laps[keep]
    laps["LapTime_s"] = td_seconds(laps["LapTime"])
    laps["PitIn"] = laps["PitInTime"].notna()
    laps["PitOut"] = laps["PitOutTime"].notna()
    laps = laps.drop(columns=["LapTime", "PitInTime", "PitOutTime"])
    laps.to_csv(OUT / "fp2_laps.csv", index=False)

    fp2_res = fp2.results[["Abbreviation", "FullName", "TeamName", "Position"]].copy()
    fp2_res.to_csv(OUT / "fp2_results.csv", index=False)
    weather = fp2.weather_data.copy()
    weather["Time_s"] = td_seconds(weather["Time"])
    weather.drop(columns=["Time"]).to_csv(OUT / "fp2_weather.csv", index=False)

    # ---- Qualifying: Q1/Q2/Q3 best laps ----------------------------------------------
    q = fastf1.get_session(YEAR, EVENT, "Q")
    q.load(laps=False, telemetry=False, weather=False, messages=False)
    res = q.results[["Abbreviation", "FullName", "TeamName", "Position", "Q1", "Q2", "Q3"]].copy()
    for col in ["Q1", "Q2", "Q3"]:
        res[f"{col}_s"] = td_seconds(res[col])
    res = res.drop(columns=["Q1", "Q2", "Q3"])
    res.to_csv(OUT / "quali_results.csv", index=False)

    print(f"FP2 laps: {len(laps)} rows, drivers: {laps['Driver'].nunique()}")
    print(f"Quali results: {len(res)} rows")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
