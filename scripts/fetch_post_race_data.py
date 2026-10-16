"""Fetch the ACTUAL 2026 Italian GP race outcome - for the post-race epilogue ONLY.

Data-integrity rule: nothing in data/post_race/ is read before the clearly labelled post-race
'Epilogue' section of the notebook, and nothing in it is used to build, calibrate or select any
part of the model.

Source: OpenF1 public API (https://openf1.org). FastF1's archive did not serve this race
session at the time of writing.

Usage:
    .venv/bin/python scripts/fetch_post_race_data.py
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "post_race"
API = "https://api.openf1.org/v1"


def get(endpoint: str, **query: object) -> pd.DataFrame:
    qs = "&".join(f"{k}={v}" for k, v in query.items())
    try:
        with urllib.request.urlopen(f"{API}/{endpoint}?{qs}", timeout=60) as resp:
            return pd.DataFrame(json.load(resp))
    except urllib.error.HTTPError as err:   # OpenF1 answers 404 when an endpoint has no rows
        print(f"{endpoint}: HTTP {err.code} - no data")
        return pd.DataFrame()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sessions = get("sessions", year=2026, circuit_short_name="Monza", session_name="Race")
    key = int(sessions["session_key"].iloc[0])
    drivers = get("drivers", session_key=key)[["driver_number", "name_acronym", "team_name"]].drop_duplicates("driver_number")

    tables = {
        "race_results": get("session_result", session_key=key),
        "starting_grid": get("starting_grid", session_key=key),
        "race_stints": get("stints", session_key=key),
        "pit_stops": get("pit", session_key=key),
        "race_control": get("race_control", session_key=key),
    }
    for name, df in tables.items():
        if df.empty:
            continue
        if "driver_number" in df.columns:
            df = df.merge(drivers, on="driver_number", how="left")
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name}: {len(df)} rows")
    # Official classification incl. grid positions (FastF1 results table; its lap data was unavailable)
    import fastf1
    fastf1.Cache.enable_cache(str(ROOT / "data" / ".fastf1_cache"))
    race = fastf1.get_session(2026, "Italian Grand Prix", "R")
    race.load(laps=False, telemetry=False, weather=False, messages=False)
    race.results[["Abbreviation", "TeamName", "GridPosition", "Position", "ClassifiedPosition", "Status"]].to_csv(
        OUT / "race_results_fastf1.csv", index=False)
    (OUT / "SOURCE.txt").write_text(f"OpenF1 API, 2026 Italian GP race, session_key={key}; "
                                    "grid positions from the FastF1 results table\n")


if __name__ == "__main__":
    main()
