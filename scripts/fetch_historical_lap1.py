"""Historical (pre-2026) evidence on the field spread at the end of lap 1 at Monza.

The 2026 model needs starting offsets, which Ruth does not publish. This script measures, for the 2023-2025
Italian Grands Prix, each car's gap to the leader when it completed lap 1, together with its grid slot. It is
historical data only (no 2026 race information).

Sources: OpenF1 `laps` (date_start of lap 2 = time the car completed lap 1); FastF1 results table for grid slots.

Usage:
    .venv/bin/python scripts/fetch_historical_lap1.py
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import fastf1
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "historical"
API = "https://api.openf1.org/v1"
SESSIONS = {2023: 9157, 2024: 9590, 2025: 9912}


def openf1(endpoint: str, **query: object) -> pd.DataFrame:
    qs = "&".join(f"{k}={v}" for k, v in query.items())
    with urllib.request.urlopen(f"{API}/{endpoint}?{qs}", timeout=60) as resp:
        return pd.DataFrame(json.load(resp))


def main() -> None:
    (ROOT / "data" / ".fastf1_cache").mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(ROOT / "data" / ".fastf1_cache"))
    rows = []
    for year, key in SESSIONS.items():
        lap2 = openf1("laps", session_key=key, lap_number=2)[["driver_number", "date_start"]]
        drivers = openf1("drivers", session_key=key)[["driver_number", "name_acronym"]].drop_duplicates("driver_number")
        race = fastf1.get_session(year, "Italian Grand Prix", "R")
        race.load(laps=False, telemetry=False, weather=False, messages=False)
        grid = race.results[["Abbreviation", "GridPosition", "Status"]].rename(columns={"Abbreviation": "name_acronym"})
        df = lap2.merge(drivers, on="driver_number").merge(grid, on="name_acronym", how="left")
        df["t_end_lap1"] = pd.to_datetime(df["date_start"])
        df["gap_to_leader_end_lap1_s"] = (df["t_end_lap1"] - df["t_end_lap1"].min()).dt.total_seconds()
        df["year"] = year
        rows.append(df[["year", "name_acronym", "GridPosition", "Status", "gap_to_leader_end_lap1_s"]])
    out = pd.concat(rows).sort_values(["year", "gap_to_leader_end_lap1_s"])
    out.to_csv(OUT / "monza_lap1_gaps_2023_2025.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
