"""Fetch tyre-stint lengths from the 2025 Italian GP (historical, pre-2026-race evidence).

Ruth Buscombe cites "last year's stint lengths from Pirelli's 2025 race report" as the
evidence that the hard tyre can run ~51 laps at Monza. We use the official 2025 timing
data for the same purpose: to put an evidence-based number on how long each compound
can last, which our Safety-Car rule ("pit if the next compound can reach the flag")
requires but the article does not state.

Usage:
    .venv/bin/python scripts/fetch_historical_stints.py
"""
from __future__ import annotations

from pathlib import Path

import fastf1

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "historical"
CACHE = ROOT / "data" / ".fastf1_cache"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE))

    race = fastf1.get_session(2025, "Italian Grand Prix", "R")
    race.load(laps=True, telemetry=False, weather=False, messages=False)
    laps = race.laps
    stints = (laps.groupby(["Driver", "Stint", "Compound"], as_index=False)
                  .agg(first_lap=("LapNumber", "min"), last_lap=("LapNumber", "max"),
                       laps=("LapNumber", "size"), tyre_life_end=("TyreLife", "max")))
    stints = stints.sort_values(["Compound", "laps"], ascending=[True, False])
    stints.to_csv(OUT / "monza_2025_race_stints.csv", index=False)
    print(stints.groupby("Compound")[["laps", "tyre_life_end"]].max())
    print(stints.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
