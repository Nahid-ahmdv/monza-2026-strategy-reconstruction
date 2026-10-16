"""Assemble the 22-car field for one scenario and run Ruth's 3 x 5 Antonelli matrix.

Every operative choice (target car, its strategies and pit-lane start, the field's compounds, pit window and
pit-lap assignment, the Safety-Car scenarios) comes from the `ModelSpec` loaded from the YAML configuration.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd

from .config import ModelSpec, RaceParams
from .engine import RaceResult, simulate_race
from .safety_car import SafetyCar
from .strategies import Entry, Strategy, field_pit_laps, one_stop


def field_laps_for(spec: ModelSpec, grid: pd.DataFrame, mode: str | None = None,
                   pace: Mapping[str, float] | None = None, seed: int | None = None) -> dict[str, int]:
    """In-lap of every car other than the target, inside the configured window."""
    order = [d for d in grid.sort_values("grid_slot")["driver"] if d != spec.target]
    first, last = spec.field.window
    return field_pit_laps(order, first, last, mode=mode or spec.field.assignment,
                          pace=dict(pace) if pace else None, seed=seed)


def build_field(spec: ModelSpec, grid: pd.DataFrame, pace: Mapping[str, float], target_strategy: Strategy,
                field_laps: Mapping[str, int]) -> list[Entry]:
    """22 entries: the target car with `target_strategy`, every other car on the configured field plan."""
    entries = []
    for row in grid.itertuples():
        if row.driver == spec.target:
            entries.append(Entry(spec.target, pace[spec.target], None if spec.target_pit_lane_start else row.grid_slot,
                                 target_strategy))
        else:
            entries.append(Entry(row.driver, pace[row.driver], row.grid_slot,
                                 one_stop(spec.field.start_compound, field_laps[row.driver], spec.field.next_compound)))
    return entries


def run_scenario(spec: ModelSpec, grid: pd.DataFrame, pace: Mapping[str, float], strategy: Strategy | str,
                 sc_lap: int | None, field_laps: Mapping[str, int], record: bool = True,
                 params: RaceParams | None = None) -> RaceResult:
    p = params or spec.params
    strat = spec.strategies[strategy] if isinstance(strategy, str) else strategy
    sc = SafetyCar(sc_lap, p.sc_duration_laps) if sc_lap is not None else None
    return simulate_race(build_field(spec, grid, pace, strat, field_laps), p, sc, record=record)


def run_matrix(spec: ModelSpec, grid: pd.DataFrame, pace: Mapping[str, float],
               params: RaceParams | None = None, strategies: Mapping[str, Strategy] | None = None,
               sc_laps: Sequence[int | None] | None = None, field_laps: Mapping[str, int] | None = None,
               record: bool = False) -> tuple[pd.DataFrame, dict[tuple[str, str], RaceResult]]:
    """Target car's finishing position, gap and race time for every (strategy, safety-car) pair."""
    strategies = strategies or spec.strategies
    sc_laps = spec.sc_laps if sc_laps is None else sc_laps
    field_laps = field_laps or field_laps_for(spec, grid)
    rows, results = [], {}
    for key, strat in strategies.items():
        for sc_lap in sc_laps:
            res = run_scenario(spec, grid, pace, strat, sc_lap, field_laps, record=record, params=params)
            sc_label = "none" if sc_lap is None else str(sc_lap)
            results[(key, sc_label)] = res
            cls = res.classification
            rows.append({"strategy": key, "sc_lap": sc_label, "position": res.position(spec.target),
                         "gap_to_winner_s": round(res.gap_to_winner(spec.target), 2),
                         "race_time_s": round(float(cls.loc[spec.target, "race_time_s"]), 3),
                         "winner": cls.index[0]})
    return pd.DataFrame(rows), results
