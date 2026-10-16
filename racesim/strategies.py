"""Race strategies (start tyre + planned stops) and the competitor pit schedule.

Following Heilmeier et al. (2018), strategies are INPUTS to the race simulation: every car
arrives with a start compound and a list of planned in-laps. The only strategy decision the
engine takes by itself is Ruth's Safety-Car rule (see `engine.py`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PitStop:
    in_lap: int     # the car pits at the END of this lap ...
    compound: str   # ... and starts lap in_lap + 1 on a new set of this compound


@dataclass(frozen=True)
class Strategy:
    start_compound: str
    stops: tuple[PitStop, ...]
    label: str = ""

    def compounds_used(self) -> set[str]:
        return {self.start_compound, *(s.compound for s in self.stops)}

    def uses_two_dry_specs(self) -> bool:
        """FIA 2026 Sporting Regulations B6.3.6: at least two dry specifications in the race."""
        return len(self.compounds_used()) >= 2


@dataclass(frozen=True)
class Entry:
    """One car in the race."""
    code: str
    pace_s: float             # race-pace gap per lap to the fastest car (Ruth's estimate)
    grid_slot: int | None     # None = pit-lane start
    strategy: Strategy


def one_stop(start: str, in_lap: int, to: str, label: str = "") -> Strategy:
    return Strategy(start_compound=start, stops=(PitStop(in_lap, to),), label=label)


def field_pit_laps(order: Sequence[str], first: int = 14, last: int = 24,
                   mode: str = "grid_linear", pace: dict[str, float] | None = None,
                   seed: int | None = None) -> dict[str, int]:
    """Assign each competitor an in-lap inside Ruth's window [first, last].

    mode
      grid_linear  leaders latest: linear in starting order from `last` (P1) to `first`
      pace_linear  the same interpolation, but ordered by race pace (fastest latest)
      reverse      leaders earliest (the opposite of Ruth's description)
      uniform      everyone stops on the middle lap of the window
      random       the grid_linear laps randomly permuted (requires `seed`)
    """
    codes = list(order)
    if mode == "pace_linear":
        if pace is None:
            raise ValueError("pace_linear needs the pace dictionary")
        codes = sorted(codes, key=lambda c: pace[c])
    n = len(codes)
    span = last - first
    laps = [last - math.floor(rank * span / (n - 1) + 0.5) for rank in range(n)]
    if mode == "reverse":
        laps = laps[::-1]
    elif mode == "uniform":
        laps = [math.floor((first + last) / 2 + 0.5)] * n
    elif mode == "random":
        if seed is None:
            raise ValueError("random mode needs a seed for reproducibility")
        laps = list(np.random.default_rng(seed).permutation(laps))
    elif mode not in ("grid_linear", "pace_linear"):
        raise ValueError(f"unknown mode {mode!r}")
    return {code: int(lap) for code, lap in zip(codes, laps)}
