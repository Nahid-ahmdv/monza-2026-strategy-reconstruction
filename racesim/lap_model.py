"""The clean-air lap-time model (Heilmeier et al. 2018, Eq. 2, in Ruth's parameterisation).

    t_lap = t_ref + pace_i + tyre(compound, age, warm) + fuel(lap)

Traffic (dirty air, being held up, overtaking) is NOT part of this function: it is added afterwards by
`traffic.resolve_racing_lap`, because it depends on the other cars. Whether the warm-up penalty applies on a
given lap (`warm`) is decided by the engine's warm-up policy, not here.
"""
from __future__ import annotations

from .config import Compound, RaceParams


def fuel_penalty(lap: int, total_laps: int, fuel_s_per_lap: float) -> float:
    """Fuel-mass time penalty of `lap` (1-based), measured relative to the final lap.

    Convention: lap k carries f*(N-k), so the penalty is 0 on lap N. The physical fuel load at the start of
    lap k is N-k+1 laps' worth; the convention differs from it by the constant f, which is identical for every
    car and therefore cancels in every gap and position.
    """
    return fuel_s_per_lap * (total_laps - lap)


def tyre_penalty(compound: Compound, age: int, warm: bool) -> float:
    """Time relative to a NEW hard tyre for a set that has already completed `age` laps.

    offset + linear degradation, plus the compound's warm-up penalty if this is the set's warm-up lap.
    """
    return compound.offset_s + compound.deg_s_per_lap * age + (compound.warmup_s if warm else 0.0)


def clean_air_lap_time(params: RaceParams, pace_s: float, compound: str, age: int, warm: bool, lap: int) -> float:
    """Lap time the car would do on an empty track."""
    return (params.reference_lap_s
            + pace_s
            + tyre_penalty(params.compounds[compound], age, warm)
            + fuel_penalty(lap, params.total_laps, params.fuel_s_per_lap))
