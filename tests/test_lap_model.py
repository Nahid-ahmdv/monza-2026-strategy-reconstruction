"""Isolated-car tests: the lap-time equation must match hand calculation exactly."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import car, no_stop, one_stop
from racesim.config import variant
from racesim.engine import simulate_race
from racesim.lap_model import clean_air_lap_time, fuel_penalty, tyre_penalty


def test_fuel_penalty_is_linear_and_zero_on_last_lap():
    assert fuel_penalty(1, 53, 0.07) == pytest.approx(52 * 0.07)
    assert fuel_penalty(53, 53, 0.07) == pytest.approx(0.0)
    assert np.allclose(np.diff([fuel_penalty(k, 53, 0.07) for k in range(1, 54)]), -0.07)


def test_tyre_penalty_offsets_degradation_and_warmup(params):
    H, M, S = (params.compounds[c] for c in ("HARD", "MEDIUM", "SOFT"))
    assert tyre_penalty(H, 0, False) == pytest.approx(0.0)
    assert tyre_penalty(H, 0, True) == pytest.approx(0.6)
    assert tyre_penalty(H, 3, True) == pytest.approx(0.645)     # warm-up deferred to a later (green) lap
    assert tyre_penalty(H, 20, False) == pytest.approx(0.30)
    assert tyre_penalty(M, 0, True) == pytest.approx(-0.20)      # no medium warm-up
    assert tyre_penalty(M, 10, False) == pytest.approx(0.30)
    assert tyre_penalty(S, 5, False) == pytest.approx(0.15)


def test_medium_vs_hard_crossover_is_after_six_laps(params):
    """Same-age medium minus hard = -0.20 + 0.035 a: negative for ages 0..5, positive from age 6."""
    H, M = params.compounds["HARD"], params.compounds["MEDIUM"]
    diff = [tyre_penalty(M, a, False) - tyre_penalty(H, a, False) for a in range(10)]
    assert all(d < 0 for d in diff[:6]) and all(d > 0 for d in diff[6:])


def test_isolated_car_lap_times_match_closed_form(params):
    """Single car on hards, no stop: t_k = t_ref + pace + 0.015 (k-1) + 0.07 (53-k)."""
    res = simulate_race([car("X", 0.3, 1)], params)
    k = np.arange(1, 54)
    expected = 84.0 + 0.3 + 0.015 * (k - 1) + 0.07 * (53 - k)
    assert np.allclose(res.laps.set_index("lap")["lap_time_s"].to_numpy(), expected, atol=1e-12)
    assert clean_air_lap_time(params, 0.3, "HARD", 0, False, 1) == pytest.approx(expected[0])


def test_isolated_car_gets_faster_as_fuel_burns(params):
    lt = simulate_race([car("X", 0.0, 1)], variant(params, HARD__deg_s_per_lap=0.0)).laps["lap_time_s"].to_numpy()
    assert np.allclose(np.diff(lt), -0.07)


def test_tyre_degradation_accumulates(params):
    lt = simulate_race([car("X", 0.0, 1, no_stop("MEDIUM"))], variant(params, fuel_s_per_lap=0.0)).laps["lap_time_s"].to_numpy()
    assert np.allclose(np.diff(lt), 0.05)
    assert lt[0] == pytest.approx(84.0 - 0.20)


def test_pit_stop_costs_exactly_the_pit_loss_once(flat_params):
    base = simulate_race([car("X", 0.0, 1, no_stop("HARD"))], flat_params)
    stop = simulate_race([car("X", 0.0, 1, one_stop("HARD", 20, "HARD"))], flat_params)
    diff = stop.classification.loc["X", "race_time_s"] - base.classification.loc["X", "race_time_s"]
    assert diff == pytest.approx(22.0)
    assert stop.laps["pit_loss_s"].sum() == pytest.approx(22.0)
    assert stop.laps.set_index("lap").loc[20, "pit_loss_s"] == pytest.approx(22.0)


def test_tyre_age_resets_and_green_warmup_applies_once(params):
    res = simulate_race([car("X", 0.0, 1, one_stop("MEDIUM", 10, "HARD"))], params)
    laps = res.laps.set_index("lap")
    assert list(laps.loc[1:10, "tyre_age"]) == list(range(10))
    assert list(laps.loc[11:14, "tyre_age"]) == [0, 1, 2, 3]
    assert list(laps["warmup_applied"]).count(True) == 1 and laps.loc[11, "warmup_applied"]
    assert laps.loc[11, "lap_time_s"] == pytest.approx(84.0 + 0.6 + 0.07 * (53 - 11))
    assert laps.loc[12, "lap_time_s"] == pytest.approx(84.0 + 0.015 + 0.07 * (53 - 12))
