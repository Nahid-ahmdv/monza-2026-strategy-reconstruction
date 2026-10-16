"""Hand-calculated traffic cases (2, 3 and 4 cars) and randomised field invariants.

Expected values are computed by hand from the rules, independently of the implementation.
Baseline rules: dirty air +0.25 s within 2.0 s (start-of-lap gap); a car may pass only if it is at least
0.35 s/lap faster in clean air; a pass happens only when its elapsed time is earlier (time crossing); a car
that may not pass is held 0.5 s behind the car ahead.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import car, check_passes_time_feasible, check_physical, no_stop, one_stop
from racesim.config import variant
from racesim.engine import simulate_race
from racesim.safety_car import SafetyCar
from racesim.traffic import resolve_racing_lap


def lap(p, starts: dict, clean: dict):
    order = sorted(starts, key=starts.get)
    return resolve_racing_lap(order, starts, clean, p)


def test_two_car_example_no_time_is_taken_from_the_leader(params):
    """F 1.0 s behind, 0.40 s faster, dirty air: projected 1.0 + 89.6 + 0.25 = 90.85 > 90.0, so no pass this
    lap and nobody is delayed (an instant-pass reading would declare a pass and delay the leader to 91.35 s)."""
    out = lap(params, {"L": 0.0, "F": 1.0}, {"L": 90.0, "F": 89.6})
    assert out.order == ["L", "F"] and not out.passes
    assert out.end_time == pytest.approx({"L": 90.0, "F": 90.85})
    assert not out.held["F"] and not out.pushed


def test_pass_happens_when_times_cross(params):
    """F 0.2 s behind, 0.5 s faster: projected 0.2 + 89.5 + 0.25 = 89.95 < 90.0 -> F ahead; times unchanged."""
    out = lap(params, {"L": 0.0, "F": 0.2}, {"L": 90.0, "F": 89.5})
    assert out.order == ["F", "L"] and out.passes == [("F", "L")]
    assert out.end_time == pytest.approx({"F": 89.95, "L": 90.0})


def test_ineligible_car_is_held_at_min_gap(params):
    """F 0.5 s behind, 0.30 s faster: would arrive at 0.5 + 89.7 + 0.25 = 90.45 < 90.5 -> held at 90.5."""
    out = lap(params, {"L": 0.0, "F": 0.5}, {"L": 90.0, "F": 89.7})
    assert out.order == ["L", "F"] and out.held["F"]
    assert out.end_time["F"] == pytest.approx(90.5)


def test_exactly_threshold_is_eligible(params):
    """0.35 s faster counts as 'at least 0.35': not held even when it ends closer than 0.5 s."""
    out = lap(params, {"L": 0.0, "F": 0.5}, {"L": 90.0, "F": 89.65})
    assert not out.held["F"] and out.end_time["F"] == pytest.approx(90.4) and out.attacking["F"]


def test_dirty_air_only_within_two_seconds(params):
    near = lap(params, {"L": 0.0, "F": 1.9}, {"L": 90.0, "F": 90.0})
    far = lap(params, {"L": 0.0, "F": 2.1}, {"L": 90.0, "F": 90.0})
    assert near.dirty_air["F"] and not far.dirty_air["F"] and not near.dirty_air["L"]
    assert near.end_time["F"] == pytest.approx(1.9 + 90.25)


def test_three_cars_pass_one_attack_the_next(params):
    """L 0.0/90.0; M 0.5/90.4 (dirty -> 91.15); F 1.0/89.5 (dirty -> 90.75).
    F crosses M (90.75 < 91.15, 0.9 s faster) but not L (90.75 > 90.0); F is 0.5 s faster than L -> not held."""
    out = lap(params, {"L": 0.0, "M": 0.5, "F": 1.0}, {"L": 90.0, "M": 90.4, "F": 89.5})
    assert out.order == ["L", "F", "M"] and out.passes == [("F", "M")]
    assert out.end_time == pytest.approx({"L": 90.0, "F": 90.75, "M": 91.15})
    assert not out.pushed                                  # the passed car loses nothing


def test_four_car_train_is_held_at_min_gaps(params):
    """Each car 0.30 s faster than the car ahead (< 0.35, may not pass) and 0.5 s behind it:
    B would arrive at 0.5 + 89.7 + 0.25 = 90.45 < 90.5 -> held at 90.5; C at 1.0 + 89.4 + 0.25 = 90.65 < 91.0
    -> held at 91.0; D at 1.5 + 89.1 + 0.25 = 90.85 < 91.5 -> held at 91.5."""
    starts = {"A": 0.0, "B": 0.5, "C": 1.0, "D": 1.5}
    clean = {"A": 90.0, "B": 89.7, "C": 89.4, "D": 89.1}
    out = lap(params, starts, clean)
    assert out.order == ["A", "B", "C", "D"]
    assert [out.end_time[c] for c in "ABCD"] == pytest.approx([90.0, 90.5, 91.0, 91.5])
    assert all(out.held[c] for c in "BCD")


def test_fast_car_passes_a_train_in_one_lap_only_if_times_cross(params):
    """A ends 90.0; B (slower, dirty air) ends 0.5 + 90.2 + 0.25 = 90.95; F (3 s/lap faster, dirty air)
    projects 1.0 + 87.0 + 0.25 = 88.25 and crosses both: F, A, B with nobody's time changed."""
    out = lap(params, {"A": 0.0, "B": 0.5, "F": 1.0}, {"A": 90.0, "B": 90.2, "F": 87.0})
    assert out.order == ["F", "A", "B"] and len(out.passes) == 2
    assert out.end_time == pytest.approx({"F": 88.25, "A": 90.0, "B": 90.95})


def test_no_pass_when_both_end_up_held_behind_the_same_car(params):
    """A 0.0/90.0. V 0.1/90.05 (slower than A) would arrive at 0.1 + 90.05 + 0.25 = 90.40 < 90.50 -> held at 90.50.
    F 0.5/89.69 is 0.36 s faster than V (eligible) but only 0.31 s faster than A (not eligible). F would arrive at
    0.5 + 89.69 + 0.25 = 90.44, earlier than V's 90.50, but it must be held at 90.50 behind A, which is not earlier
    than V: no time-feasible pass. Expected order A, V, F with F alongside V (gap 0, attacking)."""
    out = lap(params, {"A": 0.0, "V": 0.1, "F": 0.5}, {"A": 90.0, "V": 90.05, "F": 89.69})
    assert out.order == ["A", "V", "F"] and not out.passes
    assert out.end_time == pytest.approx({"A": 90.0, "V": 90.5, "F": 90.5})
    assert out.held["V"] and out.held["F"] and out.attacking["F"]       # F is blocked alongside V


def test_time_crossing_never_takes_time_from_another_car(params):
    """Property over 50 random 8-car laps: no car ever ends earlier than its own projection; a car that is not
    held ends exactly at its projection; a held car ends exactly at, or 0.5 s behind, a car that finishes ahead
    of it (the car that blocked it; a later attacker may still slot in between). So a car's time can only be
    increased by a car in front, never by a follower."""
    rng = np.random.default_rng(7)
    for _ in range(50):
        n = 8
        starts = {f"C{i}": float(s) for i, s in enumerate(np.cumsum(rng.uniform(0.0, 1.5, n)))}
        clean = {c: 90.0 + float(rng.uniform(0, 2)) for c in starts}
        out = lap(params, starts, clean)
        for i, c in enumerate(out.order):
            proj = starts[c] + clean[c] + (0.25 if out.dirty_air[c] else 0.0)
            assert out.end_time[c] >= proj - 1e-9
            if not out.held[c]:
                assert out.end_time[c] == pytest.approx(proj)
            else:
                ahead_times = [out.end_time[x] for x in out.order[:i]]
                assert any(abs(out.end_time[c] - t) < 1e-9 or abs(out.end_time[c] - t - params.min_follow_gap_s) < 1e-9
                           for t in ahead_times)


def test_slower_car_never_passes_and_train_forms(flat_params):
    res = simulate_race([car("L", 0.30, 1), car("F", 0.00, 2)], flat_params)   # only 0.30 faster
    f = res.laps[res.laps.car == "F"]
    assert (f["position"] == 2).all()
    assert np.allclose(f["gap_ahead_s"].iloc[1:], flat_params.min_follow_gap_s)


def test_faster_car_closes_then_passes(flat_params):
    """0.40 s faster from 2.46 s back: closes at 0.40 s/lap in clean air, at 0.15 s/lap in dirty air, then passes."""
    res = simulate_race([car("L", 0.40, 1), car("F", 0.00, 22)], flat_params)
    pos_f = res.laps[res.laps.car == "F"].set_index("lap")["position"]
    assert pos_f.iloc[0] == 2 and pos_f.iloc[-1] == 1


@pytest.mark.parametrize("seed", range(12))
def test_random_races_with_stops_and_safety_cars_stay_physical(params, seed):
    """Randomised 22-car races with random one-stop strategies and a random Safety Car."""
    rng = np.random.default_rng(seed)
    comps = ["SOFT", "MEDIUM", "HARD"]
    entries = []
    for i in range(22):
        a, b = rng.choice(3, 2, replace=False)
        entries.append(car(f"C{i:02d}", float(rng.uniform(0, 3)), i + 1,
                           one_stop(comps[a], int(rng.integers(5, 45)), comps[b])))
    sc = SafetyCar(int(rng.integers(1, 45))) if seed % 3 else None
    res = simulate_race(entries, params, sc)
    check_physical(res, params)
    check_passes_time_feasible(res)


@pytest.mark.parametrize("resolution", ["passed_car_yields", "passer_jumps"])
def test_instant_pass_variants_keep_order(params, resolution):
    p = variant(params, pass_resolution=resolution)
    rng = np.random.default_rng(99)
    res = simulate_race([car(f"C{i}", float(rng.uniform(0, 3)), i + 1) for i in range(22)], p)
    check_physical(res, p)


def test_simultaneous_stops_keep_order_and_spacing(params):
    """Two cars 0.6 s apart pit on the same green lap with the same loss: the order is kept and the
    rejoining gap is at least the minimum gap (pit-lane congestion/double-stacking is not modelled)."""
    entries = [car("A", 0.0, 1, one_stop("MEDIUM", 10, "HARD")), car("B", 0.0, 6, one_stop("MEDIUM", 10, "HARD"))]
    res = simulate_race(entries, params)
    lap10 = res.laps[res.laps.lap == 10].sort_values("position")
    assert list(lap10["car"]) == ["A", "B"]
    assert lap10["gap_ahead_s"].iloc[1] >= params.min_follow_gap_s - 1e-9
    check_physical(res, params)
