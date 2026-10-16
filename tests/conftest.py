"""Shared fixtures (the model specification, the published grid and pace, tiny synthetic races) and independent
invariant checks (physical and model-internal)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from racesim.config import load_spec, variant  # noqa: E402
from racesim.lap_model import clean_air_lap_time  # noqa: E402
from racesim.safety_car import follow_gap, lap_phase  # noqa: E402
from racesim.strategies import Entry, Strategy, one_stop  # noqa: E402

CONFIG = ROOT / "config" / "monza_2026.yaml"


@pytest.fixture(scope="session")
def spec():
    return load_spec(CONFIG)


@pytest.fixture(scope="session")
def params(spec):
    return spec.params


@pytest.fixture(scope="session")
def grid():
    return pd.read_csv(ROOT / "data" / "published" / "starting_grid.csv")


@pytest.fixture(scope="session")
def fw_pace():
    df = pd.read_csv(ROOT / "data" / "published" / "ruth_friday_weighted_pace.csv")
    return df.set_index("driver")["pace_gap_s"].to_dict()


@pytest.fixture
def flat_params(params):
    """No degradation, no warm-up, no fuel: lap time = reference + pace + compound offset."""
    return variant(params, fuel_s_per_lap=0.0, HARD__deg_s_per_lap=0.0, HARD__warmup_s=0.0,
                   MEDIUM__deg_s_per_lap=0.0, SOFT__deg_s_per_lap=0.0)


def no_stop(compound: str = "HARD") -> Strategy:
    return Strategy(start_compound=compound, stops=())


def car(code: str, pace: float, slot: int | None, strategy: Strategy | None = None) -> Entry:
    return Entry(code, pace, slot, strategy or no_stop())


def per_car_steps(laps: pd.DataFrame) -> np.ndarray:
    """Change of every car's elapsed race time from one completed lap to the next (incl. start offset)."""
    out = []
    for _, g in laps.sort_values(["car", "lap"]).groupby("car"):
        t = g["race_time_s"].to_numpy()
        out.append(np.diff(t))
    return np.concatenate(out) if out else np.array([])


def check_physical(res, p) -> None:
    """Independent invariants every simulated race must satisfy."""
    L = res.laps
    cars = set(res.classification.index)
    # conservation: every car completes exactly laps 1..N once; no car appears or disappears
    for c, g in L.groupby("car"):
        assert sorted(g["lap"]) == list(range(1, p.total_laps + 1)), f"lap set of {c}"
    assert set(L["car"]) == cars
    for lap, g in L.groupby("lap"):
        g = g.sort_values("position")
        assert list(g["position"]) == list(range(1, len(cars) + 1)), f"positions lap {lap}"
        assert (np.diff(g["race_time_s"].to_numpy()) >= -1e-9).all(), f"order vs time lap {lap}"
    # elapsed time never decreases and every lap takes a physically positive time
    assert (per_car_steps(L) > 0).all(), "elapsed race time decreased"
    assert (L["lap_time_s"] > 0).all()
    for col in ("pit_loss_s", "gap_adjust_s"):
        assert (L[col] >= -1e-12).all(), f"negative {col}"
    # accounting identity
    for _, g in L.sort_values(["car", "lap"]).groupby("car"):
        inc = g[["lap_time_s", "pit_loss_s", "gap_adjust_s"]].sum(axis=1).to_numpy()
        t = g["race_time_s"].to_numpy()
        assert np.allclose(np.diff(t), inc[1:], atol=1e-9)


def check_passes_time_feasible(res) -> None:
    """Every recorded pass must leave the attacker STRICTLY earlier than the passed car at the end of that lap
    (times taken before that lap's pit losses and post-pit spacing)."""
    passes = res.events[res.events["type"] == "pass"] if len(res.events) else res.events
    if passes is None or len(passes) == 0:
        return
    L = res.laps.set_index(["lap", "car"])
    t = L["race_time_s"] - L["pit_loss_s"] - L["gap_adjust_s"]
    for ev in passes.itertuples():
        assert t.loc[(ev.lap, ev.car)] < t.loc[(ev.lap, ev.other)], f"lap {ev.lap}: {ev.car} recorded passing {ev.other}"


def check_timeline(res, p) -> None:
    """Invariants of the shared race clock. Labels: [P] physical requirement, [M] property of this model.

    1 [P] coherent lap progress: every lap starts where the previous one (plus pit loss and spacing) ended,
          and takes a positive time;
    2 [M] one deployment and one restart instant for the whole field: t_d = the first completion of lap N,
          t_r = t_d + D L_sc; every lap's label and neutralised seconds follow from its own times and the window;
    3 [P] speed limits: no lap is covered faster than the car's clean-air pace outside the window and the
          full-course-yellow pace inside it;
    4 [P] no car completes lap N+j before the Safety Car;
    5 [M] the pit price is set by the race time at which the in-lap ends;
    6 [P] no overtaking while neutralised: an order change on a lap cut by the deployment was complete at t_d;
          no car passes another that is still on its restart lap or behind the Safety Car;
    7 [M] spacing at the line for laps that start inside the window (time for the car ahead to cover one
          spacing); on the lap cut at t_d, the distances at t_d are kept (nobody closes in or is pushed back);
    8 [P] every car on track through the window spends the whole window neutralised, in laps or in the pits,
          whatever lap it is on (no advantage or loss from lap-index differences).
    """
    w = res.window
    L = res.laps.sort_values(["car", "lap"])
    for _, g in L.groupby("car"):                                                          # 1
        start, line, t = (g[c].to_numpy() for c in ("lap_start_s", "line_time_s", "race_time_s"))
        assert np.allclose(start[1:], t[:-1], atol=1e-9, rtol=0)
        assert np.allclose(t, line + g["pit_loss_s"].to_numpy() + g["gap_adjust_s"].to_numpy(), atol=1e-9, rtol=0)
        assert (line > start).all()
    if w is None:
        assert (L["track_status"] == "GREEN").all()
        return
    t_d, t_r = w.t_deploy, w.t_restart                                                     # 2
    assert t_r == pytest.approx(t_d + p.sc_duration_laps * p.sc_lap_s, abs=1e-9)
    assert L[L.lap == w.deploy_lap]["line_time_s"].min() == t_d
    for r in L.itertuples():
        assert r.track_status == lap_phase(r.lap_start_s, r.line_time_s, w)
        assert r.neutral_s == pytest.approx(w.neutral_overlap(r.lap_start_s, r.line_time_s), abs=1e-9)
    pace = res.classification["pace_s"]
    for r in L.itertuples():                                                               # 3
        if p.sc_deployment_remainder != "fcy_lap":
            break                                          # sensitivity option 'own_lap_scaled' is not bound by L_fcy
        tau = clean_air_lap_time(p, pace[r.car], r.compound, int(r.tyre_age), bool(r.warmup_applied), int(r.lap))
        green_time = r.line_time_s - r.lap_start_s - r.neutral_s
        assert green_time / tau + r.neutral_s / p.fcy_lap_s >= 1 - 1e-9, (r.car, r.lap)
    for j in range(1, p.sc_duration_laps + 1):                                             # 4
        k = w.deploy_lap + j
        if k <= p.total_laps:
            assert (L[L.lap == k]["line_time_s"] >= t_d + j * p.sc_lap_s - 1e-9).all()
    if len(res.events):                                                                    # 5
        for ev in res.events[res.events.type == "pit"].itertuples():
            assert ev.safety_car_price == w.neutral(ev.pit_entry_s)
            assert ev.loss_s == (p.pit_loss_sc_s if ev.safety_car_price else p.pit_loss_green_s)
    for _, g in L.groupby("lap"):                                                          # 6
        rows = list(g.itertuples())
        for a in rows:
            for b in rows:
                if not (a.lap_start_s < b.lap_start_s and b.line_time_s < a.line_time_s):
                    continue                                       # b did not overtake a on this lap
                assert b.track_status in ("GREEN", "SC_DEPLOYED"), (b.car, a.car, b.lap)
                assert a.track_status not in ("SC", "SC_RESTART"), (b.car, a.car, b.lap)
                if b.track_status == "SC_DEPLOYED" and a.line_time_s > t_d:
                    x = lambda r: (t_d - r.lap_start_s) / (r.green_end_s - r.lap_start_s)
                    assert x(b) >= x(a) - 1e-9, (b.car, a.car, b.lap)    # already ahead at t_d
        seq = sorted(rows, key=lambda r: r.line_time_s)                                   # 7
        for a, b in zip(seq, seq[1:]):
            if b.track_status in ("SC", "SC_RESTART"):
                assert b.line_time_s - a.line_time_s >= follow_gap(a.line_time_s, w, p) - 1e-9, (a.car, b.car, b.lap)
            elif (b.track_status == "SC_DEPLOYED" and a.track_status == "SC_DEPLOYED" and a.line_time_s > t_d
                  and p.sc_deployment_remainder == "fcy_lap"):
                xa = (t_d - a.lap_start_s) / (a.green_end_s - a.lap_start_s)       # fractions of the lap done at t_d
                xb = (t_d - b.lap_start_s) / (b.green_end_s - b.lap_start_s)
                assert b.line_time_s - a.line_time_s == pytest.approx((xa - xb) * p.fcy_lap_s, abs=1e-9)
    for _, g in L.groupby("car"):                                                          # 8
        if g["lap_start_s"].iloc[0] <= t_d and g["race_time_s"].iloc[-1] >= t_r:
            parked = sum(w.neutral_overlap(a, b) for a, b in zip(g["line_time_s"], g["race_time_s"]))
            assert g["neutral_s"].sum() + parked == pytest.approx(t_r - t_d, abs=1e-6)


__all__ = ["car", "no_stop", "one_stop", "per_car_steps", "check_physical", "check_passes_time_feasible",
           "check_timeline"]
