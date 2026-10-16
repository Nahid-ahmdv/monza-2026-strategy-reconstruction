"""Full 22-car race: completion, conservation, legality, determinism, invariances and configuration integrity."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from conftest import ROOT, check_passes_time_feasible, check_physical, check_timeline
from racesim.config import effective_fingerprint, load_spec, variant
from racesim.engine import start_offset
from racesim.scenarios import build_field, field_laps_for, run_matrix, run_scenario


def test_field_pit_laps_follow_the_configured_window(spec, grid):
    laps = field_laps_for(spec, grid)
    assert len(laps) == 21 and spec.target not in laps
    assert min(laps.values()) == spec.field.window[0] and max(laps.values()) == spec.field.window[1]
    assert laps["GAS"] == 24 and laps["LAW"] == 14                      # leaders latest


def test_every_strategy_uses_two_dry_specifications(spec, grid, fw_pace):
    for strat in spec.strategies.values():
        assert strat.uses_two_dry_specs()
        for e in build_field(spec, grid, fw_pace, strat, field_laps_for(spec, grid)):
            assert e.strategy.uses_two_dry_specs()


def test_starting_offsets(spec, grid, fw_pace):
    entries = {e.code: e for e in build_field(spec, grid, fw_pace, spec.strategies["A"], field_laps_for(spec, grid))}
    p = spec.params
    assert start_offset(entries["GAS"], p) == 0.0
    assert start_offset(entries["LAW"], p) == pytest.approx(21 * 0.117)
    assert start_offset(entries["ANT"], p) - start_offset(entries["LAW"], p) == pytest.approx(6.0)
    assert start_offset(entries["ALB"], p) == pytest.approx(20 * 0.117)  # P20 left empty


@pytest.mark.parametrize("key", ["A", "B", "C"])
@pytest.mark.parametrize("sc_lap", [None, 1, 5, 20, 35])
def test_all_scenarios_complete_and_stay_physical(spec, grid, fw_pace, key, sc_lap):
    res = run_scenario(spec, grid, fw_pace, key, sc_lap, field_laps_for(spec, grid))
    check_physical(res, spec.params)
    check_timeline(res, spec.params)
    check_passes_time_feasible(res)
    assert (res.classification["stops"] == 1).all()                      # one-stop race for everyone
    # green-flag gaps: at least the minimum gap unless the follower is attacking or has just been passed
    L = res.laps.sort_values(["lap", "position"])
    green = L[L.track_status == "GREEN"]
    small = green[(green["position"] > 1) & (green["gap_ahead_s"] < spec.params.min_follow_gap_s - 1e-9)]
    assert (small["attacking"] | (small["times_passed"] > 0) | (small["passes_made"] > 0)).all()


def test_simulation_is_deterministic(spec, grid, fw_pace):
    a, _ = run_matrix(spec, grid, fw_pace)
    b, _ = run_matrix(spec, grid, fw_pace)
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("change", [{"reference_lap_s": 80.0}, {"fuel_s_per_lap": 0.0}, {"fuel_s_per_lap": 0.12}])
def test_common_mode_parameters_do_not_change_outcomes(spec, grid, fw_pace, change):
    """Reference lap and fuel effect are identical for all cars on green laps. Under the Safety Car the queue
    and catch-up lap times scale with the reference lap, so only green-flag runs are compared exactly."""
    base, _ = run_matrix(spec, grid, fw_pace, sc_laps=(None,))
    alt, _ = run_matrix(spec, grid, fw_pace, params=variant(spec.params, **change), sc_laps=(None,))
    assert (base["position"] == alt["position"]).all()
    assert np.allclose(base["gap_to_winner_s"], alt["gap_to_winner_s"], atol=0.011)


def _write_modified(tmp_path, mutate) -> object:
    raw = yaml.safe_load((ROOT / "config" / "monza_2026.yaml").read_text())
    mutate(raw)
    path = tmp_path / "modified.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_spec(path)


def test_yaml_strategy_change_changes_behaviour_and_fingerprint(spec, grid, fw_pace, tmp_path):
    def move_b(raw):
        raw["antonelli"]["strategies"]["B"]["stops"]["value"] = [[30, "MEDIUM"]]
    spec2 = _write_modified(tmp_path, move_b)
    assert spec2.strategies["B"].stops[0].in_lap == 30
    laps = field_laps_for(spec, grid)
    fp1 = effective_fingerprint(spec, fw_pace, laps, grid)
    fp2 = effective_fingerprint(spec2, fw_pace, laps, grid)
    assert fp1 != fp2
    r1 = run_scenario(spec, grid, fw_pace, "B", None, laps, record=False).classification.loc["ANT", "race_time_s"]
    r2 = run_scenario(spec2, grid, fw_pace, "B", None, laps, record=False).classification.loc["ANT", "race_time_s"]
    assert r1 != r2


def test_yaml_window_change_changes_field_laps(spec, grid, tmp_path):
    def widen(raw):
        raw["field"]["pit_window_laps"]["value"] = [12, 26]
    spec2 = _write_modified(tmp_path, widen)
    assert field_laps_for(spec2, grid)["GAS"] == 26 and field_laps_for(spec2, grid)["LAW"] == 12


def test_descriptive_text_does_not_change_effective_fingerprint(spec, grid, fw_pace, tmp_path):
    def edit_note(raw):
        raw["race"]["total_laps"]["note"] = "edited note only"
    spec2 = _write_modified(tmp_path, edit_note)
    laps = field_laps_for(spec, grid)
    assert effective_fingerprint(spec, fw_pace, laps, grid) == effective_fingerprint(spec2, fw_pace, laps, grid)
