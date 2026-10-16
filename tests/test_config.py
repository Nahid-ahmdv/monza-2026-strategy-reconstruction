"""Configuration contract: every YAML entry is either OPERATIVE (changing it changes the effective specification)
or FIXED (changing it is rejected); unsupported switch values and parameter values are rejected, never silently
ignored."""
from __future__ import annotations

import copy

import pytest
import yaml

from conftest import CONFIG
from racesim.config import FIXED_CHOICES, _canonical, load_spec, parameter_register, variant
from racesim.pace import t975


def _records(node, prefix=""):
    if isinstance(node, dict) and "value" in node and "status" in node:
        yield prefix, node
    elif isinstance(node, dict):
        for k, v in node.items():
            yield from _records(v, f"{prefix}.{k}" if prefix else k)


def _perturb(v):
    if isinstance(v, bool):
        return not v
    if isinstance(v, int):
        return v + 1
    if isinstance(v, float):
        return v + 0.5
    if isinstance(v, str):
        return v + "_changed"
    if isinstance(v, list):
        return v[:-1] if len(v) > 1 else v + v + [0]
    if v is None:
        return 1
    raise TypeError(type(v))


def test_every_yaml_entry_is_operative_or_fixed(tmp_path):
    raw = yaml.safe_load(CONFIG.read_text())
    base = _canonical(load_spec(CONFIG))
    paths = [p for section, node in raw.items() if section not in ("meta", "sources") for p, _ in _records(node, section)]
    assert len(paths) == len(parameter_register(CONFIG)) > 50
    for path in paths:
        mutated = copy.deepcopy(raw)
        node = mutated
        for part in path.split("."):
            node = node[part]
        node["value"] = _perturb(node["value"])
        f = tmp_path / "m.yaml"
        f.write_text(yaml.safe_dump(mutated))
        try:
            spec = load_spec(f)
        except (ValueError, KeyError, TypeError):
            continue                                                   # rejected: fixed or validated
        assert _canonical(spec) != base, f"{path} was changed but nothing the simulator uses changed"


def test_fixed_entries_are_labelled_and_reject_other_values(tmp_path):
    raw = yaml.safe_load(CONFIG.read_text())
    reg = parameter_register(CONFIG).set_index("parameter")
    assert set(reg.index[reg["role"] == "fixed"]) == set(FIXED_CHOICES)
    for dotted, required in FIXED_CHOICES.items():
        section, key = dotted.split(".")
        mutated = copy.deepcopy(raw)
        mutated[section][key]["value"] = _perturb(required)
        f = tmp_path / "m.yaml"
        f.write_text(yaml.safe_dump(mutated))
        with pytest.raises(ValueError, match="not implemented"):
            load_spec(f)
    assert dict(load_spec(CONFIG).fixed) == FIXED_CHOICES


def test_alternative_pace_sets_are_operative(spec):
    assert spec.pace_set == "friday_weighted" and spec.alternative_pace_sets == ("saturday_rebased", "blend_50_50")


@pytest.mark.parametrize("change, message", [
    ({"sc_eligibility": "sometimes"}, "not supported"),
    ({"pass_resolution": "instant"}, "not supported"),
    ({"sc_deployment_remainder": "own_lap"}, "not supported"),
    ({"pit_lane_start_gap_s": -2.043}, "negative"),
    ({"grid_slot_gap_s": 0.5, "pit_lane_start_gap_s": 8.457 - 21 * 0.5}, "negative"),   # outside the probe domain
    ({"sc_duration_laps": 0}, "duration"),
    ({"fcy_lap_factor": 1.0}, "factors > 1"),
])
def test_unsupported_values_and_combinations_are_rejected(params, change, message):
    with pytest.raises(ValueError, match=message):
        variant(params, **change)


def test_supported_values_are_accepted(params):
    variant(params, sc_eligibility="deployment_lap_only", sc_deployment_remainder="own_lap_scaled")
    variant(params, pass_resolution="passer_jumps", warmup_policy="first_lap_of_stint")
    variant(params, sc_duration_laps=2)
    variant(params, grid_slot_gap_s=0.0, grid_lap1_loss_per_slot_s=0.117)
    variant(params, grid_slot_gap_s=0.4, pit_lane_start_gap_s=8.457 - 21 * 0.4)    # 0.057 s >= 0: valid


def test_student_t_quantile_comes_from_scipy():
    assert t975(35) == pytest.approx(2.0301, abs=1e-4)       # pooled degradation fit (35 degrees of freedom)
    assert t975(10) == pytest.approx(2.2281, abs=1e-4)
    assert t975(1) == pytest.approx(12.706, abs=1e-3)
