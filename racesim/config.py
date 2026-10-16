"""Load the YAML assumption register into typed, immutable model objects.

The YAML file stores every parameter as {value, status, source, note} (optionally `role`). `load_spec` turns it
into a `ModelSpec`: the numerical `RaceParams` *and* every operative modelling choice (the three Antonelli
strategies, the field's compounds, pit window and pit-lap assignment rule, the pace sets and the Safety-Car
scenarios). Nothing that changes simulator behaviour is hard-coded elsewhere, so the fingerprint of the effective
specification (`effective_fingerprint`) covers everything the simulator actually uses.

Every YAML entry is either OPERATIVE (read and used) or FIXED (`role: fixed`): read and validated, with any value
other than the one the code implements rejected. `validate_params` also rejects unknown switch values and
parameter values the engine does not support, so a variant can never be silently ignored.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import pandas as pd
import yaml

from .strategies import PitStop, Strategy

# Pace sets the notebook can build from the data (section 4); configuration names are checked against this list.
KNOWN_PACE_SETS = ("friday_weighted", "saturday_rebased", "blend_50_50", "friday_weighted_unrounded",
                   "saturday_rebased_unrounded", "friday_weighted_independent")

# FIXED choices: the only value each key may take, because it is the only behaviour the code implements.
FIXED_CHOICES = {
    "safety_car.model": "global_timeline",                 # Safety Car on the shared race clock (safety_car.py)
    "tyres.starting_sets_new": True,                       # every car starts on a new set (age 0)
    "start.pit_lane_starter_slot_left_empty": True,        # published grid used unchanged; P20 stays empty
    "safety_car.deployment_instant": "leader_completes_lap_N",
    "pace.long_run_filter": "symmetric_2pct_green_only",   # data pipeline of notebook section 4
    "pace.bearman_exception": "source_matching",           # data pipeline of notebook section 4
}

_ALLOWED = {
    "sc_eligibility": ("every_cheap_lap", "deployment_lap_only"),
    "warmup_policy": ("first_green_lap", "first_lap_of_stint"),
    "pass_resolution": ("time_crossing", "passed_car_yields", "passer_jumps"),
    "pass_proximity": ("any_point_in_lap", "start_of_lap"),
    "pass_basis": ("intrinsic", "actual"),
    "sc_deployment_remainder": ("fcy_lap", "own_lap_scaled"),
}


@dataclass(frozen=True)
class Compound:
    """A dry tyre compound. All times are seconds relative to a NEW hard tyre."""
    name: str
    offset_s: float        # pace of a new set relative to a new hard (negative = faster)
    deg_s_per_lap: float   # linear time loss per lap of tyre age
    warmup_s: float        # extra time on the warm-up lap of a set fitted at a pit stop
    max_life_laps: int     # longest stint allowed (used by the Safety-Car pit rule)


@dataclass(frozen=True)
class RaceParams:
    """Every number and switch the race engine uses. Frozen: variants are made with `variant()`."""
    total_laps: int
    reference_lap_s: float
    sc_lap_factor: float           # queue (Safety-Car) lap = factor x reference lap
    fcy_lap_factor: float          # full-course-yellow (catch-up) lap = factor x reference lap
    fuel_s_per_lap: float
    compounds: Mapping[str, Compound]
    warmup_at_race_start: bool
    warmup_policy: str             # "first_green_lap" (baseline) | "first_lap_of_stint" (sensitivity)
    pit_loss_green_s: float
    pit_loss_sc_s: float
    dirty_air_window_s: float
    dirty_air_penalty_s: float
    pass_window_s: float
    pass_proximity: str            # "any_point_in_lap" | "start_of_lap"
    pass_min_advantage_s: float
    pass_basis: str                # "intrinsic" | "actual"
    pass_resolution: str           # "time_crossing" (baseline) | "passed_car_yields" | "passer_jumps" (sensitivities)
    duel_loss_attacker_s: float
    duel_loss_defender_s: float
    min_follow_gap_s: float
    tol: float
    grid_slot_gap_s: float         # starting HEADWAY per grid slot (elapsed time at the start line)
    pit_lane_start_gap_s: float    # pit-lane RELEASE DELAY after the last grid slot (>= 0)
    grid_lap1_loss_per_slot_s: float  # lap-1 TIME LOSS per grid slot (TUM's form); 0 = not used
    sc_duration_laps: int
    sc_eligibility: str            # "every_cheap_lap" (baseline) | "deployment_lap_only" (sensitivity)
    sc_queue_gap_s: float
    sc_opportunistic_stops: bool
    sc_deployment_remainder: str   # "fcy_lap" (baseline) | "own_lap_scaled": factor x own racing time (sensitivity)

    @property
    def sc_lap_s(self) -> float:
        """Lap time of the Safety Car (and of the car directly behind it)."""
        return self.sc_lap_factor * self.reference_lap_s

    @property
    def fcy_lap_s(self) -> float:
        """Lap time at the full-course-yellow delta pace: the fastest any car may lap while neutralised."""
        return self.fcy_lap_factor * self.reference_lap_s


def validate_params(p: RaceParams) -> RaceParams:
    """Reject unknown switch values and parameter combinations the engine does not implement."""
    for name, allowed in _ALLOWED.items():
        if getattr(p, name) not in allowed:
            raise ValueError(f"{name}={getattr(p, name)!r} is not supported; allowed: {allowed}")
    for name in ("grid_slot_gap_s", "pit_lane_start_gap_s", "grid_lap1_loss_per_slot_s", "min_follow_gap_s",
                 "sc_queue_gap_s", "pit_loss_green_s", "pit_loss_sc_s", "duel_loss_attacker_s",
                 "duel_loss_defender_s", "dirty_air_penalty_s"):
        if getattr(p, name) < 0:
            raise ValueError(f"{name}={getattr(p, name)} is negative: not a physical value of this model")
    if p.sc_duration_laps < 1 or p.sc_lap_factor <= 1 or p.fcy_lap_factor <= 1:
        raise ValueError("Safety-Car duration must be >= 1 lap and both neutralised lap factors > 1")
    return p


@dataclass(frozen=True)
class FieldPolicy:
    """How the 21 other cars race: one stop, start compound -> next compound, inside a pit window."""
    start_compound: str
    next_compound: str
    window: tuple[int, int]
    assignment: str                # see strategies.field_pit_laps


@dataclass(frozen=True)
class ModelSpec:
    """The complete effective model: parameters plus every operative modelling choice."""
    spec_id: str
    params: RaceParams
    target: str
    target_pit_lane_start: bool
    strategies: Mapping[str, Strategy]
    field: FieldPolicy
    pace_set: str
    sc_laps: tuple[int | None, ...]
    alternative_pace_sets: tuple[str, ...] = ()
    fixed: Mapping[str, Any] = dataclasses.field(default_factory=lambda: MappingProxyType({}))

    def with_params(self, params: RaceParams) -> "ModelSpec":
        return dataclasses.replace(self, params=validate_params(params))

    def with_strategies(self, strategies: Mapping[str, Strategy]) -> "ModelSpec":
        return dataclasses.replace(self, strategies=MappingProxyType(dict(strategies)))


def _get(node: dict[str, Any], key: str) -> Any:
    """Value of a {value, status, source, note} record."""
    if key not in node:
        raise KeyError(f"configuration key {key!r} is missing")
    return node[key]["value"]


def load_raw(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def config_fingerprint(path: str | Path) -> str:
    """SHA-256 of the configuration file bytes (says nothing about *when* the file was written)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _params(raw: dict[str, Any]) -> RaceParams:
    t, tr, sc, st = raw["tyres"], raw["traffic"], raw["safety_car"], raw["start"]
    compounds = MappingProxyType({
        name: Compound(name=name, offset_s=_get(t[name], "offset_s"), deg_s_per_lap=_get(t[name], "deg_s_per_lap"),
                       warmup_s=_get(t[name], "warmup_s"), max_life_laps=_get(t[name], "max_life_laps"))
        for name in ("HARD", "MEDIUM", "SOFT")})
    return validate_params(RaceParams(
        total_laps=_get(raw["race"], "total_laps"),
        reference_lap_s=_get(raw["race"], "reference_lap_time_s"),
        sc_lap_factor=_get(raw["race"], "sc_lap_time_factor"),
        fcy_lap_factor=_get(raw["race"], "fcy_lap_time_factor"),
        fuel_s_per_lap=_get(raw["fuel"], "effect_s_per_lap"),
        compounds=compounds,
        warmup_at_race_start=_get(t, "warmup_at_race_start"),
        warmup_policy=_get(t, "warmup_policy"),
        pit_loss_green_s=_get(raw["pit"], "loss_green_s"),
        pit_loss_sc_s=_get(raw["pit"], "loss_safety_car_s"),
        dirty_air_window_s=_get(tr, "dirty_air_window_s"),
        dirty_air_penalty_s=_get(tr, "dirty_air_penalty_s"),
        pass_window_s=_get(tr, "pass_window_s"),
        pass_proximity=_get(tr, "pass_proximity_measure"),
        pass_min_advantage_s=_get(tr, "pass_min_advantage_s"),
        pass_basis=_get(tr, "pass_advantage_basis"),
        pass_resolution=_get(tr, "pass_resolution"),
        duel_loss_attacker_s=_get(tr, "duel_loss_attacker_s"),
        duel_loss_defender_s=_get(tr, "duel_loss_defender_s"),
        min_follow_gap_s=_get(tr, "min_follow_gap_s"),
        tol=_get(tr, "decision_tolerance_s"),
        grid_slot_gap_s=_get(st, "grid_slot_gap_s"),
        pit_lane_start_gap_s=_get(st, "pit_lane_start_gap_s"),
        grid_lap1_loss_per_slot_s=_get(st, "grid_lap1_loss_per_slot_s"),
        sc_duration_laps=_get(sc, "duration_laps"),
        sc_eligibility=_get(sc, "eligibility"),
        sc_queue_gap_s=_get(sc, "queue_gap_s"),
        sc_opportunistic_stops=_get(sc, "opportunistic_stops"),
        sc_deployment_remainder=_get(sc, "deployment_remainder_pace"),
    ))


def load_params(path: str | Path) -> RaceParams:
    return _params(load_raw(path))


def _fixed(raw: dict[str, Any]) -> dict[str, Any]:
    """Read every FIXED choice and reject any value other than the implemented one."""
    out = {}
    for dotted, required in FIXED_CHOICES.items():
        section, key = dotted.split(".")
        value = _get(raw[section], key)
        if value != required:
            raise ValueError(f"{dotted}={value!r} is not implemented; the only supported value is {required!r}")
        out[dotted] = value
    return out


def load_spec(path: str | Path) -> ModelSpec:
    """Parse the whole YAML into the effective model specification."""
    raw = load_raw(path)
    ant = raw["antonelli"]
    strategies = {}
    for key, s in ant["strategies"].items():
        stops = tuple(PitStop(int(lap), str(comp)) for lap, comp in _get(s, "stops"))
        strategies[key] = Strategy(start_compound=_get(s, "start_compound"), stops=stops, label=s.get("label", key))
    f = raw["field"]
    window = _get(f, "pit_window_laps")
    if not (isinstance(window, list) and len(window) == 2 and 1 <= int(window[0]) <= int(window[1])):
        raise ValueError(f"field.pit_window_laps must be [first, last] with 1 <= first <= last, not {window!r}")
    sc_laps = tuple(None if x is None else int(x) for x in _get(raw["scenarios"], "safety_car_laps"))
    pace_set = _get(raw["pace"], "baseline_set")
    alternatives = tuple(_get(raw["pace"], "alternative_sets"))
    for name in (pace_set, *alternatives):
        if name not in KNOWN_PACE_SETS:
            raise ValueError(f"unknown pace set {name!r}; known: {KNOWN_PACE_SETS}")
    return ModelSpec(
        spec_id=raw["meta"]["id"], params=_params(raw), target=raw["meta"]["target_driver"],
        target_pit_lane_start=_get(ant, "pit_lane_start"), strategies=MappingProxyType(strategies),
        field=FieldPolicy(start_compound=_get(f, "start_compound"), next_compound=_get(f, "second_compound"),
                          window=(int(window[0]), int(window[1])), assignment=_get(f, "pit_lap_assignment")),
        pace_set=pace_set, sc_laps=sc_laps, alternative_pace_sets=alternatives,
        fixed=MappingProxyType(_fixed(raw)))


def variant(params: RaceParams, **changes: Any) -> RaceParams:
    """Return a validated copy of `params` with some fields changed (the baseline is never mutated).

    Compound fields are changed with keys of the form 'HARD__deg_s_per_lap'.
    """
    compound_changes = {k: v for k, v in changes.items() if "__" in k}
    plain = {k: v for k, v in changes.items() if "__" not in k}
    compounds = dict(params.compounds)
    for key, value in compound_changes.items():
        name, attr = key.split("__", 1)
        compounds[name] = dataclasses.replace(compounds[name], **{attr: value})
    return validate_params(dataclasses.replace(params, compounds=MappingProxyType(compounds), **plain))


def _canonical(obj: Any) -> Any:
    """JSON-serialisable canonical form of the specification objects."""
    if dataclasses.is_dataclass(obj):
        return {f.name: _canonical(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 12)
    return obj


def effective_fingerprint(spec: ModelSpec, pace: Mapping[str, float], field_laps: Mapping[str, int],
                          grid: pd.DataFrame) -> str:
    """SHA-256 of everything the simulator consumes: spec (incl. fixed choices and the alternative pace sets),
    pace vector, field pit laps and grid."""
    payload = {"spec": _canonical(spec), "pace": _canonical(dict(pace)), "field_laps": _canonical(dict(field_laps)),
               "grid": grid[["grid_slot", "driver"]].to_dict("records")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def code_fingerprint(paths: list[Path]) -> str:
    """SHA-256 over the source files of the simulator (sorted by name)."""
    h = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.name):
        h.update(path.name.encode() + b"\0" + path.read_bytes())
    return h.hexdigest()


def parameter_register(path: str | Path) -> pd.DataFrame:
    """Every {value, status, source, note} record in the YAML as one tidy table (role: operative | fixed)."""
    raw = load_raw(path)
    rows: list[dict[str, Any]] = []

    def walk(node: Any, prefix: str) -> None:
        if isinstance(node, dict) and "value" in node and "status" in node:
            rows.append({"parameter": prefix, "value": node["value"], "status": node["status"],
                         "role": node.get("role", "operative"), "source": node.get("source", ""),
                         "note": " ".join(str(node.get("note", "")).split())})
        elif isinstance(node, dict):
            for key, child in node.items():
                walk(child, f"{prefix}.{key}" if prefix else key)

    for section, node in raw.items():
        if section not in ("meta", "sources"):
            walk(node, section)
    return pd.DataFrame(rows)
