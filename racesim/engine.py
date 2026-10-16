"""The lap-wise race engine: 22 cars, 53 laps, deterministic.

The Safety-Car period is a window [t_d, t_r) of the shared race clock (see `safety_car`). Order of events inside
every lap k:
  1. Start-of-lap order = cars sorted by elapsed time at the line (ties keep the previous order). The cars split,
     front to back, into those whose lap k starts before t_d (racing), inside the window (neutralised) and after
     t_r (racing again).
  2. Racing cars: clean-air lap times (lap_model) and traffic, front to back (traffic.resolve_racing_lap). A lap
     still running at t_d is cut there: its remaining distance is covered at the full-course-yellow pace (order
     and distances kept), and an order change stands only if it was complete at t_d. Laps that start inside the
     window: full-course-yellow pace until t_r, racing pace after it, in the order they started, never closer
     than the queue spacing and never ahead of the Safety Car. Cars racing after t_r may not pass a car still on
     its restart lap.
  3. Every tyre ages by one lap; a lap that started at racing speed consumes a due warm-up.
  4. End-of-lap pit stops, priced by the race time at which the car reaches the end of its in-lap.
  5. Re-order; restore the minimum following gap around cars rejoining from the pits (only ever adds time).
  6. Record the lap.
t_d is the time at which the leader completes lap N in the race as run up to that instant. Everything before t_d
is green, so it is computed exactly by running the same engine without a Safety Car up to lap N.

Accounting identity (tested): T_i(k) = T_i(k-1) + lap_time_i(k) + pit_loss_i(k) + gap_adjust_i(k), with every term
non-negative, so elapsed time can never decrease.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

from .config import RaceParams, validate_params
from .lap_model import clean_air_lap_time
from .safety_car import SafetyCar, SafetyCarWindow, can_reach_flag, follow_gap, free_lap_end, lap_phase
from .strategies import Entry, PitStop
from .traffic import enforce_min_gap, resolve_racing_lap

N_GRID_SLOTS = 22


@dataclass
class CarState:
    entry: Entry
    race_time: float
    compound: str
    tyre_age: int = 0              # laps completed on the current set
    warmup_pending: bool = False   # the current set still owes its warm-up lap
    next_stop: int = 0             # index of the next planned stop in entry.strategy.stops
    stops_made: int = 0

    def planned_stop(self) -> PitStop | None:
        stops = self.entry.strategy.stops
        return stops[self.next_stop] if self.next_stop < len(stops) else None


@dataclass
class RaceResult:
    laps: pd.DataFrame            # one row per car per lap (empty if record=False)
    classification: pd.DataFrame  # final order
    events: pd.DataFrame          # passes, pit stops, Safety-Car events
    window: SafetyCarWindow | None = None                            # the Safety-Car window on the race clock
    first_crossing: dict[int, float] = field(default_factory=dict)  # earliest line crossing of each lap

    def position(self, code: str) -> int:
        return int(self.classification.loc[code, "position"])

    def gap_to_winner(self, code: str) -> float:
        return float(self.classification.loc[code, "gap_to_winner_s"])


def start_offset(entry: Entry, p: RaceParams) -> float:
    """Elapsed time at which the car first crosses the start line (ASSUMED headway, see config):
    grid-slot headway, or the last slot's headway plus the pit-lane release delay for a pit-lane start.
    Both components are validated to be non-negative (a pit-lane starter cannot leave before the last grid car)."""
    if entry.grid_slot is None:
        return (N_GRID_SLOTS - 1) * p.grid_slot_gap_s + p.pit_lane_start_gap_s
    return (entry.grid_slot - 1) * p.grid_slot_gap_s


def lap1_grid_loss(entry: Entry, p: RaceParams) -> float:
    """Optional first-lap time loss per grid slot (TUM's representation of the grid-position effect). A pit-lane
    starter pays the loss of the last slot; its release delay is part of the start offset. 0 in the baseline."""
    slot = N_GRID_SLOTS if entry.grid_slot is None else entry.grid_slot
    return (slot - 1) * p.grid_lap1_loss_per_slot_s


def simulate_race(entries: Iterable[Entry], p: RaceParams, sc: SafetyCar | None = None,
                  record: bool = True) -> RaceResult:
    """Run one race, optionally with a Safety Car."""
    validate_params(p)
    return _simulate(list(entries), p, sc, record)


def _decide_stop(car: CarState, lap: int, p: RaceParams, cheap: bool,
                   first_cheap: bool) -> tuple[PitStop | None, bool]:
    """Does `car` pit at the end of `lap`? `cheap` = it reaches the end of the lap inside the Safety-Car window;
    `first_cheap` = this is the first time it does so (used by the 'deployment_lap_only' policy)."""
    stop = car.planned_stop()
    if stop is None or lap >= p.total_laps:
        return None, False
    if stop.in_lap == lap:                                     # planned stop
        return stop, cheap
    if (cheap and p.sc_opportunistic_stops and car.stops_made == 0 and stop.in_lap > lap
            and can_reach_flag(p, stop.compound, lap)
            and (p.sc_eligibility == "every_cheap_lap" or first_cheap)):
        return stop, True                                      # Ruth's rule: take the cheap one if it can reach the flag
    return None, False


def _simulate(entries: list[Entry], p: RaceParams, sc: SafetyCar | None, record: bool,
                       until_lap: int | None = None) -> RaceResult:
    cars = {e.code: CarState(entry=e, race_time=start_offset(e, p), compound=e.strategy.start_compound,
                             warmup_pending=p.warmup_at_race_start)
            for e in entries}
    lap1_extra = {e.code: lap1_grid_loss(e, p) for e in entries} if p.grid_lap1_loss_per_slot_s > 0 else None
    window = None
    if sc is not None and sc.deploy_lap <= p.total_laps:
        # Everything before t_d is green: the same engine without a Safety Car gives t_d exactly.
        before = _simulate(entries, p, None, record=False, until_lap=sc.deploy_lap)
        window = SafetyCarWindow(sc.deploy_lap, sc.duration_laps, before.first_crossing[sc.deploy_lap], p.sc_lap_s)
    slot = {c: (cars[c].entry.grid_slot or N_GRID_SLOTS + 1) for c in cars}
    order = sorted(cars, key=lambda c: (cars[c].race_time, slot[c]))
    last_lap = p.total_laps if until_lap is None else min(until_lap, p.total_laps)

    rows: list[dict] = []
    events: list[dict] = []
    first_crossing: dict[int, float] = {}
    cheap_seen = {c: False for c in cars}

    for lap in range(1, last_lap + 1):
        start = {c: cars[c].race_time for c in order}
        tyre_at_start = {c: (cars[c].compound, cars[c].tyre_age) for c in order}
        dirty = {c: False for c in order}
        held = {c: False for c in order}
        attacking = {c: False for c in order}
        warm = {c: False for c in order}
        green_end = {c: math.nan for c in order}
        pushed: dict[str, float] = {}
        passes: list[tuple[str, str]] = []
        end: dict[str, float] = {}

        # ---- 1. split the field by where its lap starts on the shared clock --------------------------------------
        if window is None:
            pre, win, post = list(order), [], []
        else:
            pre = [c for c in order if start[c] < window.t_deploy]
            win = [c for c in order if window.t_deploy <= start[c] < window.t_restart]
            post = [c for c in order if start[c] >= window.t_restart]

        def race(group: list[str]):
            """Racing lap for `group` (traffic among these cars only, front to back)."""
            for c in group:
                warm[c] = cars[c].warmup_pending and (cars[c].tyre_age == 0 if p.warmup_policy == "first_lap_of_stint"
                                                      else True)
            clean = {c: clean_air_lap_time(p, cars[c].entry.pace_s, cars[c].compound, cars[c].tyre_age, warm[c], lap)
                     for c in group}
            out = resolve_racing_lap(group, start, clean, p, extra=lap1_extra if lap == 1 else None)
            for c in group:
                end[c] = green_end[c] = out.end_time[c]
                dirty[c], held[c], attacking[c] = out.dirty_air[c], out.held[c], out.attacking.get(c, False)
            pushed.update(out.pushed)
            return out

        # ---- 2a. laps that start before t_d: race, then cut at t_d ---------------------------------------------
        new_order: list[str] = []
        if pre:
            out = race(pre)
            new_order, lap_passes = list(out.order), list(out.passes)
            if window is not None and any(end[c] > window.t_deploy for c in pre):
                # Rest of the lap at the full-course-yellow pace. Every such car then moves at the same pace, so
                # distances (and the order at t_d) are kept: no overtaking, no closing in. The queue spacing is NOT
                # imposed at this line - the queue forms during the first full lap behind the Safety Car.
                for c in pre:
                    if end[c] > window.t_deploy and p.sc_deployment_remainder == "fcy_lap":
                        end[c] = free_lap_end(start[c], end[c] - start[c], window, p.fcy_lap_s)
                    elif end[c] > window.t_deploy:          # sensitivity option: factor x own remaining racing time
                        end[c] = window.t_deploy + p.fcy_lap_factor * (end[c] - window.t_deploy)
                new_order = sorted(new_order, key=lambda c: end[c])          # = running order at t_d
                lap_passes = [(a, v) for a, v in lap_passes if end[a] < end[v] - p.tol]   # complete before t_d
            passes.extend(lap_passes)

        # ---- 2b. laps that start inside the window: neutralised, then racing pace after t_r, no overtaking ------
        for c in win:
            tau = clean_air_lap_time(p, cars[c].entry.pace_s, cars[c].compound, cars[c].tyre_age, False, lap)
            free = free_lap_end(start[c], tau, window, p.fcy_lap_s)
            need = window.sc_line_time(lap)
            if new_order:
                ahead = new_order[-1]
                need = max(need, end[ahead] + follow_gap(end[ahead], window, p))
            end[c] = max(free, need)
            held[c] = need > free + p.tol
            new_order.append(c)

        # ---- 2c. laps that start after t_r: race; no pass on a car still on its restart lap ---------------------
        if post:
            out = race(post)
            post_order = list(out.order)
            if new_order and new_order[-1] in win:
                lead = new_order[-1]
                chain = [lead] + post_order
                times = {c: end[c] for c in chain}
                for c in enforce_min_gap(chain, times, p.min_follow_gap_s, p.tol, rejoined={lead}):
                    end[c], held[c] = times[c], True
            new_order += post_order
            passes.extend(out.passes)

        for attacker, victim in passes:
            events.append({"lap": lap, "type": "pass", "car": attacker, "other": victim})
        if window is not None and lap == window.deploy_lap:
            events.append({"lap": lap, "type": "sc_deployed", "car": new_order[0], "other": f"t={window.t_deploy:.3f}"})
        if window is not None and lap == window.deploy_lap + window.duration_laps:
            events.append({"lap": lap, "type": "sc_restart", "car": "", "other": f"t={window.t_restart:.3f}"})
        first_crossing[lap] = min(end.values())
        lap_time = {c: end[c] - start[c] for c in new_order}

        # ---- 3. tyre ageing and warm-up bookkeeping -------------------------------------------------------------
        racing = set(pre) | set(post)
        for c in new_order:
            cars[c].race_time = end[c]
            cars[c].tyre_age += 1
            if p.warmup_policy == "first_lap_of_stint" or c in racing:
                cars[c].warmup_pending = False      # a lap started at racing speed consumes a due warm-up

        # ---- 4. pit stops, priced by the race time at which the in-lap ends ------------------------------------
        pit_loss = {c: 0.0 for c in new_order}
        for c in new_order:
            cheap = window is not None and window.neutral(end[c])
            first_cheap = cheap and not cheap_seen[c]
            cheap_seen[c] = cheap_seen[c] or cheap
            stop, at_sc_price = _decide_stop(cars[c], lap, p, cheap, first_cheap)
            if stop is None:
                continue
            loss = p.pit_loss_sc_s if at_sc_price else p.pit_loss_green_s
            events.append({"lap": lap, "type": "pit", "car": c, "other": f"{cars[c].compound}->{stop.compound}",
                           "loss_s": loss, "safety_car_price": at_sc_price, "planned_in_lap": stop.in_lap,
                           "pit_entry_s": end[c]})
            car = cars[c]
            car.race_time += loss
            car.compound, car.tyre_age, car.warmup_pending = stop.compound, 0, True
            car.next_stop += 1
            car.stops_made += 1
            pit_loss[c] = loss

        # ---- 5. re-order; minimum gap after pit stops -----------------------------------------------------------
        order = sorted(new_order, key=lambda c: cars[c].race_time)
        times = {c: cars[c].race_time for c in order}
        rejoined = None if p.pass_resolution != "time_crossing" else {c for c in order if pit_loss[c] > 0}
        gap_adjust = enforce_min_gap(order, times, p.min_follow_gap_s, p.tol, rejoined=rejoined)
        for c in order:
            cars[c].race_time = times[c]

        # ---- 6. record ----------------------------------------------------------------------------------------
        if record:
            leader_time = cars[order[0]].race_time
            passes_made = {c: 0 for c in order}
            times_passed = {c: 0 for c in order}
            for attacker, victim in passes:
                passes_made[attacker] += 1
                times_passed[victim] += 1
            for pos, c in enumerate(order, start=1):
                rows.append({
                    "lap": lap, "car": c, "position": pos, "race_time_s": cars[c].race_time,
                    "lap_time_s": lap_time[c], "gap_to_leader_s": cars[c].race_time - leader_time,
                    "gap_ahead_s": cars[c].race_time - cars[order[pos - 2]].race_time if pos > 1 else 0.0,
                    "compound": tyre_at_start[c][0], "tyre_age": tyre_at_start[c][1], "warmup_applied": warm[c],
                    "dirty_air": dirty[c], "held_up": held[c], "attacking": attacking[c],
                    "passes_made": passes_made[c], "times_passed": times_passed[c],
                    "overtaken_loss_s": pushed.get(c, 0.0), "pit_loss_s": pit_loss[c],
                    "gap_adjust_s": gap_adjust.get(c, 0.0),
                    "track_status": lap_phase(start[c], end[c], window),
                    "lap_start_s": start[c], "line_time_s": end[c], "green_end_s": green_end[c],
                    "neutral_s": window.neutral_overlap(start[c], end[c]) if window is not None else 0.0,
                })

    winner_time = cars[order[0]].race_time
    classification = pd.DataFrame(
        [{"car": c, "position": i, "race_time_s": cars[c].race_time,
          "gap_to_winner_s": cars[c].race_time - winner_time,
          "grid_slot": cars[c].entry.grid_slot, "pace_s": cars[c].entry.pace_s,
          "stops": cars[c].stops_made, "final_compound": cars[c].compound}
         for i, c in enumerate(order, start=1)]).set_index("car")
    return RaceResult(laps=pd.DataFrame(rows), classification=classification, events=pd.DataFrame(events),
                      window=window, first_crossing=first_crossing)
