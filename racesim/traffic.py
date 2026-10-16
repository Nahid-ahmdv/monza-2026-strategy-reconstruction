"""Car-to-car interaction on a green-flag lap: dirty air, overtaking, being held up.

Ruth's rules:
  * a car within 2 s of the car ahead loses 0.25 s/lap (dirty air);
  * a car passes when it is at least 0.35 s/lap faster and within 1.2 s;
  * otherwise it 'sits at the pace of the car ahead'.

Cars are processed front to back in their start-of-lap order (as in Heilmeier et al. 2018 / TUM), so every
car ahead of the current one already has its final end-of-lap time.

Pass resolution (how a permitted pass is turned into end-of-lap times):
  time_crossing      (baseline) a car that is eligible to pass is not held; it takes the position only when its
                     own elapsed time is earlier than the other car's. No time is created or transferred, so
                     every order change is time-feasible. A car that is not eligible is held at the minimum gap.
  passed_car_yields  (sensitivity) instant pass; the pass is declared at once and the passed car is pushed behind
                     the attacker.
  passer_jumps       (sensitivity) instant pass; the attacker's time is moved just ahead of the passed car (a tow
                     gain).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import RaceParams


@dataclass
class LapOutcome:
    order: list[str]                     # end-of-lap running order
    end_time: dict[str, float]           # end-of-lap cumulative race time
    dirty_air: dict[str, bool]
    held: dict[str, bool]                # lap time increased by the car directly ahead (held at the gap, or blocked)
    attacking: dict[str, bool] = field(default_factory=dict)   # eligible attacker closer than the min gap
    passes: list[tuple[str, str]] = field(default_factory=list)  # (attacker, passed car)
    pushed: dict[str, float] = field(default_factory=dict)      # time added to passed cars (instant-pass rules, duel loss)


def _dirty_air(order: list[str], start_time: dict[str, float], clean_lap: dict[str, float], p: RaceParams,
               extra: dict[str, float] | None = None):
    """Dirty air depends on the gap to the car directly ahead at the START of the lap. `extra` is a time loss
    that is not part of the car's intrinsic pace (the optional lap-1 grid loss); it slows the car like dirty air
    and does not enter the 'faster by 0.35 s' comparison."""
    dirty, eff = {}, {}
    for i, car in enumerate(order):
        gap_ahead = start_time[car] - start_time[order[i - 1]] if i > 0 else float("inf")
        dirty[car] = gap_ahead <= p.dirty_air_window_s + p.tol
        eff[car] = clean_lap[car] + (p.dirty_air_penalty_s if dirty[car] else 0.0)
        if extra:
            eff[car] += extra.get(car, 0.0)
    return dirty, eff


def eligible_to_pass(p: RaceParams, car: str, ahead: str, start_time: dict[str, float], clean_lap: dict[str, float],
                     eff_lap: dict[str, float], end: dict[str, float], projected: float) -> bool:
    """Ruth's rule: within the pass window during the lap AND faster by at least the threshold."""
    start_gap = start_time[car] - start_time[ahead]
    end_gap = projected - end[ahead]
    proximity = start_gap if p.pass_proximity == "start_of_lap" else min(start_gap, end_gap)
    if p.pass_basis == "intrinsic":
        advantage = clean_lap[ahead] - clean_lap[car]
    else:  # "actual": attacker's dirty-air lap vs the realised lap of the car ahead
        advantage = (end[ahead] - start_time[ahead]) - eff_lap[car]
    return proximity <= p.pass_window_s + p.tol and advantage >= p.pass_min_advantage_s - p.tol


def resolve_racing_lap(order: list[str], start_time: dict[str, float], clean_lap: dict[str, float],
                       p: RaceParams, extra: dict[str, float] | None = None) -> LapOutcome:
    """Resolve one green-flag lap for the cars in `order` (see module docstring)."""
    dirty, eff = _dirty_air(order, start_time, clean_lap, p, extra)
    if p.pass_resolution == "time_crossing":
        return _time_crossing(order, start_time, clean_lap, dirty, eff, p)
    return _instant_pass(order, start_time, clean_lap, dirty, eff, p)


def _time_crossing(order, start_time, clean_lap, dirty, eff, p) -> LapOutcome:
    new_order: list[str] = []
    end: dict[str, float] = {}
    held = {car: False for car in order}
    attacking = {car: False for car in order}
    passes: list[tuple[str, str]] = []
    pushed: dict[str, float] = {}

    for car in order:
        projected = start_time[car] + eff[car]
        pos, passed = len(new_order), []
        # 1) Pass every car ahead whose end time the attacker beats, as long as it is eligible to pass it.
        while pos > 0:
            ahead = new_order[pos - 1]
            duel = p.duel_loss_attacker_s if not passed else 0.0
            if (projected + duel < end[ahead] - p.tol
                    and eligible_to_pass(p, car, ahead, start_time, clean_lap, eff, end, projected)):
                projected += duel
                passed.append(ahead)
                pos -= 1
            else:
                break
        # 2) Final time behind the car now directly ahead: an ineligible follower is held at the minimum gap.
        #    A pass stands only if the attacker's FINAL time is still strictly earlier than the passed car's;
        #    otherwise (both end up held behind the same car) the pass is unwound, frontmost first.
        while True:
            t, is_held = projected, False
            if pos > 0:
                ahead = new_order[pos - 1]
                if eligible_to_pass(p, car, ahead, start_time, clean_lap, eff, end, projected):
                    t = max(projected, end[ahead])           # still behind (not crossed): no time is added
                else:
                    floor = end[ahead] + p.min_follow_gap_s
                    if projected < floor - p.tol:
                        t, is_held = floor, True
            if passed and t >= end[passed[-1]] - p.tol:
                passed.pop()
                pos += 1
                if not passed:
                    projected -= p.duel_loss_attacker_s
                continue
            break
        held[car] = is_held or t > projected + p.tol      # lap time increased by the car ahead (held or blocked)
        if pos > 0:
            attacking[car] = (not is_held) and t - end[new_order[pos - 1]] < p.min_follow_gap_s - p.tol
        passes.extend((car, victim) for victim in passed)
        new_order.insert(pos, car)
        end[car] = t
        # 3) Optional duel loss for the cars that were passed (Heilmeier's t_overtake,lose); keep the order valid.
        if passed and p.duel_loss_defender_s > 0:
            for victim in passed:
                end[victim] += p.duel_loss_defender_s
                pushed[victim] = pushed.get(victim, 0.0) + p.duel_loss_defender_s
            tail = sorted(new_order[pos + 1:], key=lambda c: end[c])
            new_order[pos + 1:] = tail
    return LapOutcome(order=new_order, end_time=end, dirty_air=dirty, held=held, attacking=attacking,
                      passes=passes, pushed=pushed)


def _instant_pass(order, start_time, clean_lap, dirty, eff, p) -> LapOutcome:
    """Instant-pass readings ('passed_car_yields', 'passer_jumps'), used as sensitivity variants."""
    new_order: list[str] = []
    end: dict[str, float] = {}
    held = {car: False for car in order}
    passes: list[tuple[str, str]] = []
    pushed: dict[str, float] = {}

    for car in order:
        projected = start_time[car] + eff[car]
        pos = len(new_order)
        while pos > 0:
            ahead = new_order[pos - 1]
            if eligible_to_pass(p, car, ahead, start_time, clean_lap, eff, end, projected):
                passes.append((car, ahead))
                pos -= 1
            else:
                break
        t = projected
        if pos > 0:
            floor = end[new_order[pos - 1]] + p.min_follow_gap_s
            if t < floor - p.tol:
                t, held[car] = floor, True
        if p.pass_resolution == "passer_jumps" and pos < len(new_order):
            t = max(min(t, end[new_order[pos]] - p.min_follow_gap_s),
                    end[new_order[pos - 1]] + p.min_follow_gap_s if pos > 0 else float("-inf"))
        new_order.insert(pos, car)
        end[car] = t
        for j in range(pos + 1, len(new_order)):
            need = end[new_order[j - 1]] + p.min_follow_gap_s
            if end[new_order[j]] < need - p.tol:
                pushed[new_order[j]] = pushed.get(new_order[j], 0.0) + need - end[new_order[j]]
                end[new_order[j]] = need
    return LapOutcome(order=new_order, end_time=end, dirty_air=dirty, held=held,
                      attacking={c: False for c in order}, passes=passes, pushed=pushed)


def enforce_min_gap(order: list[str], times: dict[str, float], gap: float, tol: float = 1e-9,
                    rejoined: set[str] | None = None) -> dict[str, float]:
    """Restore the minimum gap around cars that rejoined from the pits (Heilmeier: the minimum time gap must
    also be kept after the pit stops have been completed).

    Only pairs involving a car that pitted this lap - or a car that had to back off because of one - are
    checked, so a car legitimately attacking closer than the minimum gap on track is not pushed back.
    `rejoined=None` checks every pair (used with the instant-pass readings, whose traffic never leaves a car
    closer than the minimum gap). Only ever ADDS time; returns the time added.
    """
    added: dict[str, float] = {}
    affected = set(order) if rejoined is None else set(rejoined)
    for prev, car in zip(order, order[1:]):
        if prev not in affected and car not in affected:
            continue
        need = times[prev] + gap
        if times[car] < need - tol:
            added[car] = need - times[car]
            times[car] = need
            affected.add(car)
    return added
