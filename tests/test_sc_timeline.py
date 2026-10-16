"""Safety Car on the shared race clock: focused boundary cases with HAND-DERIVED expectations.

Common set-up for the hand calculations (unless a test says otherwise): no fuel effect, no degradation, no warm-up,
no dirty air, every car on hards with pace 0, so every racing lap takes exactly t_ref = 84.0 s; a grid slot is 1 s
of starting headway. Then:
    L_sc  = 1.6 x 84 = 134.4 s     (Safety-Car lap)          L_fcy = 1.4 x 84 = 117.6 s (full-course-yellow lap)
    SC on lap 5:  t_d = 5 x 84 = 420.0 s (leader completes lap 5),  t_r = 420 + 3 x 134.4 = 823.2 s.
Distance argument used throughout: inside the window a car not yet in the queue gains on the Safety Car
403.2 x (1/117.6 - 1/134.4) = 3/7 lap = 36.0 s at 84 s/lap.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import car, check_passes_time_feasible, check_physical, check_timeline, no_stop, one_stop
from racesim.config import variant
from racesim.engine import simulate_race
from racesim.safety_car import SafetyCar, SafetyCarWindow, can_reach_flag, follow_gap, free_lap_end

T_D, T_R = 420.0, 823.2


@pytest.fixture
def clock(params):
    """84.0 s racing laps for pace 0 on hards, 1 s headway per grid slot, no dirty air."""
    return variant(params, fuel_s_per_lap=0.0, HARD__deg_s_per_lap=0.0, HARD__warmup_s=0.0, MEDIUM__deg_s_per_lap=0.0,
                   SOFT__deg_s_per_lap=0.0, dirty_air_penalty_s=0.0, grid_slot_gap_s=1.0)


def line(res, code: str, lap: int) -> float:
    L = res.laps
    return float(L[(L.car == code) & (L.lap == lap)]["line_time_s"].iloc[0])


def status(res, code: str) -> list[str]:
    L = res.laps
    return list(L[L.car == code].sort_values("lap")["track_status"])


def gap_at_flag(res, code: str, ref: str = "A") -> float:
    c = res.classification
    return float(c.loc[code, "race_time_s"] - c.loc[ref, "race_time_s"])


# ---- building blocks -------------------------------------------------------------------------------------------

def test_free_lap_end_by_hand():
    w = SafetyCarWindow(deploy_lap=5, duration_laps=3, t_deploy=T_D, sc_lap_s=134.4)
    assert free_lap_end(100.0, 84.0, w, 117.6) == pytest.approx(184.0)                  # entirely green
    assert free_lap_end(T_D - 42.0, 84.0, w, 117.6) == pytest.approx(T_D + 58.8)        # half a lap left at t_d
    assert free_lap_end(T_D + 10.0, 84.0, w, 117.6) == pytest.approx(T_D + 127.6)       # entirely neutralised
    assert free_lap_end(T_R - 58.8, 84.0, w, 117.6) == pytest.approx(T_R + 42.0)        # half a lap left at t_r
    short = SafetyCarWindow(deploy_lap=5, duration_laps=1, t_deploy=100.0, sc_lap_s=20.0)  # window [100, 120)
    # 0.5 lap green by t=100, + 20/117.6 lap in the window, remaining (0.5 - 20/117.6) x 84 = 27.714286 s
    assert free_lap_end(58.0, 84.0, short, 117.6) == pytest.approx(120.0 + (0.5 - 20 / 117.6) * 84.0)


def test_follow_gap_is_one_spacing_and_continuous(params):
    """0.8 s at Safety-Car pace and 0.5 s at racing pace are the same distance; the time needed by the car ahead
    to cover it is interpolated when it changes pace within that spacing."""
    w = SafetyCarWindow(deploy_lap=5, duration_laps=3, t_deploy=T_D, sc_lap_s=134.4)
    assert 0.8 / 134.4 == pytest.approx(0.5 / 84.0)
    expected = {T_D - 1.0: 0.5, T_D - 0.5: 0.5, T_D - 0.25: 0.25 + 0.5 * 0.8, T_D: 0.8, T_D + 100: 0.8,
                T_R - 0.8: 0.8, T_R - 0.4: 0.4 + 0.5 * 0.5, T_R: 0.5, T_R + 5: 0.5}
    for t, g in expected.items():
        assert follow_gap(t, w, params) == pytest.approx(g), t
    ts = np.linspace(T_D - 2, T_D + 2, 4001).tolist() + np.linspace(T_R - 2, T_R + 2, 4001).tolist()
    vals = np.array([follow_gap(t, w, params) for t in ts])
    assert np.abs(np.diff(vals)).max() < 2e-3                                              # no jump


# ---- 1. two close cars, deployment mid-lap, restart -------------------------------------------------------------

def test_two_close_cars_by_hand(params):
    """L (pace 0, slot 1) and F (pace 0.2, 0.6 s behind, dirty air 0.25 s); SC on lap 2.
    Lap 1: F = 0.6 + 84.45 = 85.05. Lap 2: L = 168 = t_d; F would end at 85.05 + 84.45 = 169.5, i.e. 1.5/84.45 of
    a lap from the line at t_d, covered at 117.6 s/lap: 168 + 1.5 x 117.6/84.45 = 170.088810.
    Laps 3-5: L on the Safety Car's line times 302.4, 436.8, 571.2 (= t_r); F 0.8 s behind, and 0.5 s behind at the
    restart line (one spacing at racing pace). Lap 6 racing: 655.2 and 571.7 + 84.45 = 656.15."""
    p = variant(params, fuel_s_per_lap=0.0, HARD__deg_s_per_lap=0.0, HARD__warmup_s=0.0, grid_slot_gap_s=0.6)
    res = simulate_race([car("L", 0.0, 1), car("F", 0.2, 2)], p, SafetyCar(2))
    assert res.window.t_deploy == pytest.approx(168.0) and res.window.t_restart == pytest.approx(571.2)
    assert line(res, "F", 1) == pytest.approx(85.05)
    L = res.laps.set_index(["car", "lap"])
    assert L.loc[("F", 2), "green_end_s"] == pytest.approx(169.5)
    assert line(res, "F", 2) == pytest.approx(168 + 1.5 * 117.6 / 84.45)
    assert [line(res, "L", k) for k in (3, 4, 5, 6)] == pytest.approx([302.4, 436.8, 571.2, 655.2])
    assert [line(res, "F", k) for k in (3, 4, 5, 6)] == pytest.approx([303.2, 437.6, 571.7, 656.15])
    assert status(res, "L")[:6] == ["GREEN", "SC_DEPLOYED", "SC", "SC", "SC", "GREEN"]
    assert status(res, "F")[:6] == ["GREEN", "SC_DEPLOYED", "SC", "SC", "SC_RESTART", "GREEN"]
    for c in "LF":
        assert res.laps[res.laps.car == c]["neutral_s"].sum() == pytest.approx(403.2)
    check_physical(res, p)
    check_timeline(res, p)


def test_close_cars_are_not_spread_at_the_deployment_line_by_hand(clock):
    """SC on lap 1. B (pace 0) starts 0.3 s behind A (pace 0.5) and passes it on lap 1: B 84.3 = t_d. A is then
    0.2/84.5 of a lap from the line and covers it at 117.6 s/lap: 84.3 + 0.2 x 117.6/84.5 = 84.578343 - only
    0.278 s behind, NOT pushed to the 0.8 s queue spacing (the queue has not formed yet; distances are kept).
    Lap 2 is the first full lap behind the Safety Car: B 84.3 + 134.4 = 218.7, A one spacing behind, 219.5."""
    p = variant(clock, grid_slot_gap_s=0.3)
    res = simulate_race([car("A", 0.5, 1), car("B", 0.0, 2)], p, SafetyCar(1))
    assert res.window.t_deploy == pytest.approx(84.3)
    assert line(res, "A", 1) == pytest.approx(84.3 + 0.2 * 117.6 / 84.5)
    assert [line(res, "B", 2), line(res, "A", 2)] == pytest.approx([218.7, 219.5])
    passes = res.events[res.events.type == "pass"]
    assert list(zip(passes.lap, passes.car, passes.other)) == [(1, "B", "A")]          # complete at t_d: stands
    check_physical(res, p)
    check_timeline(res, p)
    check_passes_time_feasible(res)


# ---- 2. several gaps: finite catch-up, some cars never reach the queue -------------------------------------------

def test_multiple_gaps_catch_up_by_hand(clock):
    """Gaps at the start: B 1, C 10, D 40, E 70 s. Lap 5 is cut at t_d = 420: a car x s behind finishes it at
    420 + 1.4x. C reaches the queue on lap 6; D and E never do and restart mid-lap (at t_r) with
    40 - 36 = 4.0 s and 70 - 36 = 34.0 s. Queue cars restart one spacing (0.5 s) apart."""
    entries = [car("A", 0.0, 1), car("B", 0.0, 2), car("C", 0.0, 11), car("D", 0.0, 41), car("E", 0.0, 71)]
    res = simulate_race(entries, clock, SafetyCar(5))
    expected = {5: [420.0, 421.4, 434.0, 476.0, 518.0],
                6: [554.4, 555.2, 556.0, 593.6, 635.6],
                7: [688.8, 689.6, 690.4, 711.2, 753.2],
                8: [823.2, 823.7, 824.2, 827.2, 857.2],
                9: [907.2, 907.7, 908.2, 911.2, 941.2]}
    for lap, times in expected.items():
        assert [line(res, c, lap) for c in "ABCDE"] == pytest.approx(times), lap
    assert [gap_at_flag(res, c) for c in "BCDE"] == pytest.approx([0.5, 1.0, 4.0, 34.0])
    L = res.laps.set_index(["car", "lap"])
    assert bool(L.loc[("C", 6), "held_up"]) and not L.loc[("D", 6), "held_up"]          # C caught, D not
    check_physical(res, clock)
    check_timeline(res, clock)


# ---- 3. cars more than one lap apart; restart with cars on different laps ----------------------------------------

def test_lapped_cars_use_the_same_clock_by_hand(clock):
    """X starts 100 s and Y 150 s behind A (more than a lap). The deployment cuts X's and Y's LAP 4 (not lap 5):
    X: 352 -> 436 green becomes 420 + 16 x 1.4 = 442.4; Y: 402 -> 486 becomes 420 + 66 x 1.4 = 512.4.
    Then neutralised laps of 117.6 s. At t_r = 823.2 X is on lap 8 and Y on lap 7:
    X lap 8 from 795.2: 28/117.6 lap by t_r, rest 64.0 s at racing pace -> 887.2;
    Y lap 7 from 747.6: 75.6/117.6 lap by t_r, rest 30.0 s -> 853.2, and Y's lap 8 is a racing lap (937.2)
    although A's lap 8 was a Safety-Car lap. Both gain exactly 36.0 s, like any car that misses the queue."""
    res = simulate_race([car("A", 0.0, 1), car("X", 0.0, 101), car("Y", 0.0, 151)], clock, SafetyCar(5))
    assert [line(res, "X", k) for k in range(4, 10)] == pytest.approx([442.4, 560.0, 677.6, 795.2, 887.2, 971.2])
    assert [line(res, "Y", k) for k in range(4, 9)] == pytest.approx([512.4, 630.0, 747.6, 853.2, 937.2])
    assert status(res, "X")[3:9] == ["SC_DEPLOYED", "SC", "SC", "SC", "SC_RESTART", "GREEN"]
    assert status(res, "Y")[3:8] == ["SC_DEPLOYED", "SC", "SC", "SC_RESTART", "GREEN"]
    lap8 = res.laps[res.laps.lap == 8].set_index("car")["track_status"].to_dict()
    assert lap8 == {"A": "SC", "X": "SC_RESTART", "Y": "GREEN"}                           # same lap, three phases
    assert gap_at_flag(res, "X") == pytest.approx(64.0) and gap_at_flag(res, "Y") == pytest.approx(114.0)
    for c in "AXY":
        assert res.laps[res.laps.car == c]["neutral_s"].sum() == pytest.approx(403.2)
    check_physical(res, clock)
    check_timeline(res, clock)


def test_own_lap_deployment_option(params):
    """Sensitivity option 'own_lap_scaled': the rest of the cut lap takes 1.4 x the car's own remaining racing time
    instead of the driver-independent 117.6 s/lap. Two-car case of test_two_close_cars_by_hand:
    F's lap 2 = 168 + 1.4 x 1.5 = 170.1 instead of 170.088810; everything else unchanged."""
    p = variant(params, fuel_s_per_lap=0.0, HARD__deg_s_per_lap=0.0, HARD__warmup_s=0.0, grid_slot_gap_s=0.6,
                sc_deployment_remainder="own_lap_scaled")
    res = simulate_race([car("L", 0.0, 1), car("F", 0.2, 2)], p, SafetyCar(2))
    assert line(res, "F", 2) == pytest.approx(170.1)
    assert [line(res, "F", k) for k in (3, 4, 5)] == pytest.approx([303.2, 437.6, 571.7])
    check_timeline(res, p)


def test_outcome_depends_on_race_time_not_lap_index(clock):
    """A car starting s seconds behind (20 <= s <= 200, crossing several lap boundaries relative to t_d) finishes
    max(s - 36, 0.5) s behind: continuous and monotonic in s, with no step where its lap index changes."""
    for s in np.arange(20.0, 200.01, 0.5):
        p = variant(clock, grid_slot_gap_s=float(s))
        res = simulate_race([car("A", 0.0, 1), car("X", 0.0, 2)], p, SafetyCar(5), record=False)
        assert gap_at_flag(res, "X") == pytest.approx(max(s - 36.0, 0.5), abs=1e-6), s


def test_pass_in_progress_at_deployment_is_cancelled_and_restart_lap_car_is_protected(clock):
    """Two lapped cars behind leader A (t_d = 420, t_r = 823.2): W (pace 3.0, 87 s laps, 195.5 s back) and P
    (pace 0, 84 s laps, 204.2 s back), so P gains 3 s/lap.
    Lap 3 (cut by t_d): racing would put P ahead at the line (456.2 < 456.5), but at t_d W has 36.5/87 of the lap
    left and P 36.2/84 (more), so W is still ahead: the pass is cancelled. W: 420 + 36.5/87 x 117.6 = 469.337931,
    P: 420 + 36.2/84 x 117.6 = 470.68. Laps 4-5 at 117.6 s. Lap 6: W 822.137931 (< t_r); P restarts in that lap:
    705.88 + 117.32 s neutralised + 0.2 s racing = 823.4. Lap 7: W is on its restart lap (starts in the window):
    823.2 + (1 - 1.062069/117.6) x 87 = 909.414286; P starts after t_r and would finish 907.4, but may not pass a
    car still on its restart lap: held to 909.914286."""
    p = variant(clock, grid_slot_gap_s=0.1)
    entries = [car("A", 0.0, 1), car("W", 3.0, 1956), car("P", 0.0, 2043)]
    res = simulate_race(entries, p, SafetyCar(5))
    assert res.window.t_deploy == pytest.approx(420.0)
    L = res.laps.set_index(["car", "lap"])
    assert L.loc[("P", 3), "green_end_s"] < L.loc[("W", 3), "green_end_s"]              # racing alone: P ahead
    assert line(res, "W", 3) == pytest.approx(420 + 36.5 / 87 * 117.6)
    assert line(res, "P", 3) == pytest.approx(420 + 36.2 / 84 * 117.6)
    assert res.events[(res.events.type == "pass") & (res.events.lap == 3)].empty
    assert line(res, "W", 6) == pytest.approx(420 + 36.5 / 87 * 117.6 + 3 * 117.6)
    assert line(res, "P", 6) == pytest.approx(823.4)
    w7 = 823.2 + (1 - (823.2 - line(res, "W", 6)) / 117.6) * 87.0
    assert line(res, "W", 7) == pytest.approx(909.414286, abs=1e-6) and w7 == pytest.approx(909.414286, abs=1e-6)
    assert line(res, "P", 7) == pytest.approx(w7 + 0.5) and bool(L.loc[("P", 7), "held_up"])
    assert (L.loc[("W", 7), "track_status"], L.loc[("P", 7), "track_status"]) == ("SC_RESTART", "GREEN")
    check_physical(res, p)
    check_timeline(res, p)
    check_passes_time_feasible(res)


def test_first_neutralised_car_follows_the_car_actually_ahead_at_the_line(clock):
    """t_d = 420 (A). Three lapped cars: V is held 0.5 s behind U from lap 1; R (0.5 s/lap faster) passes V on lap 4.
    Lap 4 ends at 419.6 (U) and 419.9 (R), before t_d; V's would end at 420.1, so the deployment cuts V's LAP 4:
    420 + 0.1 x 1.4 = 420.14 (a first hand calculation missed this and expected 420.1).
    Lap 5: racing alone, R would pass U, but only at t = 470 > t_d, so U stays ahead: U 420 + 83.6/84 x 117.6 =
    537.04, R 420 + 83.4/83.5 x 117.6 = 537.459162. V's lap 5 starts inside the window: free 420.14 + 117.6 =
    537.74, but it must stay one queue spacing behind the car really ahead at the line (R, not U): 538.259162.
    After the restart (lap 8: U 870.8, R one racing-pace spacing behind, 871.3) R, 0.5 s/lap faster, ends lap 9
    exactly alongside U (954.8; a pass needs a strictly earlier time) and passes it on lap 10 (1038.3 < 1038.8)."""
    p = variant(clock, grid_slot_gap_s=0.1)
    entries = [car("A", 0.0, 1), car("U", 0.0, 837), car("V", 0.0, 841), car("R", -0.5, 860)]
    res = simulate_race(entries, p, SafetyCar(5))
    assert [line(res, c, 4) for c in ("U", "R", "V")] == pytest.approx([419.6, 419.9, 420.0 + 0.1 * 1.4])
    assert line(res, "U", 5) == pytest.approx(420 + 83.6 / 84 * 117.6)
    assert line(res, "R", 5) == pytest.approx(420 + 83.4 / 83.5 * 117.6)
    assert line(res, "V", 5) == pytest.approx(420 + 83.4 / 83.5 * 117.6 + 0.8)
    passes = res.events[res.events.type == "pass"]
    assert list(zip(passes.lap, passes.car, passes.other)) == [(4, "R", "V"), (10, "R", "U")]
    assert [line(res, "U", 8), line(res, "R", 8)] == pytest.approx([870.8, 871.3])
    assert [line(res, "U", 9), line(res, "R", 9)] == pytest.approx([954.8, 954.8])
    check_physical(res, p)
    check_timeline(res, p)
    check_passes_time_feasible(res)


# ---- 4. pit stops just before deployment, during the window and just after the restart --------------------------

def test_pit_price_follows_the_race_clock_by_hand(clock):
    """Planned stops (Ruth's opportunistic rule off). In-lap end times and prices, derived by hand:
      P1 (83 s back, in-lap 4) 419.0 < t_d          -> 22 s   just before the deployment
      P2 (85 s back, in-lap 4) 420 + 1.4 = 421.4    -> 11 s   same LAP as P1, other side of t_d
      P3 (2 s back, in-lap 6)  queued, 556.0        -> 11 s   during the neutralisation
      P6 (100 s back, in-lap 7) 795.2 < t_r         -> 11 s
      P5 (150 s back, in-lap 7) 853.2 > t_r         -> 22 s   same LAP as P6, after the restart
      A (leader, in-lap 8) exactly t_r = 823.2      -> 22 s   the Safety Car has gone in
      Q (1 s back, in-lap 8) 823.7                  -> 22 s   just after the restart
    A rule by lap number (laps 5, 6, 7 cheap) would instead charge P2 22 s and P5 11 s."""
    p = variant(clock, sc_opportunistic_stops=False)
    entries = [car("A", 0.0, 1, one_stop("HARD", 8, "HARD")), car("Q", 0.0, 2, one_stop("HARD", 8, "HARD")),
               car("P3", 0.0, 3, one_stop("HARD", 6, "HARD")), car("P1", 0.0, 84, one_stop("HARD", 4, "HARD")),
               car("P2", 0.0, 86, one_stop("HARD", 4, "HARD")), car("P6", 0.0, 101, one_stop("HARD", 7, "HARD")),
               car("P5", 0.0, 151, one_stop("HARD", 7, "HARD"))]
    res = simulate_race(entries, p, SafetyCar(5))
    pits = res.events[res.events.type == "pit"].set_index("car")
    expected = {"P1": (4, 419.0, 22.0), "P2": (4, 421.4, 11.0), "P3": (6, 556.0, 11.0), "P6": (7, 795.2, 11.0),
                "P5": (7, 853.2, 22.0), "A": (8, 823.2, 22.0), "Q": (8, 823.7, 22.0)}
    for c, (lap, entry, loss) in expected.items():
        assert int(pits.loc[c, "lap"]) == lap, c
        assert float(pits.loc[c, "pit_entry_s"]) == pytest.approx(entry), c
        assert float(pits.loc[c, "loss_s"]) == loss, c
    check_physical(res, p)
    check_timeline(res, p)


@pytest.mark.parametrize("policy, life, lap, loss", [("every_cheap_lap", 45, 36, 22.0), ("every_cheap_lap", 46, 7, 11.0),
                                                     ("every_cheap_lap", 48, 5, 11.0), ("deployment_lap_only", 46, 36, 22.0)])
def test_ruths_rule_at_each_window_crossing(params, policy, life, lap, loss):
    """Leader X (hard, medium planned for lap 36), SC on lap 5: X crosses the line inside the window at the end of
    laps 5, 6 and 7 (t_d, t_d + L_sc, t_d + 2 L_sc; lap 8 ends at t_r). Laps left 48/47/46."""
    p = variant(params, MEDIUM__max_life_laps=life, sc_eligibility=policy)
    res = simulate_race([car("X", 0.0, 1, one_stop("HARD", 36, "MEDIUM")), car("Y", 0.3, 2)], p, SafetyCar(5))
    pit = res.events[(res.events.type == "pit") & (res.events.car == "X")].iloc[0]
    assert pit["lap"] == lap and pit["loss_s"] == loss
    check_timeline(res, p)


@pytest.mark.parametrize("policy, expected", [("first_green_lap", [14]), ("first_lap_of_stint", [])])
def test_warmup_after_a_safety_car_stop_is_paid_on_the_first_racing_lap(params, policy, expected):
    """Leader X fits hards at t_d (end of lap 10) and rejoins behind Y; laps 11-13 start inside the window; the
    first lap that starts at racing speed is lap 14."""
    p = variant(params, warmup_policy=policy)
    res = simulate_race([car("X", 0.0, 1, one_stop("MEDIUM", 10, "HARD")), car("Y", 0.3, 2)], p, SafetyCar(10))
    x = res.laps[res.laps.car == "X"].set_index("lap")
    assert list(x.index[x["warmup_applied"]]) == expected
    check_timeline(res, p)


# ---- 5. whole-race invariants ------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(12))
def test_random_races_satisfy_the_timeline_invariants(params, seed):
    """22 random cars, random one-stop plans, random starting spreads (some cars lapped), a Safety Car on a
    random lap (1-50), dirty air and passing on."""
    rng = np.random.default_rng(1000 + seed)
    p = variant(params, grid_slot_gap_s=float(rng.choice([0.117, 2.0, 6.0])))
    entries = []
    for i in range(22):
        start, nxt = rng.choice(["MEDIUM", "HARD"]), "HARD"
        strat = one_stop(str(start), int(rng.integers(2, 45)), nxt if start == "MEDIUM" else "MEDIUM")
        entries.append(car(f"R{i:02d}", float(rng.uniform(0.0, 3.5)), i + 1, strat))
    res = simulate_race(entries, p, SafetyCar(int(rng.integers(1, 51))))
    check_physical(res, p)
    check_timeline(res, p)
    check_passes_time_feasible(res)


def test_lap1_grid_loss_and_headway_differ_only_through_lap1_interactions(params):
    """Two representations of a 4 s grid deficit (slot 5, 1 s per slot): as a starting headway, C starts 4 s back
    in clean air (88.0); as TUM's lap-1 loss, C starts alongside, in dirty air, and loses 4 s on lap 1 (88.25)."""
    flat = variant(params, fuel_s_per_lap=0.0, HARD__deg_s_per_lap=0.0)
    headway = variant(flat, grid_slot_gap_s=1.0)
    loss = variant(flat, grid_slot_gap_s=0.0, grid_lap1_loss_per_slot_s=1.0)
    solo_h = simulate_race([car("C", 0.0, 5)], headway)
    solo_l = simulate_race([car("C", 0.0, 5)], loss)
    assert line(solo_h, "C", 1) == pytest.approx(88.0) and line(solo_l, "C", 1) == pytest.approx(88.0)
    duo_h = simulate_race([car("A", 0.0, 1), car("C", 0.0, 5)], headway)
    duo_l = simulate_race([car("A", 0.0, 1), car("C", 0.0, 5)], loss)
    assert line(duo_h, "C", 1) == pytest.approx(88.0) and line(duo_l, "C", 1) == pytest.approx(88.25)


# ---- 6. pit-stop bookkeeping under the Safety Car -----------------------------------------------------------------

def test_can_reach_flag(params):
    assert can_reach_flag(params, "HARD", 1)          # 52 laps left, hard life 52
    assert not can_reach_flag(params, "MEDIUM", 5)    # 48 > 46
    assert not can_reach_flag(params, "MEDIUM", 6)    # 47 > 46
    assert can_reach_flag(params, "MEDIUM", 7)        # 46 <= 46


def test_elapsed_time_never_decreases_through_a_late_safety_car(params):
    """A car about 100 s behind (lapped) when a Safety Car comes out on lap 35."""
    entries = [car("F", 0.0, 1), car("M", 1.0, 2), car("S", 3.8, 3)]
    res = simulate_race(entries, params, SafetyCar(35))
    check_physical(res, params)
    check_timeline(res, params)


def test_cheap_stop_is_charged_once_at_sc_price(params):
    """Eight cars planning a stop on lap 30; SC on lap 10. Every car ends its lap 10 inside the window and a hard
    can reach the flag from lap 10 (43 laps), so all eight stop at the end of lap 10, at the Safety-Car price, once."""
    entries = [car(f"C{i}", 0.5 * i, i + 1, one_stop("MEDIUM", 30, "HARD")) for i in range(8)]
    res = simulate_race(entries, params, SafetyCar(10))
    pits = res.events[res.events.type == "pit"]
    assert len(pits) == 8 and (pits["lap"] == 10).all() and (pits["loss_s"] == params.pit_loss_sc_s).all()
    assert np.allclose(res.laps.groupby("car")["pit_loss_s"].sum(), params.pit_loss_sc_s)
    check_timeline(res, params)


def test_car_that_already_stopped_does_not_stop_again(params):
    res = simulate_race([car("X", 0.0, 1, one_stop("MEDIUM", 1, "HARD")), car("Y", 0.3, 2)], params, SafetyCar(20))
    assert list(res.events[(res.events.type == "pit") & (res.events.car == "X")]["lap"]) == [1]
