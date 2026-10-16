"""Safety-Car modelling (absent from Heilmeier et al. 2018; Ruth's rules, made explicit).

The Safety-Car period is a window of the SHARED race clock, identical for every car, whatever lap it is on:

  deployment   t_d = the instant the race leader completes lap N ('SC lap N'). The Safety Car joins the track at
               the line in front of the leader.
  restart      t_r = t_d + D * L_sc: the Safety Car completes D laps of its own (L_sc = sc_lap_factor x reference
               lap) and pulls in as it reaches the line. Racing resumes for the whole field at t_r.
  window       W = [t_d, t_r). Every car-lap is cut at t_d and t_r by its START time and its progress:
               * before t_d and after t_r a car moves at its own racing pace;
               * inside W at the full-course-yellow delta pace L_fcy = fcy_lap_factor x reference lap, the same for
                 every car (the FIA delta is not car-specific), unless it has caught the queue;
               * progress within a lap is uniform in time at the pace in force ("scaled-profile" assumption:
                 every car's speed profile is a scaled copy of the same lap, so time fractions are distance
                 fractions).
  line rules   (applied where cars are compared: at the line, as everywhere in this lap-wise model)
               * no car may complete lap N+j (1 <= j <= D) before the Safety Car does, at t_d + j L_sc;
               * no overtaking while neutralised: an order change during a lap that is cut at t_d stands only if
                 it was complete at t_d; a lap that starts inside W keeps the order it started in, also through the
                 restart until the car reaches the line (FIA: no overtaking before the line);
               * on the lap cut at t_d every car covers its remaining distance at the same pace, so distances are
                 kept and nobody closes in; the queue spacing is not imposed at that line (the queue forms during
                 the first full lap behind the Safety Car);
               * on laps that START inside W, minimum spacing at the line: the time the car ahead needs to cover
                 one minimum following distance. That is the queue gap g_sc (0.8 s) at Safety-Car pace and the
                 minimum gap g_min (0.5 s) at racing pace - the same distance, because 0.5 s x 1.6 = 0.8 s -
                 interpolated continuously when the car ahead crosses within one spacing of t_d or t_r
                 (`follow_gap`).
  pit price    a stop costs the Safety-Car price iff the car reaches the end of its in-lap inside W (pit entry
               ~ the line). The price is fixed at pit entry.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .config import RaceParams


@dataclass(frozen=True)
class SafetyCar:
    """A Safety-Car scenario as Ruth specifies it: deployed on lap N, out for D laps."""
    deploy_lap: int     # 'SC lap N'
    duration_laps: int = 3


@dataclass(frozen=True)
class SafetyCarWindow:
    """The Safety-Car period on the shared race clock."""
    deploy_lap: int       # N
    duration_laps: int    # D
    t_deploy: float       # t_d, seconds of race time
    sc_lap_s: float       # L_sc

    @property
    def t_restart(self) -> float:
        return self.sc_line_time(self.deploy_lap + self.duration_laps)

    def sc_line_time(self, lap: int) -> float:
        """Instant the Safety Car completes `lap` (j = lap - N laps after picking up the leader). No car may
        complete that lap earlier. -inf for laps outside N+1..N+D (no constraint)."""
        j = lap - self.deploy_lap
        return self.t_deploy + j * self.sc_lap_s if 1 <= j <= self.duration_laps else -math.inf

    def neutral(self, t: float) -> bool:
        """Is the race neutralised at race time t? W = [t_d, t_r)."""
        return self.t_deploy <= t < self.t_restart

    def neutral_overlap(self, start: float, end: float) -> float:
        """Seconds of the interval [start, end] that lie inside the window."""
        return max(0.0, min(end, self.t_restart) - max(start, self.t_deploy))


def can_reach_flag(p: RaceParams, compound: str, in_lap: int) -> bool:
    """Ruth: a car takes the cheap stop only 'if their next compound can reach the flag'."""
    return p.total_laps - in_lap <= p.compounds[compound].max_life_laps


def free_lap_end(start: float, green_lap_s: float, w: SafetyCarWindow | None, fcy_lap_s: float) -> float:
    """End of a lap that starts at `start`, before any interaction at the line.

    The lap is covered uniformly in time at the pace in force: `green_lap_s` (the car's own racing lap time)
    outside the window, `fcy_lap_s` inside it. Works for every case: a lap entirely green or neutralised, cut
    at t_d, cut at t_r, or spanning the whole window.
    """
    if w is None:
        return start + green_lap_s
    t, done = start, 0.0
    for lo, hi, pace in ((-math.inf, w.t_deploy, green_lap_s), (w.t_deploy, w.t_restart, fcy_lap_s),
                         (w.t_restart, math.inf, green_lap_s)):
        if t >= hi:
            continue
        t = max(t, lo)
        finish = t + (1.0 - done) * pace
        if finish <= hi:
            return finish
        done += (hi - t) / pace
        t = hi
    raise AssertionError("unreachable")


def follow_gap(t_ahead: float, w: SafetyCarWindow | None, p: RaceParams) -> float:
    """Minimum time between the car ahead crossing the line at `t_ahead` and the next car crossing it.

    One minimum following DISTANCE, expressed as the time the car ahead needs to cover it: g_min at racing pace,
    g_sc at Safety-Car pace; if the car ahead changes pace within that spacing (it crosses less than one spacing
    before t_d or t_r), the time is the sum of the two parts, so the rule is continuous at both boundaries.
    """
    g_min, g_sc = p.min_follow_gap_s, p.sc_queue_gap_s
    if w is None:
        return g_min
    if w.t_deploy - g_min < t_ahead < w.t_deploy:            # still racing for s seconds, then neutralised
        s = w.t_deploy - t_ahead
        return s + (1.0 - s / g_min) * g_sc
    if w.neutral(t_ahead):
        s = w.t_restart - t_ahead
        return g_sc if s >= g_sc else s + (1.0 - s / g_sc) * g_min   # restart within one spacing
    return g_min


def lap_phase(start: float, end: float, w: SafetyCarWindow | None) -> str:
    """Label of one car-lap on the shared clock."""
    if w is None or end < w.t_deploy:
        return "GREEN"
    if start < w.t_deploy:
        return "SC_DEPLOYED"          # cut at t_d (or ends exactly at t_d: the leader's lap N)
    if start >= w.t_restart:
        return "GREEN"
    if end <= w.t_restart:
        return "SC"                   # entirely neutralised
    return "SC_RESTART"               # cut at t_r: neutralised, then racing pace, no overtaking before the line
