"""Ruth's pace estimators reproduced from timing; filter definitions; the degradation estimator."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT
from racesim.pace import (fp2_long_runs, friday_weighted, fuel_corrected_degradation, push_laps, quali_gaps,
                          saturday_rebased)

PUB = ROOT / "data" / "published"
TIM = ROOT / "data" / "timing"
HALF_UNIT = 0.0051       # Ruth publishes hundredths: agreement within half a rounding unit (+ float slack)


@pytest.fixture(scope="module")
def timing():
    return pd.read_csv(TIM / "fp2_laps.csv"), pd.read_csv(TIM / "quali_results.csv")


def test_symmetric_and_one_sided_filters_select_the_same_runs(timing):
    laps, _ = timing
    a = fp2_long_runs(laps, symmetric=True, green_only=True)
    b = fp2_long_runs(laps, symmetric=False, green_only=False)
    assert list(a.index) == list(b.index)
    assert all(x == y for x, y in zip(a["lap_numbers"], b["lap_numbers"]))


def test_bearman_is_found_by_the_literal_rule(timing):
    laps, _ = timing
    assert "BEA" in fp2_long_runs(laps).index            # source-matching exception documented in the notebook


def test_fp2_long_runs_match_ruth_chart(timing):
    laps, _ = timing
    ours = fp2_long_runs(laps, exclude=("BEA",))
    ruth = pd.read_csv(PUB / "ruth_fp2_long_runs.csv").set_index("driver")
    assert set(ours.index) == set(ruth.index)
    assert (ours.loc[ruth.index, "gap_s"] - ruth["gap_s"]).abs().max() <= HALF_UNIT
    assert (ours.loc[ruth.index, "clean_laps"] == ruth["clean_laps"]).all()


def test_push_laps_are_valid_laps(timing):
    laps, _ = timing
    push = push_laps(laps)
    raw_min = laps.groupby("Driver")["LapTime_s"].min()
    assert np.allclose(push.loc[raw_min.index], raw_min)  # every fastest lap is green and accurately timed


def test_friday_weighted_matches_ruth_graphic(timing):
    laps, quali = timing
    fw = friday_weighted(fp2_long_runs(laps, exclude=("BEA",))["gap_s"], quali_gaps(quali))
    ruth = pd.read_csv(PUB / "ruth_friday_weighted_pace.csv").set_index("driver")["pace_gap_s"]
    assert (fw.loc[ruth.index] - ruth).abs().max() <= HALF_UNIT


def test_saturday_rebased_matches_ruth_text(timing):
    laps, quali = timing
    sr = saturday_rebased(quali_gaps(quali), fp2_long_runs(laps, exclude=("BEA",))["mean_lap_s"], push_laps(laps))
    ruth = pd.read_csv(PUB / "ruth_saturday_rebased_pace.csv").set_index("driver")["pace_gap_s"]
    assert (sr.loc[ruth.index, "saturday_rebased_s"] - ruth).abs().max() <= HALF_UNIT


def test_degradation_estimator_recovers_a_known_slope():
    """Synthetic stint: d = 0.02 s/lap, fuel 0.07 s/lap, small noise -> slope -0.05, d_hat ~ 0.02."""
    rng = np.random.default_rng(3)
    x = np.arange(1, 18)
    y = 86.0 + (0.02 - 0.07) * x + rng.normal(0, 0.05, x.size)
    fit = fuel_corrected_degradation(x, y, fuel_s_per_lap=0.07)
    assert fit["deg_ci_low"] < 0.02 < fit["deg_ci_high"]
    assert abs(fit["deg_hat"] - 0.02) < 0.01
    # the estimate shifts one-for-one with the assumed fuel effect (non-identifiability within a stint)
    assert fuel_corrected_degradation(x, y, 0.09)["deg_hat"] - fit["deg_hat"] == pytest.approx(0.02)
