from __future__ import annotations

import numpy as np
import pytest

from gkcoverage.model import CoverageSurfaceModel
from gkcoverage.simulate import simulate_penalties, true_upper_outer_contrast


@pytest.mark.slow
def test_twenty_run_simulation_recovers_asymmetry_with_ninety_percent_interval_coverage() -> None:
    truth = true_upper_outer_contrast()
    covered = 0
    sign_correct = 0
    for seed in range(20):
        fit = CoverageSurfaceModel().fit(simulate_penalties(200, seed=seed))
        summary = fit.asymmetry_summary()
        estimate = float(summary["time_left_minus_right_s"])
        lower = float(summary["time_ci_lower_s"])
        upper = float(summary["time_ci_upper_s"])
        covered += int(lower <= truth <= upper)
        sign_correct += int(estimate * truth > 0)
    assert covered >= 18
    assert sign_correct == 20


def test_surface_is_continuous_across_the_centre_line() -> None:
    # f = S(|x|,y) + sign(x) A(|x|,y) jumps by 2 A(0,y) at the centre line unless
    # A(0,y) is pinned to zero. The jump reached 59 ms before the basis was
    # constrained, against a headline asymmetry effect of roughly 37 ms.
    fit = CoverageSurfaceModel().fit(simulate_penalties(200, seed=0))
    surface = fit.time_surface
    y = np.linspace(0.0, 2.44, 13)
    eps = 1e-9

    left = surface.predict(np.c_[np.full_like(y, -eps), y])
    right = surface.predict(np.c_[np.full_like(y, eps), y])
    assert np.max(np.abs(right["mean"] - left["mean"])) < 1e-6
    assert np.max(np.abs(right["sd"] - left["sd"])) < 1e-6

    asymmetric = surface.predict_components(np.c_[np.zeros_like(y), y])["asymmetric"]
    assert np.max(np.abs(asymmetric["mean"])) < 1e-9


def test_prior_dominance_does_not_invert_when_small_sample_widens_the_prior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Holding the data fixed, only the prior mode changes. A wider prior means
    # more real uncertainty, so reported prior dominance must not fall. Dividing
    # by each fit's own prior inverted this: widening the prior inflated the
    # denominator faster than sd, so the banner-ON fit claimed to be less
    # prior-dominated than the banner-OFF one.
    records = simulate_penalties(149, seed=0)
    corner = np.array([[-3.65, 2.43]])

    widened = CoverageSurfaceModel().fit(records)
    assert widened.small_sample

    monkeypatch.setattr("gkcoverage.model.SMALL_SAMPLE_THRESHOLD", 0)
    standard = CoverageSurfaceModel().fit(records)
    assert not standard.small_sample

    wide = widened.time_surface.predict(corner)
    tight = standard.time_surface.predict(corner)
    assert wide["sd"][0] > tight["sd"][0]
    assert wide["prior_dominance"][0] >= tight["prior_dominance"][0]


def test_non_contact_shots_are_used_as_censored_observations() -> None:
    records = simulate_penalties(200, seed=17)
    assert any(record.censored for record in records)
    fit = CoverageSurfaceModel().fit(records)
    assert fit.time_surface.n_observations == 200
    prediction = fit.time_surface.predict(np.array([[0.0, 1.2]]))
    assert prediction["lower"][0] < prediction["mean"][0] < prediction["upper"][0]
