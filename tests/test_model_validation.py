from __future__ import annotations

import numpy as np
import pytest

from gkcoverage.constants import GOAL_HEIGHT_M, HALF_GOAL_WIDTH_M
from gkcoverage.model import (
    SIGMA_BOUNDS_S,
    STANDARD_PRIOR,
    CoverageSurfaceModel,
    PriorScale,
    _effective_dof,
    _effective_sample_size,
    _residual_scale_correction,
)
from gkcoverage.simulate import ground_truth_time, simulate_penalties, true_upper_outer_contrast


def _coverage_of_true_surface(n: int, seeds: range) -> float:
    xx, yy = np.meshgrid(np.linspace(-3.0, 3.0, 9), np.linspace(0.3, 2.1, 6))
    points = np.c_[xx.ravel(), yy.ravel()]
    truth = ground_truth_time(points)
    covered = []
    for seed in seeds:
        try:
            fit = CoverageSurfaceModel().fit(simulate_penalties(n, seed=seed))
        except RuntimeError:  # degenerate residual scale, refused by design
            continue
        prediction = fit.time_surface.predict(points)
        covered.append(np.mean((prediction["lower"] <= truth) & (truth <= prediction["upper"])))
    return float(np.mean(covered))


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


def test_degenerate_residual_scale_is_reported_rather_than_silently_pinned() -> None:
    # L-BFGS-B reports success=True at an active bound, so a residual scale that
    # collapses onto the bound used to pass unnoticed and still produce Laplace
    # intervals, which assume an interior optimum. Those fits are the badly
    # calibrated ones: at n=20 they cover 0.81 of the true surface against a
    # nominal 0.95, versus 0.91 for fits whose scale stays interior.
    # This sample drives the unbounded scale to about 2e-6 s, four orders of
    # magnitude under the bound, so the pin does not depend on optimizer details.
    with pytest.raises(RuntimeError, match="residual scale reached its lower bound"):
        CoverageSurfaceModel().fit(simulate_penalties(16, seed=11))

    # The guard must not fire on a fit whose scale is genuinely interior.
    fit = CoverageSurfaceModel().fit(simulate_penalties(200, seed=0))
    assert SIGMA_BOUNDS_S[0] < fit.time_surface.sigma < SIGMA_BOUNDS_S[1]


def test_residual_scale_correction_tracks_the_residual_degrees_of_freedom() -> None:
    assert _residual_scale_correction(200, 0.0) == 1.0  # nothing spent, nothing to correct
    assert _residual_scale_correction(200, 20.0) == pytest.approx(np.sqrt(200.0 / 180.0))
    # The same dof cost bites harder the smaller the sample.
    assert _residual_scale_correction(40, 18.0) > _residual_scale_correction(200, 18.0)
    # Guarded rather than singular when the fit spends everything.
    assert _residual_scale_correction(10, 20.0) == pytest.approx(np.sqrt(10.0))


def _prior_sd_seconds(model: CoverageSurfaceModel, small_sample: bool, row: np.ndarray) -> float:
    """Prior SD of one surface component at one point, before any data."""
    precision, _ = model._prior(small_sample=small_sample, center=0.0)
    return float(np.sqrt(row @ np.linalg.pinv(precision) @ row))


def _component_row(model: CoverageSurfaceModel, point: tuple[float, float], component: str) -> np.ndarray:
    symmetric, asymmetric = model.basis.component_design(np.atleast_2d(point))
    if component == "symmetric":
        return np.r_[symmetric[0], np.zeros(asymmetric.shape[1])]
    return np.r_[np.zeros(symmetric.shape[1]), asymmetric[0]]


@pytest.mark.parametrize(
    ("component", "point", "standard_sd", "small_sample_sd"),
    [
        ("symmetric", (0.0, 1.2), 0.15, 0.30),
        ("symmetric", (3.5, 2.3), 0.21, 0.41),
        ("asymmetric", (3.5, 2.3), 0.11, 0.26),
    ],
)
def test_prior_precisions_assert_the_documented_scale_in_seconds(
    component: str, point: tuple[float, float], standard_sd: float, small_sample_sd: float
) -> None:
    # The precisions are only auditable through what they imply about the surface, so
    # pin that rather than the raw numbers: changing a precision changes what the model
    # believes before it sees a single shot, and the table above STANDARD_PRIOR is the
    # claim being made. These are the specification; the precisions are one point that
    # meets it.
    model = CoverageSurfaceModel()
    row = _component_row(model, point, component)

    assert _prior_sd_seconds(model, False, row) == pytest.approx(standard_sd, abs=0.01)
    assert _prior_sd_seconds(model, True, row) == pytest.approx(small_sample_sd, abs=0.01)


def test_prior_is_weakly_informative_about_the_headline_contrast() -> None:
    # The report leads with the left-minus-right contrast, so a prior that pinned it
    # near zero would manufacture the conservatism the gate then measures. It does not:
    # the true effect sits well inside one prior SD, and small-sample mode widens it.
    model = CoverageSurfaceModel()
    x = np.linspace(HALF_GOAL_WIDTH_M * 0.52, HALF_GOAL_WIDTH_M * 0.92, 8)
    y = np.linspace(GOAL_HEIGHT_M * 0.55, GOAL_HEIGHT_M * 0.95, 7)
    xx, yy = np.meshgrid(x, y)
    positive = np.c_[xx.ravel(), yy.ravel()]
    negative = positive.copy()
    negative[:, 0] *= -1.0
    row = (model.basis.design(negative) - model.basis.design(positive)).mean(axis=0)

    standard = _prior_sd_seconds(model, False, row)
    assert standard == pytest.approx(0.12, abs=0.01)
    assert _prior_sd_seconds(model, True, row) == pytest.approx(0.28, abs=0.01)
    # The truth the gate recovers is a fraction of a prior SD out, not a tail event.
    assert abs(true_upper_outer_contrast()) < 0.5 * standard


@pytest.mark.slow
def test_headline_claim_sits_on_a_prior_plateau_rather_than_an_optimum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The precisions were not tuned against the simulator, which would leave the gate
    # marking its own work. This is the evidence for that being safe: the claim barely
    # moves across a 4x sweep, so no fitting happened at this scale. Regularization as
    # such is still load-bearing, which the last case shows.
    def mean_contrast(scale: PriorScale) -> float:
        monkeypatch.setattr("gkcoverage.model.STANDARD_PRIOR", scale)
        return float(
            np.mean([
                CoverageSurfaceModel()
                .fit(simulate_penalties(200, seed=seed))
                .asymmetry_summary()["time_left_minus_right_s"]
                for seed in range(8)
            ])
        )

    shipped = mean_contrast(STANDARD_PRIOR)
    halved = mean_contrast(PriorScale(11.0, 19.0, 4.0, 14.0))
    doubled = mean_contrast(PriorScale(44.0, 76.0, 16.0, 56.0))
    unregularized = mean_contrast(PriorScale(0.1, 0.1, 0.1, 0.1))

    assert abs(halved - shipped) < 0.005
    assert abs(doubled - shipped) < 0.005
    # Without a prior the contrast runs well past the truth the gate checks.
    assert unregularized > 1.5 * shipped


def test_effective_sample_size_discounts_censored_rows() -> None:
    # A censored row says only that T exceeded the flight time, so it informs the
    # residual scale less than an exact contact time does and must count for less
    # than a whole row.
    scale = 0.05
    exact = 1.0 / scale**2

    # No censoring: every weight is 1/scale^2 and the count is exactly n, which is
    # what leaves the uncensored velocity path unaffected.
    assert _effective_sample_size(np.full(30, exact), scale) == pytest.approx(30.0)

    # Twenty exact rows plus twenty half-weight censored rows are worth thirty.
    mixed = np.r_[np.full(20, exact), np.full(20, 0.5 * exact)]
    assert _effective_sample_size(mixed, scale) == pytest.approx(30.0)
    assert _effective_sample_size(mixed, scale) < 40.0


def test_censored_surface_corrects_against_effective_not_raw_sample_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The regression: the row count was passed where an information-weighted count
    # belongs, while edf came from the censoring-discounted weights. Mismatching the
    # two under-corrects the scale; at n=40 it left sigma about 15% low and pointwise
    # coverage at 0.935 against a nominal 0.95. The effective count must therefore sit
    # strictly between the exact-row count and the row count.
    records = simulate_penalties(200, seed=0)
    exact_rows = sum(not record.censored for record in records)
    assert 0 < exact_rows < len(records)  # else the bound below proves nothing

    seen: list[float] = []

    def spy(n_effective: float, edf: float) -> float:
        seen.append(n_effective)
        return _residual_scale_correction(n_effective, edf)

    monkeypatch.setattr("gkcoverage.model._residual_scale_correction", spy)
    CoverageSurfaceModel().fit(records)

    assert seen, "the censored surface must correct its residual scale"
    assert exact_rows < seen[0] < len(records)


def test_effective_dof_counts_what_the_data_determines() -> None:
    rng = np.random.default_rng(0)
    design = rng.normal(size=(200, 6))
    weights = np.full(200, 4.0)
    identity = np.eye(6)

    # A prior that dominates leaves the data determining nothing; a vanishing prior
    # leaves it determining every basis function.
    assert _effective_dof(design, weights, identity * 1e8) < 0.01
    assert _effective_dof(design, weights, identity * 1e-8) == pytest.approx(6.0, abs=1e-3)
    assert 0.0 < _effective_dof(design, weights, identity) < 6.0


def test_residual_scale_is_corrected_for_effective_dof(monkeypatch: pytest.MonkeyPatch) -> None:
    # Maximum likelihood divides by n and ignores the dof spent on the mean
    # function, so at n=40, where edf/n is about 0.45, it lands near 0.031 against
    # a true 0.045 and the intervals come out too narrow.
    records = simulate_penalties(40, seed=0)
    point = np.array([[-2.0, 1.8]])
    corrected = CoverageSurfaceModel().fit(records).time_surface

    monkeypatch.setattr("gkcoverage.model._residual_scale_correction", lambda n, edf: 1.0)
    uncorrected = CoverageSurfaceModel().fit(records).time_surface

    assert corrected.sigma > uncorrected.sigma * 1.2
    assert corrected.predict(point)["sd"][0] > uncorrected.predict(point)["sd"][0]
    assert "residual-dof correction" in corrected.uncertainty_scope


@pytest.mark.slow
def test_edf_correction_improves_small_sample_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    # Paired on identical seeds rather than compared against a fixed threshold, so
    # the assertion does not depend on the exact coverage a given toolchain lands on.
    corrected = _coverage_of_true_surface(40, range(40))

    monkeypatch.setattr("gkcoverage.model._residual_scale_correction", lambda n, edf: 1.0)
    uncorrected = _coverage_of_true_surface(40, range(40))

    assert corrected > uncorrected
    assert corrected > 0.90  # uncorrected sits near 0.896 against a nominal 0.95


def test_non_contact_shots_are_used_as_censored_observations() -> None:
    records = simulate_penalties(200, seed=17)
    assert any(record.censored for record in records)
    fit = CoverageSurfaceModel().fit(records)
    assert fit.time_surface.n_observations == 200
    prediction = fit.time_surface.predict(np.array([[0.0, 1.2]]))
    # lower < mean < upper holds by construction for any beta, so assert the mean
    # is a physically plausible time-to-contact and the interval really is the
    # stated 95% multiple of sd about it.
    assert 0.2 < prediction["mean"][0] < 0.6
    half_width = (prediction["upper"][0] - prediction["lower"][0]) / 2.0
    assert half_width == pytest.approx(1.959964 * prediction["sd"][0], rel=1e-6)
