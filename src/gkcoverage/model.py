from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import BSpline
from scipy.optimize import minimize
from scipy.special import log_ndtr
from scipy.stats import norm

from .constants import GOAL_HEIGHT_M, HALF_GOAL_WIDTH_M, SMALL_SAMPLE_THRESHOLD
from .schema import PenaltyRecord

FloatArray = NDArray[np.float64]


def _clamped_knots(lo: float, hi: float, n_basis: int, degree: int) -> FloatArray:
    internal_count = n_basis - degree - 1
    if internal_count < 0:
        raise ValueError("n_basis must be at least degree + 1")
    internal = (
        np.linspace(lo, hi, internal_count + 2, dtype=float)[1:-1]
        if internal_count
        else np.array([], dtype=float)
    )
    return np.r_[np.repeat(lo, degree + 1), internal, np.repeat(hi, degree + 1)]


def _bspline_basis(values: ArrayLike, knots: FloatArray, degree: int) -> FloatArray:
    values_arr = np.asarray(values, dtype=float)
    clipped = np.clip(values_arr, knots[degree], knots[-degree - 1])
    return BSpline.design_matrix(clipped, knots, degree, extrapolate=False).toarray()


def _difference_penalty(nx: int, ny: int) -> FloatArray:
    def second_difference(n: int) -> FloatArray:
        if n < 3:
            return np.zeros((0, n))
        d = np.zeros((n - 2, n))
        for i in range(n - 2):
            d[i, i : i + 3] = (1.0, -2.0, 1.0)
        return d

    dx = second_difference(nx)
    dy = second_difference(ny)
    px = np.kron(dx.T @ dx, np.eye(ny)) if dx.size else np.zeros((nx * ny, nx * ny))
    py = np.kron(np.eye(nx), dy.T @ dy) if dy.size else np.zeros((nx * ny, nx * ny))
    return px + py


@dataclass(frozen=True)
class SplineSpecification:
    nx: int = 5
    ny: int = 4
    degree: int = 3

    @property
    def n_terms(self) -> int:
        return self.nx * self.ny


class SymmetricSplineBasis:
    """Tensor spline basis with an explicit symmetric/antisymmetric split."""

    def __init__(self, spec: SplineSpecification) -> None:
        self.spec = spec
        self.x_knots = _clamped_knots(0.0, HALF_GOAL_WIDTH_M, spec.nx, spec.degree)
        self.y_knots = _clamped_knots(0.0, GOAL_HEIGHT_M, spec.ny, spec.degree)
        self.penalty = _difference_penalty(spec.nx, spec.ny)

    def component_design(self, xy: ArrayLike) -> tuple[FloatArray, FloatArray]:
        points = np.asarray(xy, dtype=float)
        points = np.atleast_2d(points)
        bx = _bspline_basis(np.abs(points[:, 0]), self.x_knots, self.spec.degree)
        by = _bspline_basis(points[:, 1], self.y_knots, self.spec.degree)
        tensor = np.einsum("ni,nj->nij", bx, by).reshape(points.shape[0], -1)
        signs = np.sign(points[:, 0])[:, None]
        return tensor, tensor * signs

    def design(self, xy: ArrayLike) -> FloatArray:
        symmetric, asymmetric = self.component_design(xy)
        return np.c_[symmetric, asymmetric]


@dataclass
class GaussianSurfaceFit:
    basis: SymmetricSplineBasis
    beta: FloatArray
    covariance: FloatArray
    sigma: float
    prior_precision: FloatArray
    prior_mean: FloatArray
    n_observations: int
    response_name: str
    units: str
    uncertainty_scope: str = "Laplace/linear-Gaussian mean-function interval; smoothing hyperparameters fixed"

    def predict(self, xy: ArrayLike, level: float = 0.95) -> dict[str, FloatArray]:
        design = self.basis.design(xy)
        mean = design @ self.beta
        variance = np.einsum("ni,ij,nj->n", design, self.covariance, design)
        sd = np.sqrt(np.maximum(variance, 0.0))
        z = norm.ppf(0.5 + level / 2.0)
        prior_cov = np.linalg.pinv(self.prior_precision)
        prior_variance = np.einsum("ni,ij,nj->n", design, prior_cov, design)
        prior_sd = np.sqrt(np.maximum(prior_variance, 1e-15))
        return {
            "mean": mean,
            "sd": sd,
            "lower": mean - z * sd,
            "upper": mean + z * sd,
            "prior_dominance": np.clip(sd / prior_sd, 0.0, 1.0),
        }

    def predict_components(self, xy: ArrayLike, level: float = 0.95) -> dict[str, FloatArray]:
        symmetric, asymmetric = self.basis.component_design(xy)
        m = self.basis.spec.n_terms
        design_s = np.c_[symmetric, np.zeros_like(asymmetric)]
        design_a = np.c_[np.zeros_like(symmetric), asymmetric]
        z = norm.ppf(0.5 + level / 2.0)
        prior_cov = np.linalg.pinv(self.prior_precision)

        def summarize(design: FloatArray) -> dict[str, FloatArray]:
            mean = design @ self.beta
            variance = np.einsum("ni,ij,nj->n", design, self.covariance, design)
            sd = np.sqrt(np.maximum(variance, 0.0))
            prior_variance = np.einsum("ni,ij,nj->n", design, prior_cov, design)
            prior_sd = np.sqrt(np.maximum(prior_variance, 1e-15))
            return {
                "mean": mean,
                "sd": sd,
                "lower": mean - z * sd,
                "upper": mean + z * sd,
                "prior_dominance": np.clip(sd / prior_sd, 0.0, 1.0),
            }

        return {"symmetric": summarize(design_s), "asymmetric": summarize(design_a)}

    def paired_contrast(self, positive_xy: ArrayLike, level: float = 0.95) -> dict[str, float]:
        positive = np.asarray(positive_xy, dtype=float)
        positive = np.atleast_2d(positive)
        if np.any(positive[:, 0] < 0):
            raise ValueError("paired contrast expects points on positive-x half")
        negative = positive.copy()
        negative[:, 0] *= -1.0
        # Left minus right. Positive time difference means slower/weaker left side.
        contrast_row = (self.basis.design(negative) - self.basis.design(positive)).mean(axis=0)
        estimate = float(contrast_row @ self.beta)
        variance = float(contrast_row @ self.covariance @ contrast_row)
        sd = float(np.sqrt(max(variance, 0.0)))
        z = float(norm.ppf(0.5 + level / 2.0))
        return {
            "estimate": estimate,
            "sd": sd,
            "lower": estimate - z * sd,
            "upper": estimate + z * sd,
            "level": level,
        }


@dataclass
class SurfaceFit:
    time_surface: GaussianSurfaceFit
    velocity_surface: GaussianSurfaceFit
    small_sample: bool
    banner: str
    keeper_id: str
    n_records: int
    median_flight_time_s: float

    def asymmetry_summary(self) -> dict[str, float | str]:
        x = np.linspace(HALF_GOAL_WIDTH_M * 0.52, HALF_GOAL_WIDTH_M * 0.92, 8)
        y = np.linspace(GOAL_HEIGHT_M * 0.55, GOAL_HEIGHT_M * 0.95, 7)
        xx, yy = np.meshgrid(x, y)
        points = np.c_[xx.ravel(), yy.ravel()]
        time = self.time_surface.paired_contrast(points)
        velocity = self.velocity_surface.paired_contrast(points)
        displacement_scale = self.median_flight_time_s * 100.0
        displacement = {key: velocity[key] * displacement_scale for key in ("estimate", "sd", "lower", "upper")}
        side = "left" if time["estimate"] > 0 else "right"
        return {
            "region": "upper outer goal: |x|=52–92% half-width, y=55–95% height",
            "weaker_side_by_time": side,
            "time_left_minus_right_s": time["estimate"],
            "time_ci_lower_s": time["lower"],
            "time_ci_upper_s": time["upper"],
            "reach_left_minus_right_cm_at_median_flight": displacement["estimate"],
            "reach_ci_lower_cm": displacement["lower"],
            "reach_ci_upper_cm": displacement["upper"],
            "median_flight_time_s": self.median_flight_time_s,
            "uncertainty_note": self.time_surface.uncertainty_scope,
        }


class CoverageSurfaceModel:
    """Censoring-aware penalized spline estimator.

    The primary coverage surface models time-to-contact in seconds. Saves are
    exact observations. Goals/woodwork contribute a right-censored likelihood
    P(T > observed flight time). A companion surface models displacement speed
    (keeper displacement magnitude divided by ball flight time) without
    censoring. Both surfaces enforce f(x,y)=S(|x|,y)+sign(x)A(|x|,y).
    """

    def __init__(self, spec: SplineSpecification | None = None) -> None:
        self.basis = SymmetricSplineBasis(spec or SplineSpecification())

    def _prior(self, n: int, small_sample: bool, center: float) -> tuple[FloatArray, FloatArray]:
        m = self.basis.spec.n_terms
        # Wider priors in deliberate small-sample mode: less ridge and less smoothness.
        smooth_s = 22.0 if not small_sample else 7.0
        smooth_a = 38.0 if not small_sample else 10.0
        ridge_s = 8.0 if not small_sample else 2.0
        ridge_a = 28.0 if not small_sample else 5.0
        p = self.basis.penalty
        precision_s = smooth_s * p + ridge_s * np.eye(m)
        precision_a = smooth_a * p + ridge_a * np.eye(m)
        precision = np.block(
            [[precision_s, np.zeros((m, m))], [np.zeros((m, m)), precision_a]]
        )
        mean = np.r_[np.full(m, center), np.zeros(m)]
        return precision, mean

    def _fit_censored_time(
        self,
        xy: FloatArray,
        observed_or_limit: FloatArray,
        censored: NDArray[np.bool_],
        small_sample: bool,
    ) -> GaussianSurfaceFit:
        design = self.basis.design(xy)
        precision, prior_mean = self._prior(len(xy), small_sample, center=0.42)

        def objective(theta: FloatArray) -> tuple[float, FloatArray]:
            beta = theta[:-1]
            eta = theta[-1]
            sigma = float(np.exp(eta))
            mu = design @ beta
            z = (observed_or_limit - mu) / sigma
            exact = ~censored
            nll = 0.0
            grad_beta = np.zeros_like(beta)
            grad_eta = 0.0
            if np.any(exact):
                ze = z[exact]
                nll += float(np.sum(0.5 * ze**2 + eta + 0.5 * np.log(2.0 * np.pi)))
                residual = mu[exact] - observed_or_limit[exact]
                grad_beta += design[exact].T @ (residual / sigma**2)
                grad_eta += float(np.sum(1.0 - ze**2))
            if np.any(censored):
                zc = z[censored]
                log_survival = log_ndtr(-zc)
                nll -= float(np.sum(log_survival))
                log_phi = -0.5 * zc**2 - 0.5 * np.log(2.0 * np.pi)
                mills = np.exp(np.clip(log_phi - log_survival, -50.0, 50.0))
                grad_beta += design[censored].T @ (-mills / sigma)
                grad_eta += float(np.sum(-mills * zc))
            delta = beta - prior_mean
            nll += 0.5 * float(delta @ precision @ delta)
            grad_beta += precision @ delta
            return nll, np.r_[grad_beta, grad_eta]

        pseudo = observed_or_limit + censored.astype(float) * 0.035
        initial_hessian = (design.T @ design) / 0.09**2 + precision
        initial_rhs = (design.T @ pseudo) / 0.09**2 + precision @ prior_mean
        initial_beta = np.linalg.solve(initial_hessian, initial_rhs)
        initial = np.r_[initial_beta, np.log(0.075)]
        result = minimize(
            fun=lambda t: objective(t)[0],
            x0=initial,
            jac=lambda t: objective(t)[1],
            method="L-BFGS-B",
            bounds=[(None, None)] * len(prior_mean) + [(np.log(0.015), np.log(0.35))],
            options={"maxiter": 350, "ftol": 1e-9, "gtol": 2e-6, "maxls": 30},
        )
        if not result.success:
            raise RuntimeError(f"censored surface optimization failed: {result.message}")
        beta = result.x[:-1]
        sigma = float(np.exp(result.x[-1]))
        mu = design @ beta
        z = (observed_or_limit - mu) / sigma
        weights = np.empty_like(z)
        weights[~censored] = 1.0 / sigma**2
        if np.any(censored):
            zc = z[censored]
            log_survival = log_ndtr(-zc)
            log_phi = -0.5 * zc**2 - 0.5 * np.log(2.0 * np.pi)
            mills = np.exp(np.clip(log_phi - log_survival, -50.0, 50.0))
            weights[censored] = np.maximum(mills * (mills - zc), 1e-8) / sigma**2
        hessian = design.T @ (weights[:, None] * design) + precision
        covariance = np.linalg.pinv(hessian, rcond=1e-10)
        return GaussianSurfaceFit(
            basis=self.basis,
            beta=beta,
            covariance=covariance,
            sigma=sigma,
            prior_precision=precision,
            prior_mean=prior_mean,
            n_observations=len(xy),
            response_name="censoring-aware time-to-contact surface",
            units="seconds",
        )

    def _fit_velocity(
        self,
        xy: FloatArray,
        velocity: FloatArray,
        small_sample: bool,
    ) -> GaussianSurfaceFit:
        design = self.basis.design(xy)
        center = float(np.median(velocity))
        precision, prior_mean = self._prior(len(xy), small_sample, center=center)
        # Pilot simplification: empirical residual scale, then conjugate penalized least squares.
        sigma = max(float(np.std(velocity, ddof=1)), 0.08)
        data_precision = 1.0 / sigma**2
        hessian = data_precision * (design.T @ design) + precision
        rhs = data_precision * (design.T @ velocity) + precision @ prior_mean
        beta = np.linalg.solve(hessian, rhs)
        residual = velocity - design @ beta
        sigma = max(float(np.sqrt(np.mean(residual**2))), 0.03)
        hessian = (design.T @ design) / sigma**2 + precision
        rhs = (design.T @ velocity) / sigma**2 + precision @ prior_mean
        beta = np.linalg.solve(hessian, rhs)
        covariance = np.linalg.pinv(hessian, rcond=1e-10)
        return GaussianSurfaceFit(
            basis=self.basis,
            beta=beta,
            covariance=covariance,
            sigma=sigma,
            prior_precision=precision,
            prior_mean=prior_mean,
            n_observations=len(xy),
            response_name="displacement-velocity surface",
            units="metres/second",
            uncertainty_scope="Linear-Gaussian mean-function interval; residual scale and smoothing hyperparameters fixed",
        )

    def fit(self, records: Iterable[PenaltyRecord], keeper_id: str | None = None) -> SurfaceFit:
        selected = [record for record in records if keeper_id is None or record.keeper_id == keeper_id]
        if not selected:
            raise ValueError("no records available for requested keeper")
        ids = {record.keeper_id for record in selected}
        if keeper_id is None and len(ids) != 1:
            raise ValueError("multiple keepers present; pass keeper_id")
        resolved_keeper = keeper_id or next(iter(ids))
        xy = np.asarray([record.ball_crossing_xy_m for record in selected], dtype=float)
        observed_or_limit = np.asarray(
            [record.time_to_contact_s if record.contact else record.flight_time_s for record in selected],
            dtype=float,
        )
        censored = np.asarray([record.censored for record in selected], dtype=bool)
        displacement = np.asarray([record.displacement_m for record in selected], dtype=float)
        flight = np.asarray([record.flight_time_s for record in selected], dtype=float)
        velocity = np.linalg.norm(displacement, axis=1) / flight
        small_sample = len(selected) < SMALL_SAMPLE_THRESHOLD
        banner = (
            f"PRIOR-DOMINATED SMALL-SAMPLE MODE: n={len(selected)} < {SMALL_SAMPLE_THRESHOLD}; "
            "priors are deliberately widened and boundary estimates should not be treated as stable."
            if small_sample
            else ""
        )
        return SurfaceFit(
            time_surface=self._fit_censored_time(xy, observed_or_limit, censored, small_sample),
            velocity_surface=self._fit_velocity(xy, velocity, small_sample),
            small_sample=small_sample,
            banner=banner,
            keeper_id=resolved_keeper,
            n_records=len(selected),
            median_flight_time_s=float(np.median(flight)),
        )
