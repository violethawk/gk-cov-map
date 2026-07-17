from __future__ import annotations

import base64
import csv
import json
from html import escape
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from .constants import GOAL_HEIGHT_M, HALF_GOAL_WIDTH_M
from .model import SurfaceFit
from .schema import PenaltyRecord


def goal_grid(nx: int = 181, ny: int = 81) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(-HALF_GOAL_WIDTH_M, HALF_GOAL_WIDTH_M, nx)
    y = np.linspace(0.0, GOAL_HEIGHT_M, ny)
    xx, yy = np.meshgrid(x, y)
    return xx, yy, np.c_[xx.ravel(), yy.ravel()]


def _draw_goal(ax: plt.Axes) -> None:
    ax.plot([-HALF_GOAL_WIDTH_M, -HALF_GOAL_WIDTH_M], [0, GOAL_HEIGHT_M], linewidth=3, color="black")
    ax.plot([HALF_GOAL_WIDTH_M, HALF_GOAL_WIDTH_M], [0, GOAL_HEIGHT_M], linewidth=3, color="black")
    ax.plot([-HALF_GOAL_WIDTH_M, HALF_GOAL_WIDTH_M], [GOAL_HEIGHT_M, GOAL_HEIGHT_M], linewidth=3, color="black")
    ax.set_xlim(-HALF_GOAL_WIDTH_M - 0.05, HALF_GOAL_WIDTH_M + 0.05)
    ax.set_ylim(-0.03, GOAL_HEIGHT_M + 0.05)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Goal-plane x (m; goalkeeper right positive)")
    ax.set_ylabel("Goal-plane y (m)")


PRIOR_DOMINANCE_HATCH = 0.78


def _hatch_prior_dominated(ax: plt.Axes, xx: np.ndarray, yy: np.ndarray, dominance: np.ndarray):
    """Hatch the region where the prior rather than the data drives the surface."""
    mask = np.ma.masked_where(dominance < PRIOR_DOMINANCE_HATCH, dominance)
    return ax.contourf(
        xx, yy, mask, levels=[PRIOR_DOMINANCE_HATCH, 1.01], colors="none", hatches=["///"]
    )


def _uncertainty_alpha(sd: np.ndarray, prior_dominance: np.ndarray) -> np.ndarray:
    sd_scaled = sd / max(float(np.nanpercentile(sd, 95)), 1e-9)
    confidence = 1.0 - np.clip(0.55 * sd_scaled + 0.45 * prior_dominance, 0.0, 1.0)
    return 0.22 + 0.78 * confidence


def _add_uncertainty_key(ax: plt.Axes) -> None:
    ax.text(
        0.01,
        0.02,
        "Uncertainty is inside the map: faded = wider posterior interval; /// = prior-dominated",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
    )


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_time_surface(fit: SurfaceFit, out_path: Path) -> dict[str, np.ndarray]:
    xx, yy, points = goal_grid()
    prediction = fit.time_surface.predict(points)
    shape = xx.shape
    mean = prediction["mean"].reshape(shape)
    sd = prediction["sd"].reshape(shape)
    lower = prediction["lower"].reshape(shape)
    upper = prediction["upper"].reshape(shape)
    dominance = prediction["prior_dominance"].reshape(shape)
    alpha = _uncertainty_alpha(sd, dominance)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    image = ax.imshow(
        mean,
        extent=[-HALF_GOAL_WIDTH_M, HALF_GOAL_WIDTH_M, 0, GOAL_HEIGHT_M],
        origin="lower",
        aspect="equal",
        cmap="viridis_r",
        alpha=alpha,
    )
    _hatch_prior_dominated(ax, xx, yy, dominance)
    _draw_goal(ax)
    title = f"Censoring-aware time-to-contact coverage — {fit.keeper_id} (n={fit.n_records})"
    ax.set_title(title)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("Posterior mean time to contact (s); lower is faster")
    _add_uncertainty_key(ax)
    if fit.banner:
        ax.text(
            0.5,
            1.02,
            fit.banner,
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="bottom",
            bbox={"facecolor": "#fff3cd", "edgecolor": "#7a5d00", "alpha": 0.95},
        )
    _save_figure(fig, out_path)
    return {"x": xx, "y": yy, "mean": mean, "sd": sd, "lower": lower, "upper": upper, "prior_dominance": dominance}


def render_asymmetry_surface(fit: SurfaceFit, out_path: Path) -> dict[str, np.ndarray]:
    xx, yy, points = goal_grid()
    component = fit.time_surface.predict_components(points)["asymmetric"]
    shape = xx.shape
    mean_ms = 1000.0 * component["mean"].reshape(shape)
    sd_ms = 1000.0 * component["sd"].reshape(shape)
    lower_ms = 1000.0 * component["lower"].reshape(shape)
    upper_ms = 1000.0 * component["upper"].reshape(shape)
    dominance = component["prior_dominance"].reshape(shape)
    alpha = _uncertainty_alpha(sd_ms, dominance)
    bound = max(float(np.nanpercentile(np.abs(mean_ms), 98)), 1.0)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    image = ax.imshow(
        mean_ms,
        extent=[-HALF_GOAL_WIDTH_M, HALF_GOAL_WIDTH_M, 0, GOAL_HEIGHT_M],
        origin="lower",
        aspect="equal",
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound),
        alpha=alpha,
    )
    _hatch_prior_dominated(ax, xx, yy, dominance)
    _draw_goal(ax)
    ax.set_title(f"Learned asymmetry component A(x,y) — {fit.keeper_id}")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("A(x,y), milliseconds added to symmetric surface")
    _add_uncertainty_key(ax)
    summary = fit.asymmetry_summary()
    ax.text(
        0.99,
        0.02,
        (
            f"Upper-outer left − right: {1000*float(summary['time_left_minus_right_s']):+.1f} ms\n"
            f"95% CI [{1000*float(summary['time_ci_lower_s']):+.1f}, "
            f"{1000*float(summary['time_ci_upper_s']):+.1f}] ms"
        ),
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="right",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    if fit.banner:
        ax.text(
            0.5,
            1.02,
            fit.banner,
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="bottom",
            bbox={"facecolor": "#fff3cd", "edgecolor": "#7a5d00", "alpha": 0.95},
        )
    _save_figure(fig, out_path)
    return {"x": xx, "y": yy, "mean_ms": mean_ms, "sd_ms": sd_ms, "lower_ms": lower_ms, "upper_ms": upper_ms, "prior_dominance": dominance}


def render_raw_overlay(records: Iterable[PenaltyRecord], keeper_id: str, out_path: Path) -> None:
    selected = [record for record in records if record.keeper_id == keeper_id]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    style = {
        "goal": {"marker": "o", "label": "goal"},
        "save": {"marker": "s", "label": "save"},
        "woodwork": {"marker": "x", "label": "woodwork"},
    }
    for outcome, attrs in style.items():
        points = np.asarray([r.ball_crossing_xy_m for r in selected if r.outcome == outcome], dtype=float)
        if len(points):
            ax.scatter(points[:, 0], points[:, 1], s=24, marker=attrs["marker"], label=attrs["label"], alpha=0.8)
    _draw_goal(ax)
    ax.set_title(f"Raw crossing points — {keeper_id} (constant marker size)")
    ax.legend(loc="upper center", ncol=3, frameon=True)
    _save_figure(fig, out_path)


def render_velocity_surface(fit: SurfaceFit, out_path: Path) -> None:
    xx, yy, points = goal_grid()
    prediction = fit.velocity_surface.predict(points)
    shape = xx.shape
    mean = prediction["mean"].reshape(shape)
    sd = prediction["sd"].reshape(shape)
    dominance = prediction["prior_dominance"].reshape(shape)
    alpha = _uncertainty_alpha(sd, dominance)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    image = ax.imshow(
        mean,
        extent=[-HALF_GOAL_WIDTH_M, HALF_GOAL_WIDTH_M, 0, GOAL_HEIGHT_M],
        origin="lower",
        aspect="equal",
        cmap="plasma",
        alpha=alpha,
    )
    _hatch_prior_dominated(ax, xx, yy, dominance)
    _draw_goal(ax)
    ax.set_title(f"Companion displacement-velocity surface — {fit.keeper_id}")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("Posterior mean keeper displacement speed (m/s)")
    _add_uncertainty_key(ax)
    _save_figure(fig, out_path)


def _png_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def write_grid_csv(path: Path, time_grid: dict[str, np.ndarray], asym_grid: dict[str, np.ndarray]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "x_m", "y_m", "time_mean_s", "time_sd_s", "time_ci_lower_s", "time_ci_upper_s",
                "time_prior_dominance", "asymmetry_mean_ms", "asymmetry_sd_ms", "asymmetry_ci_lower_ms",
                "asymmetry_ci_upper_ms", "asymmetry_prior_dominance",
            ]
        )
        arrays = [
            time_grid["x"], time_grid["y"], time_grid["mean"], time_grid["sd"], time_grid["lower"],
            time_grid["upper"], time_grid["prior_dominance"], asym_grid["mean_ms"], asym_grid["sd_ms"],
            asym_grid["lower_ms"], asym_grid["upper_ms"], asym_grid["prior_dominance"],
        ]
        for values in zip(*(array.ravel() for array in arrays), strict=True):
            writer.writerow([f"{float(value):.9g}" for value in values])


def export_report(fit: SurfaceFit, records: list[PenaltyRecord], output_dir: str | Path) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "coverage_png": out / "coverage_surface.png",
        "asymmetry_png": out / "asymmetry_surface.png",
        "raw_png": out / "raw_overlay.png",
        "velocity_png": out / "displacement_velocity.png",
        "html": out / "report.html",
        "summary": out / "summary.json",
        "grid": out / "surface_grid.csv",
    }
    time_grid = render_time_surface(fit, paths["coverage_png"])
    asym_grid = render_asymmetry_surface(fit, paths["asymmetry_png"])
    render_raw_overlay(records, fit.keeper_id, paths["raw_png"])
    render_velocity_surface(fit, paths["velocity_png"])
    write_grid_csv(paths["grid"], time_grid, asym_grid)

    summary = fit.asymmetry_summary()
    payload = {
        "keeper_id": fit.keeper_id,
        "n_records": fit.n_records,
        "small_sample": fit.small_sample,
        "banner": fit.banner,
        "asymmetry": summary,
        "time_surface_uncertainty": fit.time_surface.uncertainty_scope,
        "velocity_surface_uncertainty": fit.velocity_surface.uncertainty_scope,
    }
    paths["summary"].write_text(json.dumps(payload, indent=2), encoding="utf-8")

    cards = []
    for title, key in [
        ("Coverage surface", "coverage_png"),
        ("Asymmetry component", "asymmetry_png"),
        ("Raw data", "raw_png"),
        ("Displacement velocity", "velocity_png"),
    ]:
        cards.append(
            f"<section><h2>{escape(title)}</h2>"
            f'<img src="{_png_data_uri(paths[key])}" alt="{escape(title)}"></section>'
        )
    banner_html = f'<div class="banner">{escape(fit.banner)}</div>' if fit.banner else ""
    # keeper_id arrives from the JSONL, which the annotation tool fills from a free-text
    # field, so it is untrusted text rather than markup: interpolated raw it both breaks
    # the document on an ordinary id like "A & B" and executes anything a shared JSONL
    # cares to carry. The summary lands in a <pre>, which a "</pre>" would escape, and
    # the banner is generated here but escaped anyway so the rule holds at every seam.
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Goalkeeper coverage report</title>
<style>body{{font-family:system-ui,sans-serif;margin:2rem;max-width:1200px}}img{{width:100%;height:auto;border:1px solid #ddd}}section{{margin:2rem 0}}.banner{{padding:1rem;background:#fff3cd;border:1px solid #7a5d00;font-weight:700}}code{{background:#f4f4f4;padding:.15rem .3rem}}</style></head>
<body><h1>Goalkeeper coverage surface: {escape(fit.keeper_id)}</h1>{banner_html}
<p>n={fit.n_records}. Exact saves and right-censored non-contact shots were fit together. Uncertainty is encoded inside each heatmap by fading and hatching.</p>
<pre>{escape(json.dumps(summary, indent=2))}</pre>{''.join(cards)}</body></html>"""
    paths["html"].write_text(html, encoding="utf-8")
    return paths
