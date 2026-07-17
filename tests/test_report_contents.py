from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from gkcoverage.constants import GOAL_HEIGHT_M, HALF_GOAL_WIDTH_M
from gkcoverage.model import CoverageSurfaceModel
from gkcoverage.schema import PenaltyRecord
from gkcoverage.simulate import simulate_penalties
from gkcoverage.visualize import (
    PRIOR_DOMINANCE_HATCH,
    _hatch_prior_dominated,
    _uncertainty_alpha,
    export_report,
    goal_grid,
)

# The region asymmetry_summary contrasts over. Duplicated on purpose: these tests
# pin the reported contract, so moving the region should have to be done twice.
UPPER_OUTER_X = np.linspace(HALF_GOAL_WIDTH_M * 0.52, HALF_GOAL_WIDTH_M * 0.92, 8)
UPPER_OUTER_Y = np.linspace(GOAL_HEIGHT_M * 0.55, GOAL_HEIGHT_M * 0.95, 7)


def _upper_outer_points() -> np.ndarray:
    xx, yy = np.meshgrid(UPPER_OUTER_X, UPPER_OUTER_Y)
    return np.c_[xx.ravel(), yy.ravel()]


def _mirror(records: list[PenaltyRecord]) -> list[PenaltyRecord]:
    """Reflect every record through the goal centre line."""
    flip = {"L": "R", "R": "L", "none": "none"}
    return [
        record.model_copy(
            update={
                "ball_crossing_xy_m": (-record.ball_crossing_xy_m[0], record.ball_crossing_xy_m[1]),
                "keeper_pos_at_strike_xy_m": (
                    -record.keeper_pos_at_strike_xy_m[0],
                    record.keeper_pos_at_strike_xy_m[1],
                ),
                "keeper_pos_at_crossing_xy_m": (
                    -record.keeper_pos_at_crossing_xy_m[0],
                    record.keeper_pos_at_crossing_xy_m[1],
                ),
                "displacement_m": (-record.displacement_m[0], record.displacement_m[1]),
                "contact_xy_m": (
                    None
                    if record.contact_xy_m is None
                    else (-record.contact_xy_m[0], record.contact_xy_m[1])
                ),
                "dive_direction": flip[record.dive_direction],
            }
        )
        for record in records
    ]


def test_summary_names_the_side_the_data_says_is_weaker() -> None:
    # The headline claim of the whole report. The simulator makes the upper left
    # slower, so left is weaker and left-minus-right time is positive. Mirroring
    # the shots must move the claim to the other post rather than relabel it, so
    # a hardcoded or inverted side cannot pass.
    records = simulate_penalties(200, seed=0)
    summary = CoverageSurfaceModel().fit(records).asymmetry_summary()
    assert summary["weaker_side_by_time"] == "left"
    assert summary["time_left_minus_right_s"] > 0

    mirrored = CoverageSurfaceModel().fit(_mirror(records)).asymmetry_summary()
    assert mirrored["weaker_side_by_time"] == "right"
    assert mirrored["time_left_minus_right_s"] < 0
    assert mirrored["time_left_minus_right_s"] == pytest.approx(
        -summary["time_left_minus_right_s"], rel=1e-6
    )
    assert mirrored["reach_left_minus_right_cm_at_median_flight"] == pytest.approx(
        -summary["reach_left_minus_right_cm_at_median_flight"], rel=1e-6
    )


def test_reach_is_reported_in_centimetres_at_the_median_flight_time() -> None:
    # reach = displacement-speed contrast (m/s) * median flight (s) * 100, so a
    # wrong unit factor shows up as a scale error against the velocity surface.
    fit = CoverageSurfaceModel().fit(simulate_penalties(200, seed=0))
    summary = fit.asymmetry_summary()
    speed = fit.velocity_surface.paired_contrast(_upper_outer_points())

    assert summary["median_flight_time_s"] == pytest.approx(
        float(np.median([r.flight_time_s for r in simulate_penalties(200, seed=0)]))
    )
    for reach_key, speed_key in (
        ("reach_left_minus_right_cm_at_median_flight", "estimate"),
        ("reach_ci_lower_cm", "lower"),
        ("reach_ci_upper_cm", "upper"),
    ):
        expected_cm = speed[speed_key] * summary["median_flight_time_s"] * 100.0
        assert summary[reach_key] == pytest.approx(expected_cm, rel=1e-9)


def test_velocity_surface_fits_displacement_divided_by_flight_time() -> None:
    records = simulate_penalties(200, seed=0)
    fit = CoverageSurfaceModel().fit(records)
    displacement = np.array([r.displacement_m for r in records])
    flight = np.array([r.flight_time_s for r in records])
    observed = np.linalg.norm(displacement, axis=1) / flight
    predicted = fit.velocity_surface.predict(np.array([r.ball_crossing_xy_m for r in records]))["mean"]

    assert fit.velocity_surface.units == "metres/second"
    # Multiplying by flight instead of dividing would land near 0.18 m/s.
    assert np.median(predicted) == pytest.approx(np.median(observed), abs=0.05)
    assert np.corrcoef(predicted, observed)[0, 1] > 0.4


def test_component_split_is_symmetric_and_antisymmetric() -> None:
    # The asymmetry map renders the A block. If it were fed the symmetric block
    # the picture would be of S while still looking plausible.
    fit = CoverageSurfaceModel().fit(simulate_penalties(200, seed=0))
    right = np.c_[np.linspace(0.2, 3.5, 24), np.linspace(0.2, 2.3, 24)]
    left = right.copy()
    left[:, 0] *= -1.0
    on_right = fit.time_surface.predict_components(right)
    on_left = fit.time_surface.predict_components(left)

    assert np.max(np.abs(on_right["asymmetric"]["mean"])) > 1e-3  # else the split proves nothing
    assert np.allclose(on_right["symmetric"]["mean"], on_left["symmetric"]["mean"], atol=1e-12)
    assert np.allclose(on_right["asymmetric"]["mean"], -on_left["asymmetric"]["mean"], atol=1e-12)
    assert np.allclose(
        fit.time_surface.predict(right)["mean"],
        on_right["symmetric"]["mean"] + on_right["asymmetric"]["mean"],
        atol=1e-12,
    )


def test_uncertainty_alpha_fades_with_uncertainty() -> None:
    # "Uncertainty is inside the map" is only true if alpha actually varies.
    alpha = _uncertainty_alpha(np.array([0.01, 0.05, 0.20]), np.array([0.02, 0.30, 0.95]))
    assert alpha[0] > alpha[1] > alpha[2]
    assert np.ptp(alpha) > 0.1


def test_prior_dominated_regions_are_actually_hatched() -> None:
    xx, yy = np.meshgrid(np.linspace(-3.0, 3.0, 40), np.linspace(0.0, 2.4, 20))
    fig, ax = plt.subplots()
    try:
        dominated = _hatch_prior_dominated(
            ax, xx, yy, np.where(xx < -2.0, PRIOR_DOMINANCE_HATCH + 0.1, 0.05)
        )
        assert dominated.hatches == ["///"]
        assert sum(len(path.vertices) for path in dominated.get_paths()) > 0

        clear = _hatch_prior_dominated(ax, xx, yy, np.full_like(xx, 0.05))
        assert sum(len(path.vertices) for path in clear.get_paths()) == 0
    finally:
        plt.close(fig)


def test_every_rendered_surface_hatches_its_prior_dominated_region(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The helper working is not enough: each map has to call it. Dropping the call
    # from one surface would quietly ship an unmarked prior-dominated map.
    import gkcoverage.visualize as visualize

    real = visualize._hatch_prior_dominated
    seen: list[np.ndarray] = []

    def spy(ax, xx, yy, dominance):  # type: ignore[no-untyped-def]
        seen.append(np.asarray(dominance).copy())
        return real(ax, xx, yy, dominance)

    monkeypatch.setattr(visualize, "_hatch_prior_dominated", spy)
    records = simulate_penalties(200, seed=4, concentrated_center=True)
    fit = CoverageSurfaceModel().fit(records)
    export_report(fit, records, tmp_path)

    assert len(seen) == 3  # coverage, asymmetry, velocity
    assert any(np.any(dominance >= PRIOR_DOMINANCE_HATCH) for dominance in seen)


def test_grid_csv_holds_the_fitted_surface(tmp_path: Path) -> None:
    records = simulate_penalties(200, seed=0)
    fit = CoverageSurfaceModel().fit(records)
    paths = export_report(fit, records, tmp_path)

    with paths["grid"].open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header, data = rows[0], rows[1:]
    _, _, points = goal_grid()
    assert len(data) == points.shape[0]
    assert {len(row) for row in data} == {len(header)}  # a column swap desynchronises these

    column = {name: index for index, name in enumerate(header)}
    sample = [0, len(data) // 3, len(data) // 2, len(data) - 1]
    sampled_xy = np.array(
        [[float(data[i][column["x_m"]]), float(data[i][column["y_m"]])] for i in sample]
    )
    expected = fit.time_surface.predict(sampled_xy)
    for position, row_index in enumerate(sample):
        row = data[row_index]
        assert float(row[column["time_mean_s"]]) == pytest.approx(expected["mean"][position], rel=1e-6)
        assert float(row[column["time_sd_s"]]) == pytest.approx(expected["sd"][position], rel=1e-6)
        assert float(row[column["time_ci_lower_s"]]) == pytest.approx(
            expected["lower"][position], rel=1e-6
        )


def test_report_html_escapes_the_keeper_id(tmp_path: Path) -> None:
    # keeper_id reaches the report from the JSONL, which the annotation tool fills from
    # a free-text field, so it is untrusted text. report.html embeds its images as data
    # URIs specifically so it can be handed to someone else, and interpolated raw this
    # payload runs when they open it.
    payload = "<script>alert(document.domain)</script>"
    records = simulate_penalties(200, seed=0, keeper_id=payload)
    fit = CoverageSurfaceModel().fit(records)
    html = export_report(fit, records, tmp_path)["html"].read_text(encoding="utf-8")

    assert payload not in html
    assert "&lt;script&gt;alert(document.domain)&lt;/script&gt;" in html
    # The id must still be legible once escaped, not dropped.
    assert "Goalkeeper coverage surface:" in html


def test_report_html_survives_an_ordinary_id_that_is_not_markup(tmp_path: Path) -> None:
    # The bug bites without an attacker: "&" and "<" are legal in a keeper id and would
    # silently corrupt the document. summary.json holds the same id and must keep it raw,
    # since JSON escaping is not HTML escaping.
    records = simulate_penalties(200, seed=0, keeper_id="A & B <1>")
    fit = CoverageSurfaceModel().fit(records)
    paths = export_report(fit, records, tmp_path)
    html = paths["html"].read_text(encoding="utf-8")

    assert "A &amp; B &lt;1&gt;" in html
    assert json.loads(paths["summary"].read_text(encoding="utf-8"))["keeper_id"] == "A & B <1>"


def test_summary_json_reports_the_fitted_asymmetry(tmp_path: Path) -> None:
    records = simulate_penalties(200, seed=0)
    fit = CoverageSurfaceModel().fit(records)
    payload = json.loads(export_report(fit, records, tmp_path)["summary"].read_text(encoding="utf-8"))
    summary = fit.asymmetry_summary()

    assert payload["keeper_id"] == fit.keeper_id
    assert payload["n_records"] == 200
    assert payload["small_sample"] is False
    assert payload["asymmetry"]["weaker_side_by_time"] == summary["weaker_side_by_time"]
    for key in (
        "time_left_minus_right_s",
        "time_ci_lower_s",
        "time_ci_upper_s",
        "reach_left_minus_right_cm_at_median_flight",
        "median_flight_time_s",
    ):
        assert payload["asymmetry"][key] == pytest.approx(summary[key])
    assert payload["asymmetry"]["time_ci_lower_s"] < payload["asymmetry"]["time_left_minus_right_s"]
