from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np

from gkcoverage.model import CoverageSurfaceModel
from gkcoverage.simulate import simulate_penalties
from gkcoverage.visualize import export_report


def test_prior_dominated_corner_is_hatched_and_center_is_data_rich(tmp_path: Path) -> None:
    records = simulate_penalties(200, seed=4, concentrated_center=True)
    fit = CoverageSurfaceModel().fit(records)
    points = np.array([[0.0, 1.1], [-3.65, 2.43]])
    dominance = fit.time_surface.predict(points)["prior_dominance"]
    assert dominance[0] < 0.25
    assert dominance[1] > 0.78

    paths = export_report(fit, records, tmp_path)
    for path in paths.values():
        assert path.exists() and path.stat().st_size > 0
    html = paths["html"].read_text(encoding="utf-8")
    assert "fading and hatching" in html
    assert "data:image/png;base64" in html


def test_small_sample_banner_appears_in_all_model_outputs(tmp_path: Path) -> None:
    records = simulate_penalties(80, seed=2)
    fit = CoverageSurfaceModel().fit(records)
    assert fit.small_sample
    assert "PRIOR-DOMINATED SMALL-SAMPLE MODE" in fit.banner
    paths = export_report(fit, records, tmp_path)
    # The banner reads "n=80 < 150", so it carries a literal "<" into the document and
    # has to be escaped like any other text. Browsers recover from a "<" followed by a
    # space, which is why the raw form looked fine, but it is not valid markup.
    assert escape(fit.banner) in paths["html"].read_text(encoding="utf-8")
    assert "n=80 &lt; 150" in paths["html"].read_text(encoding="utf-8")
    # summary.json is JSON, not HTML, so the banner stays raw there.
    assert fit.banner in paths["summary"].read_text(encoding="utf-8")
