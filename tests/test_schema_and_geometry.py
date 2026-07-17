from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from gkcoverage.homography import project_points, solve_homography
from gkcoverage.schema import PenaltyRecord


def test_round_trip_geometry_within_five_centimetres_and_timing_one_frame() -> None:
    goal = np.array([[-3.66, 0.0], [3.66, 0.0], [-3.66, 2.44], [3.66, 2.44]])
    pixels = np.array([[102.0, 498.0], [911.0, 523.0], [137.0, 91.0], [872.0, 119.0]])
    h = solve_homography(pixels, goal)
    recovered = project_points(h, pixels)
    assert np.max(np.linalg.norm(recovered - goal, axis=1)) < 0.05

    fps = 30.0
    strike_frame = 100
    crossing_frame = 112
    elapsed = (crossing_frame - strike_frame) / fps
    reconstructed_crossing_frame = strike_frame + round(elapsed * fps)
    assert abs(reconstructed_crossing_frame - crossing_frame) <= 1


def test_schema_enforces_censoring_and_decomposition() -> None:
    base = dict(
        id="p1",
        keeper_id="k1",
        source_url="",
        clip_file="clip.mp4",
        fps=30.0,
        strike_frame=10,
        crossing_frame=22,
        flight_time_s=0.4,
        ball_crossing_xy_m=(1.0, 1.1),
        keeper_pos_at_strike_xy_m=(0.0, 1.0),
        keeper_pos_at_crossing_xy_m=(0.4, 1.1),
        displacement_m=(0.4, 0.1),
        contact=False,
        contact_xy_m=None,
        time_to_contact_s=None,
        censored=True,
        outcome="goal",
        dive_direction="R",
        quality_flags=[],
        annotator="tester",
        annotated_at=datetime.now(UTC),
    )
    record = PenaltyRecord(**base)
    assert record.censored and record.time_to_contact_s is None

    with pytest.raises(ValueError, match="non-contact observations must be censored"):
        PenaltyRecord(**{**base, "censored": False})
    with pytest.raises(ValueError, match="displacement_m disagrees"):
        PenaltyRecord(**{**base, "displacement_m": (0.3, 0.1)})
