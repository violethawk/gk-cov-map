from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from .constants import GOAL_HEIGHT_M, HALF_GOAL_WIDTH_M
from .schema import PenaltyRecord, write_jsonl

FloatArray = NDArray[np.float64]


def ground_truth_time(xy: FloatArray) -> FloatArray:
    """Known smooth time-to-contact field in seconds.

    Positive x is keeper-right. The upper-left is deliberately slower than
    upper-right, yielding a positive left-minus-right time contrast.
    """

    x = xy[:, 0] / HALF_GOAL_WIDTH_M
    y = xy[:, 1] / GOAL_HEIGHT_M
    symmetric = 0.31 + 0.105 * np.abs(x) ** 1.45 + 0.085 * y**1.7 + 0.035 * np.abs(x) * y
    asymmetric = -0.032 * x * (0.25 + 0.75 * y)
    return symmetric + asymmetric


def ground_truth_velocity(xy: FloatArray) -> FloatArray:
    x = xy[:, 0] / HALF_GOAL_WIDTH_M
    y = xy[:, 1] / GOAL_HEIGHT_M
    symmetric = 1.02 + 0.22 * np.abs(x) - 0.13 * y
    asymmetric = 0.055 * x * (0.4 + 0.6 * y)
    return symmetric + asymmetric


def simulate_penalties(
    n: int = 200,
    seed: int = 1,
    keeper_id: str = "sim_keeper",
    concentrated_center: bool = False,
) -> list[PenaltyRecord]:
    rng = np.random.default_rng(seed)
    if concentrated_center:
        x = np.clip(rng.normal(0.0, 1.15, size=n), -HALF_GOAL_WIDTH_M, HALF_GOAL_WIDTH_M)
        y = np.clip(rng.normal(1.05, 0.48, size=n), 0.0, GOAL_HEIGHT_M)
    else:
        x = rng.uniform(-HALF_GOAL_WIDTH_M, HALF_GOAL_WIDTH_M, size=n)
        y = GOAL_HEIGHT_M * rng.beta(1.45, 1.35, size=n)
    xy = np.c_[x, y]
    latent_time = ground_truth_time(xy) + rng.normal(0.0, 0.045, size=n)
    latent_time = np.clip(latent_time, 0.12, 0.85)
    # Plausible penalty flight times; corners tend to have slightly longer paths.
    flight = 0.365 + 0.055 * (np.abs(x) / HALF_GOAL_WIDTH_M) + 0.025 * (y / GOAL_HEIGHT_M)
    flight += rng.normal(0.0, 0.018, size=n)
    flight = np.clip(flight, 0.29, 0.51)
    contact = latent_time <= flight
    velocity = np.clip(ground_truth_velocity(xy) + rng.normal(0.0, 0.11, size=n), 0.35, 1.8)
    direction = np.sign(x)
    direction[direction == 0] = rng.choice([-1.0, 1.0], size=np.sum(direction == 0))
    # Movement roughly follows shot x with a smaller upward component.
    distance = velocity * flight
    angle = np.arctan2(np.maximum(y - 0.8, -0.35), np.maximum(np.abs(x), 0.55))
    dx = direction * distance * np.cos(angle)
    dy = np.maximum(-0.15, distance * np.sin(angle))
    strike_pos = np.c_[rng.normal(0.0, 0.035, size=n), rng.normal(1.0, 0.025, size=n)]
    crossing_pos = strike_pos + np.c_[dx, dy]
    fps_choices = np.array([25.0, 30.0, 50.0, 60.0])
    fps = rng.choice(fps_choices, size=n)
    strike_frame = rng.integers(15, 150, size=n)
    crossing_delta_frames = np.maximum(1, np.rint(flight * fps).astype(int))
    measured_flight = crossing_delta_frames / fps

    records: list[PenaltyRecord] = []
    now = datetime.now(UTC)
    for i in range(n):
        is_contact = bool(contact[i])
        contact_time = float(latent_time[i]) if is_contact else None
        outcome = "save" if is_contact else ("woodwork" if rng.random() < 0.07 else "goal")
        records.append(
            PenaltyRecord(
                id=f"sim-{seed}-{i:04d}",
                keeper_id=keeper_id,
                source_url="synthetic://simulator",
                clip_file=f"sim_{seed}_{i:04d}.mp4",
                fps=float(fps[i]),
                strike_frame=int(strike_frame[i]),
                crossing_frame=int(strike_frame[i] + crossing_delta_frames[i]),
                flight_time_s=float(measured_flight[i]),
                ball_crossing_xy_m=(float(x[i]), float(y[i])),
                keeper_pos_at_strike_xy_m=(float(strike_pos[i, 0]), float(strike_pos[i, 1])),
                keeper_pos_at_crossing_xy_m=(float(crossing_pos[i, 0]), float(crossing_pos[i, 1])),
                displacement_m=(float(dx[i]), float(dy[i])),
                contact=is_contact,
                contact_xy_m=(float(x[i]), float(y[i])) if is_contact else None,
                time_to_contact_s=contact_time,
                censored=not is_contact,
                outcome=outcome,
                dive_direction="R" if dx[i] > 0.04 else ("L" if dx[i] < -0.04 else "none"),
                contact_body_part="hand" if is_contact else None,
                quality_flags=[],
                annotator="simulator",
                annotated_at=now,
            )
        )
    return records


def true_upper_outer_contrast() -> float:
    x = np.linspace(HALF_GOAL_WIDTH_M * 0.52, HALF_GOAL_WIDTH_M * 0.92, 8)
    y = np.linspace(GOAL_HEIGHT_M * 0.55, GOAL_HEIGHT_M * 0.95, 7)
    xx, yy = np.meshgrid(x, y)
    positive = np.c_[xx.ravel(), yy.ravel()]
    negative = positive.copy()
    negative[:, 0] *= -1.0
    return float(np.mean(ground_truth_time(negative) - ground_truth_time(positive)))


def write_simulation(path: str | Path, records: Iterable[PenaltyRecord]) -> None:
    write_jsonl(path, list(records))
