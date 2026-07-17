from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Outcome = Literal["goal", "save", "woodwork"]
DiveDirection = Literal["L", "R", "none"]


class PenaltyRecord(BaseModel):
    """One manually annotated penalty kick.

    Coordinates use metres on the goal plane, origin at bottom-centre,
    positive x to the goalkeeper's right as seen from behind the shooter,
    and positive y upward.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    keeper_id: str
    source_url: str = ""
    clip_file: str
    fps: float = Field(gt=0)
    strike_frame: int = Field(ge=0)
    crossing_frame: int = Field(ge=0)
    flight_time_s: float = Field(gt=0)
    ball_crossing_xy_m: tuple[float, float]
    keeper_pos_at_strike_xy_m: tuple[float, float]
    keeper_pos_at_crossing_xy_m: tuple[float, float]
    displacement_m: tuple[float, float]
    contact: bool
    contact_xy_m: tuple[float, float] | None
    time_to_contact_s: float | None
    censored: bool
    outcome: Outcome
    dive_direction: DiveDirection
    contact_body_part: str | None = None
    quality_flags: list[str] = Field(default_factory=list)
    annotator: str
    annotated_at: datetime
    annotation_metadata: dict[str, object] | None = None

    @field_validator(
        "ball_crossing_xy_m",
        "keeper_pos_at_strike_xy_m",
        "keeper_pos_at_crossing_xy_m",
        "displacement_m",
        "contact_xy_m",
    )
    @classmethod
    def validate_pair(cls, value: tuple[float, float] | None) -> tuple[float, float] | None:
        if value is not None and len(value) != 2:
            raise ValueError("coordinate must contain exactly two values")
        return value

    @model_validator(mode="after")
    def validate_event_logic(self) -> "PenaltyRecord":
        expected_flight = (self.crossing_frame - self.strike_frame) / self.fps
        if self.crossing_frame <= self.strike_frame:
            raise ValueError("crossing_frame must be after strike_frame")
        if abs(self.flight_time_s - expected_flight) > max(1e-6, 0.25 / self.fps):
            raise ValueError("flight_time_s disagrees with frames/FPS")
        expected_displacement = (
            self.keeper_pos_at_crossing_xy_m[0] - self.keeper_pos_at_strike_xy_m[0],
            self.keeper_pos_at_crossing_xy_m[1] - self.keeper_pos_at_strike_xy_m[1],
        )
        if any(abs(a - b) > 1e-4 for a, b in zip(self.displacement_m, expected_displacement)):
            raise ValueError("displacement_m disagrees with keeper positions")
        if self.contact:
            if self.censored:
                raise ValueError("contact observations cannot be censored")
            if self.time_to_contact_s is None or self.contact_xy_m is None:
                raise ValueError("contact observations require contact time and location")
            if self.outcome != "save":
                raise ValueError("prototype contact=true is reserved for saves")
        else:
            if not self.censored:
                raise ValueError("non-contact observations must be censored")
            if self.time_to_contact_s is not None or self.contact_xy_m is not None:
                raise ValueError("non-contact observations cannot carry contact time/location")
        return self


def read_jsonl(path: str | Path) -> list[PenaltyRecord]:
    import json

    records: list[PenaltyRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(PenaltyRecord.model_validate(json.loads(line)))
            except Exception as exc:  # pragma: no cover - message wrapper
                raise ValueError(f"invalid record at line {line_number}: {exc}") from exc
    return records


def write_jsonl(path: str | Path, records: list[PenaltyRecord]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")
