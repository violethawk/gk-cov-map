from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from gkcoverage.homography import project_points
from gkcoverage.schema import PenaltyRecord

ROOT = Path(__file__).parents[1]

# A complete, internally consistent annotation in browser terms: four calibration
# clicks plus the event markers, all in video pixels. The calibration quad is the
# one the geometry round-trip test uses, so these project onto the real goal plane.
STATE_JS = """
const state = {
  clipFile: {name: 'clip.mp4'},
  fps: 30,
  fpsMetadata: {fps: 30, variableFrameRate: false, timingSource: 'stts deltas'},
  strikeFrame: 20,
  crossingFrame: 32,
  markers: {
    left_post_base: [100, 500],
    right_post_base: [900, 520],
    left_crossbar: [130, 100],
    right_crossbar: [870, 120],
    ball_crossing: [700, 200],
    keeper_strike: [500, 400],
    keeper_crossing: [640, 260],
    contact_location: [700, 205],
  },
  keeperId: '  keeper_1  ',
  annotator: '  tester  ',
  sourceUrl: 'https://example.test/clip',
  outcome: 'save',
  diveDirection: 'R',
  bodyPart: 'hand',
  qualityFlags: ['occlusion'],
  filePickerAvailable: true,
  jsonlChosen: true,
};
"""


def run_node(script: str, cwd: Path) -> dict[str, object]:
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


def build_record(mutate: str = "") -> dict[str, object]:
    """Run the browser's record builder over STATE_JS, after applying `mutate`."""
    return run_node(
        "import {buildRecord} from './annotation_tool/record.js';\n"
        + STATE_JS
        + mutate
        + "\nconsole.log(JSON.stringify(buildRecord(state,"
        " {id: 'test-record', annotatedAt: '2026-07-17T10:00:00Z'})));",
        ROOT,
    )


def validation_errors(mutate: str = "") -> list[str]:
    return run_node(
        "import {validateAnnotation} from './annotation_tool/record.js';\n"
        + STATE_JS
        + mutate
        + "\nconsole.log(JSON.stringify(validateAnnotation(state)));",
        ROOT,
    )


@pytest.mark.parametrize("outcome", ["save", "goal", "woodwork"])
def test_browser_records_satisfy_the_python_schema(outcome: str) -> None:
    # The annotation tool writes the JSONL that gkcoverage reads, and nothing else
    # checks the two agree. PenaltyRecord forbids extra keys and re-derives flight
    # time from frames/FPS and displacement from the keeper positions, so a renamed
    # field, a dropped one, or a unit slip in the browser fails here rather than
    # surfacing as a bad fit.
    record = build_record("state.outcome = " + json.dumps(outcome) + ";")
    parsed = PenaltyRecord.model_validate(record)

    assert parsed.outcome == outcome
    assert parsed.fps == 30.0
    assert parsed.flight_time_s == pytest.approx((32 - 20) / 30)
    assert parsed.keeper_id == "keeper_1"  # the browser trims before writing
    assert parsed.annotator == "tester"


def test_save_is_exact_and_non_contact_is_censored() -> None:
    # The model reads time_to_contact_s for contacts and flight_time_s as the
    # censoring limit for everything else, so these two flags decide which
    # likelihood a shot lands in. Getting them backwards would still parse.
    save = PenaltyRecord.model_validate(build_record("state.outcome = 'save';"))
    assert save.contact and not save.censored
    assert save.contact_xy_m is not None
    # The crossing/contact frame is the terminal event, so a save's contact time is
    # its flight time rather than an independent quantity.
    assert save.time_to_contact_s == pytest.approx(save.flight_time_s)

    goal = PenaltyRecord.model_validate(build_record("state.outcome = 'goal';"))
    assert goal.censored and not goal.contact
    assert goal.time_to_contact_s is None
    assert goal.contact_xy_m is None


def test_recorded_homography_maps_the_recorded_clicks_onto_the_goal_plane() -> None:
    # README section 3 says annotation_metadata keeps every projected coordinate
    # auditable. That holds only if the stored homography actually maps the stored
    # calibration clicks onto the goal plane, so re-project them with the Python
    # implementation and hold it to the documented 5 cm bound. This also pins the
    # browser's 8x8 solve against the normalized DLT independently.
    record = build_record()
    metadata = record["annotation_metadata"]
    matrix = np.asarray(metadata["homography_pixel_to_goal"], dtype=float).reshape(3, 3)
    order = ["left_post_base", "right_post_base", "left_crossbar", "right_crossbar"]
    pixels = np.asarray([metadata["goal_reference_pixels"][name] for name in order], dtype=float)
    expected = np.asarray([[-3.66, 0.0], [3.66, 0.0], [-3.66, 2.44], [3.66, 2.44]])

    assert np.max(np.abs(project_points(matrix, pixels) - expected)) < 0.05
    assert metadata["fps_metadata"]["timingSource"] == "stts deltas"


def test_validation_blocks_annotations_the_schema_would_reject() -> None:
    # Each of these would either fail PenaltyRecord or write a silently wrong record,
    # so the browser has to refuse them at the point of annotation.
    assert validation_errors() == []
    cases = {
        "state.clipFile = null;": "Load an MP4 clip.",
        "state.fps = null;": "FPS metadata is unavailable.",
        "state.filePickerAvailable = false;": "Direct append requires Chromium File System Access API.",
        "state.jsonlChosen = false;": "Choose or create the append-only JSONL destination.",
        "state.keeperId = '   ';": "Keeper ID is required.",
        "state.annotator = '';": "Annotator is required.",
        "state.strikeFrame = null;": "Mark the strike frame.",
        "state.crossingFrame = null;": "Mark the crossing/contact frame.",
        "state.crossingFrame = state.strikeFrame;": "Crossing frame must follow strike frame.",
        "delete state.markers.ball_crossing;": "Mark ball crossing.",
        "delete state.markers.keeper_strike;": "Mark keeper COM at strike.",
        "delete state.markers.contact_location;": "Save outcome requires a contact location.",
        "state.outcome = 'excluded';": "Off-target clips are excluded and cannot be appended.",
    }
    for mutation, expected in cases.items():
        assert expected in validation_errors(mutation), mutation


def test_contact_location_is_required_only_for_saves() -> None:
    # A goal needs no contact point, so requiring it unconditionally would block
    # every censored shot the model depends on.
    drop = "delete state.markers.contact_location;"
    assert validation_errors(drop + "state.outcome = 'goal';") == []
    assert validation_errors(drop + "state.outcome = 'save';") != []


def test_calibration_clicks_that_define_no_homography_are_rejected() -> None:
    # Four markers can all be present and still be collinear, which only the solve
    # catches. Counting markers rather than solving would let this through.
    errors = validation_errors(
        "state.markers.left_post_base = [100, 500];"
        "state.markers.right_post_base = [900, 500];"
        "state.markers.left_crossbar = [300, 500];"
        "state.markers.right_crossbar = [700, 500];"
    )
    assert any("Degenerate homography points" in error for error in errors)


def test_build_record_refuses_to_write_an_invalid_annotation() -> None:
    # Validation is advisory in the UI; this is the gate that keeps a bad record out
    # of the append-only file.
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        build_record("state.outcome = 'excluded';")
    assert "Resolve validation errors before saving" in excinfo.value.stderr


def test_every_element_app_js_reaches_for_exists_in_the_markup() -> None:
    # app.js resolves its elements at import time, so a renamed or deleted id in
    # index.html breaks the tool on load rather than at the call site. Nothing else
    # links the two files.
    app = (ROOT / "annotation_tool" / "app.js").read_text(encoding="utf-8")
    html = (ROOT / "annotation_tool" / "index.html").read_text(encoding="utf-8")
    wanted = set(re.findall(r'querySelector(?:All)?\("#([A-Za-z0-9_-]+)"\)', app))
    assert wanted, "selector regex found nothing, so this test proves nothing"

    assert wanted <= set(re.findall(r'id="([A-Za-z0-9_-]+)"', html))


def test_marker_buttons_match_the_markers_the_record_module_knows() -> None:
    # The canvas click handler advances through MARKER_LABELS in order and toggles the
    # matching [data-marker] button. If the two lists drift the workflow skips or
    # stalls on a marker with no button, which no amount of record validation catches.
    html = (ROOT / "annotation_tool" / "index.html").read_text(encoding="utf-8")
    labels = run_node(
        "import {MARKER_LABELS} from './annotation_tool/record.js';"
        "console.log(JSON.stringify(Object.keys(MARKER_LABELS)));",
        ROOT,
    )
    assert set(re.findall(r'data-marker="([a-z_]+)"', html)) == set(labels)


def test_browser_geometry_implementation_round_trips() -> None:
    root = Path(__file__).parents[1]
    payload = run_node(
        """
        import {solveHomography,projectPoint,GOAL_PLANE_POINTS} from './annotation_tool/geometry.js';
        const pixels=[[100,500],[900,520],[130,100],[870,120]];
        const h=solveHomography(pixels);
        const projected=pixels.map((p)=>projectPoint(h,p));
        const errors=projected.map((p,i)=>Math.hypot(p[0]-GOAL_PLANE_POINTS[i][0],p[1]-GOAL_PLANE_POINTS[i][1]));
        console.log(JSON.stringify({maxError:Math.max(...errors)}));
        """,
        root,
    )
    assert float(payload["maxError"]) < 0.05


def test_mp4_fps_is_read_from_metadata() -> None:
    root = Path(__file__).parents[1]
    payload = run_node(
        """
        import fs from 'node:fs';
        import {parseMp4Fps} from './annotation_tool/mp4fps.js';
        const buffer=fs.readFileSync('./tests/data/fixture_30fps.mp4');
        const arrayBuffer=buffer.buffer.slice(buffer.byteOffset,buffer.byteOffset+buffer.byteLength);
        console.log(JSON.stringify(parseMp4Fps(arrayBuffer)));
        """,
        root,
    )
    assert abs(float(payload["fps"]) - 30.0) < 1e-9
    assert payload["variableFrameRate"] is False
    assert "stts" in str(payload["timingSource"])
