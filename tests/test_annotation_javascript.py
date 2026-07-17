from __future__ import annotations

import json
import subprocess
from pathlib import Path


def run_node(script: str, cwd: Path) -> dict[str, object]:
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


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
