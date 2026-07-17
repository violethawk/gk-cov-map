from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from gkcoverage.schema import write_jsonl
from gkcoverage.simulate import simulate_penalties


def test_one_command_report_generation(tmp_path: Path) -> None:
    input_path = tmp_path / "penalties.jsonl"
    output_path = tmp_path / "report"
    write_jsonl(input_path, simulate_penalties(60, seed=8))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "gkcoverage.cli",
            "run",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert (output_path / "coverage_surface.png").exists()
    assert (output_path / "asymmetry_surface.png").exists()
    assert (output_path / "raw_overlay.png").exists()
    assert (output_path / "report.html").exists()
    assert (output_path / "surface_grid.csv").exists()
