from __future__ import annotations

import argparse
import json
from pathlib import Path

from .model import CoverageSurfaceModel
from .schema import read_jsonl
from .simulate import simulate_penalties, true_upper_outer_contrast, write_simulation
from .visualize import export_report


def command_run(args: argparse.Namespace) -> int:
    records = read_jsonl(args.input)
    fit = CoverageSurfaceModel().fit(records, keeper_id=args.keeper_id)
    paths = export_report(fit, records, args.output)
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
    if fit.banner:
        print(f"\n{fit.banner}")
    return 0


def command_simulate(args: argparse.Namespace) -> int:
    records = simulate_penalties(args.n, args.seed, args.keeper_id, args.concentrated_center)
    write_simulation(args.output, records)
    print(f"wrote {len(records)} records to {args.output}")
    return 0


def command_validate(args: argparse.Namespace) -> int:
    truth = true_upper_outer_contrast()
    covered = 0
    sign_correct = 0
    rows = []
    for seed in range(args.runs):
        fit = CoverageSurfaceModel().fit(simulate_penalties(args.n, seed=seed))
        summary = fit.asymmetry_summary()
        estimate = float(summary["time_left_minus_right_s"])
        lower = float(summary["time_ci_lower_s"])
        upper = float(summary["time_ci_upper_s"])
        in_interval = lower <= truth <= upper
        covered += int(in_interval)
        sign_correct += int(estimate * truth > 0)
        rows.append({"seed": seed, "estimate_s": estimate, "lower_s": lower, "upper_s": upper, "covered": in_interval})
    result = {
        "runs": args.runs,
        "n_per_run": args.n,
        "truth_s": truth,
        "credible_interval_coverage": covered / args.runs,
        "sign_recovery": sign_correct / args.runs,
        "pass": covered / args.runs >= 0.90 and sign_correct == args.runs,
        "runs_detail": rows,
    }
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gkcoverage")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="fit surfaces and export PNG/HTML report")
    run.add_argument("--input", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--keeper-id")
    run.set_defaults(func=command_run)

    simulate = sub.add_parser("simulate", help="write synthetic JSONL annotations")
    simulate.add_argument("--output", required=True, type=Path)
    simulate.add_argument("--n", type=int, default=200)
    simulate.add_argument("--seed", type=int, default=1)
    simulate.add_argument("--keeper-id", default="sim_keeper")
    simulate.add_argument("--concentrated-center", action="store_true")
    simulate.set_defaults(func=command_simulate)

    validate = sub.add_parser("validate", help="run simulation-based asymmetry validation")
    validate.add_argument("--runs", type=int, default=20)
    validate.add_argument("--n", type=int, default=200)
    validate.set_defaults(func=command_validate)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
