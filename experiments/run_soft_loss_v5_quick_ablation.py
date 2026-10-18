"""CLI for the frozen minimal soft-loss v5 CUDA ablation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadowforge_pipeline.soft_voxel_v5.quick_ablation import (
    PROFILE_ORDER,
    TASK_ORDER,
    analyze_ablation,
    run_ablation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-v4-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--profiles", nargs="+", choices=PROFILE_ORDER, default=list(PROFILE_ORDER))
    parser.add_argument("--tasks", nargs="+", choices=TASK_ORDER, default=list(TASK_ORDER))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.smoke and args.analyze:
        raise ValueError("--smoke and --analyze are mutually exclusive")
    if args.smoke:
        args.steps = 200
        args.tasks = [args.tasks[0]]
    if args.analyze:
        report = analyze_ablation(args.output_root, args.source_v4_root)
        print(report["label"])
        return 0
    run_ablation(
        args.output_root,
        args.source_v4_root,
        device=args.device,
        steps=args.steps,
        profiles=args.profiles,
        tasks=args.tasks,
        smoke=args.smoke,
        resume=args.resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
