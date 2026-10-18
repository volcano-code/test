#!/usr/bin/env python3
"""Create a class-balanced, near-threshold subset of newly FEM-labeled rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TAU = 1.3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--unsafe-floor", type=float, default=0.5)
    parser.add_argument("--safe-ceiling", type=float, default=2.0)
    args = parser.parse_args()
    payload = json.loads(args.src.read_text())
    rows = payload["rows"]
    safe = [r for r in rows if r["fem_safe"] and TAU <= float(r["sf"]) < args.safe_ceiling * TAU]
    unsafe = [r for r in rows if (not r["fem_safe"]) and args.unsafe_floor * TAU <= float(r["sf"]) < TAU]
    n = min(len(safe), len(unsafe))
    # Closest-to-threshold rows expose the decision boundary most directly.
    safe.sort(key=lambda r: abs(float(r["sf"]) - TAU))
    unsafe.sort(key=lambda r: abs(float(r["sf"]) - TAU))
    selected = safe[:n] + unsafe[:n]
    out = {
        "schema": "balanced_nearboundary_current_candidate_fem_v1",
        "source": str(args.src),
        "tau": TAU,
        "n_rows": len(selected),
        "safe": n,
        "unsafe": n,
        "selection": "equal safe/unsafe rows nearest the FEM safety threshold",
        "rows": selected,
    }
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("n_rows", "safe", "unsafe", "selection")}, indent=2))


if __name__ == "__main__":
    main()
