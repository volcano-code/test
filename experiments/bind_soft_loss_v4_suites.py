"""Thin executable for immutable v4 suite binding."""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.soft_loss_v4_registry import binding_main


if __name__ == "__main__":
    raise SystemExit(binding_main())
