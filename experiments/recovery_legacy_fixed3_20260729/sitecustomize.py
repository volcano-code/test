"""Pin the isolated Sadekar renderer proxy before legacy code rewrites sys.path."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_PROXY = Path(__file__).resolve().with_name("sadekar_protocol.py")
_SPEC = importlib.util.spec_from_file_location("sadekar_protocol", _PROXY)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Cannot load isolated Sadekar renderer proxy: {_PROXY}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules["sadekar_protocol"] = _MODULE
_SPEC.loader.exec_module(_MODULE)
