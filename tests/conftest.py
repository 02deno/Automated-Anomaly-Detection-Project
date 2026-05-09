"""Pytest config: expose ``api/`` and ``scripts/`` on ``sys.path`` for imports."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("api", "scripts"):
    candidate = ROOT / sub
    if candidate.is_dir():
        path_str = str(candidate)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
