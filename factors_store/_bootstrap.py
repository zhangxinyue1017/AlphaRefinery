from __future__ import annotations

import sys
from pathlib import Path


def ensure_project_roots() -> None:
    roots = [
        Path("/root"),
        Path(__file__).resolve().parents[2],
    ]
    for root in roots:
        text = str(root)
        if text not in sys.path:
            sys.path.insert(0, text)

