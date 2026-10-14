"""Import paths for the ReTerm training package."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent


def ensure_local_paths() -> None:
    """Expose the bundled training libraries and task interfaces."""

    paths = [
        REPO_ROOT / "src",
        ROOT,
        ROOT / "verl",
    ]
    for path in reversed(paths):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
