"""Use the same Python 3.10 compatibility shim as the RL launcher in tests."""

from pathlib import Path
import runpy
import sys


if sys.version_info < (3, 12):
    runpy.run_path(
        str(Path(__file__).parent / "rl" / "compat" / "py310_megatron_bridge" / "sitecustomize.py")
    )
