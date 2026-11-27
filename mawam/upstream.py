"""Locate the standalone LingBot-VA source tree."""

from pathlib import Path
import sys


def add_lingbot_source(path: Path) -> None:
    root = path.resolve()
    if (root / "wan_va" / "modules" / "model.py").is_file():
        root /= "wan_va"
    if not (root / "modules" / "model.py").is_file():
        raise FileNotFoundError(f"LingBot-VA source missing at {path}; run scripts/setup_train.sh")
    sys.path.insert(0, str(root))
