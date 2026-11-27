"""Task factory wiring RoboTwin 2.0 / SAPIEN scenes with MA-Bench evaluators."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from ma_bench.envs.base_env import DryRunMultiRobotEnv, MultiRobotEnv
from ma_bench.evaluators import make_evaluator
from ma_bench.registry import get_task_info


def build_task_env(
    task: str,
    seed: int = 0,
    backend: str = "sapien",
    robotwin_root: Optional[Path] = None,
    assets_root: Optional[Path] = None,
) -> MultiRobotEnv:
    """Instantiate a MA-Bench task environment paired with its deterministic evaluator.

    Args:
        task: Task slug or alias (`bowl`, `color`, `scan`, `door`, `tray`, `food`, `balls`).
        seed: Scene randomization seed.
        backend: `"sapien"` (RoboTwin 2.0 + SAPIEN live physics) or `"dry_run"` (CPU codec/client check).
        robotwin_root: Path to RoboTwin 2.0 checkout (defaults to `$ROBOTWIN_ROOT`).
        assets_root: Path to MA-Bench / RoboTwin 3D assets (defaults to `$MA_BENCH_ASSETS`).
    """
    info = get_task_info(task)
    evaluator = make_evaluator(info.slug, seed=seed)

    if backend == "dry_run":
        return DryRunMultiRobotEnv(task_info=info, evaluator=evaluator)

    if backend != "sapien":
        raise ValueError(f"Unsupported backend {backend!r}; expected 'sapien' or 'dry_run'")

    rt_root = Path(robotwin_root or os.environ.get("ROBOTWIN_ROOT", "./third_party/RoboTwin"))
    if not rt_root.is_dir():
        raise FileNotFoundError(
            f"RoboTwin 2.0 runtime not found at {rt_root}. "
            "Clone RoboTwin 2.0 (https://github.com/RoboTwin-Platform/RoboTwin), install "
            "`requirements-sim.txt`, download assets from `EmbodiedCity/MA-Bench`, and set `ROBOTWIN_ROOT`."
        )

    try:
        import sapien  # noqa: F401
        import mplib  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "SAPIEN and mplib are required for live MA-Bench simulation. "
            "Install `requirements-sim.txt` or pass `--backend dry_run` for offline verification."
        ) from exc

    if info.slug == "find_common_color":
        from ma_bench.envs.color_env import FindCommonColorEnv

        return FindCommonColorEnv(info, evaluator, robotwin_root=rt_root, assets_root=assets_root)

    from ma_bench.envs.live_env import LiveSceneEnv

    return LiveSceneEnv(info, evaluator, robotwin_root=rt_root, assets_root=assets_root)
