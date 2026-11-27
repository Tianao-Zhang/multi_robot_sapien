#!/usr/bin/env python3
"""Build and audit a synchronized Stage 2 paired manifest from Stage 1 episodes."""

from __future__ import annotations

import argparse
import json
from os.path import relpath
from pathlib import Path
from typing import Any, Dict, List


def build_stage2_manifest(stage1_root: Path, output_root: Path, task: str) -> Dict[str, Any]:
    def stage2_path(value: str) -> str:
        path = Path(value)
        return str(path) if path.is_absolute() else relpath(stage1_root / path, output_root)

    s1 = json.loads((stage1_root / "manifest.json").read_text(encoding="utf-8"))
    by_joint: Dict[str, List[Dict[str, Any]]] = {}
    for ep in s1["episodes"]:
        by_joint.setdefault(str(ep["joint_episode_id"]), []).append(ep)

    paired: List[Dict[str, Any]] = []
    for joint_id in sorted(by_joint):
        agents = sorted(by_joint[joint_id], key=lambda r: int(r["agent_id"]))
        if len(agents) < 2:
            raise ValueError(f"Joint episode {joint_id} has only {len(agents)} agent record(s)")
        frames = int(agents[0]["frames"])
        latents = int(agents[0]["latent_frames"])
        for a in agents[1:]:
            if int(a["frames"]) != frames or int(a["latent_frames"]) != latents:
                raise ValueError(f"Misaligned frame count across agents in {joint_id}")
        paired.append(
            {
                "joint_episode_id": joint_id,
                "frames": frames,
                "latent_frames": latents,
                "split": "train",
                "agents": [
                    {
                        **agent,
                        "actions_path": stage2_path(agent["actions_path"]),
                        "latent_paths": {key: stage2_path(value) for key, value in agent["latent_paths"].items()},
                    }
                    for agent in agents
                ],
            }
        )

    manifest = {
        "schema": "mawam.stage2.paired_manifest.v1",
        "task": task,
        "num_agents": len(paired[0]["agents"]) if paired else 2,
        "joint_episodes": len(paired),
        "episodes": paired,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage1-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--task", required=True)
    args = p.parse_args()
    m = build_stage2_manifest(args.stage1_root.resolve(), args.output_root.resolve(), args.task)
    print(f"Built Stage 2 manifest with {m['joint_episodes']} synchronized pairs at {args.output_root / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
