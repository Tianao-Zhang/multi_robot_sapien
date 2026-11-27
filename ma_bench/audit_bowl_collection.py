"""Verify native bowl demonstrations and freeze the training and holdout seed split."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

from ma_bench.collect_demos import CAMERAS, write_json


ARMS = ("A.left", "A.right", "B.right", "B.left")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    result = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def verify_video(path: Path, frames: int, rate: float, size: tuple[int, int]) -> None:
    capture = cv2.VideoCapture(str(path))
    try:
        require(capture.isOpened(), f"Cannot open {path}")
        require(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == frames, f"Wrong frame count in {path}")
        require(np.isclose(capture.get(cv2.CAP_PROP_FPS), rate), f"Wrong frame rate in {path}")
        for index in (0, frames // 2, frames - 1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, rgb = capture.read()
            require(ok and rgb.shape[:2] == size and np.ptp(rgb) > 0, f"Invalid decoded frame {index} in {path}")
    finally:
        capture.release()


def audit_episode(root: Path, entry: dict) -> dict:
    path = root / entry["directory"]
    info = json.loads((path / "episode_report.json").read_text())
    seed = entry["seed"]
    require(info["seed"] == seed and info["native_exit_code"] == 0 and not info["errors"], f"Native run failed for seed {seed}")
    require(info["accepted"] and all(info["metrics"]["milestones"].values()) and len(info["metrics"]["milestones"]) == 6,
            f"Missing milestones for seed {seed}")
    lifts = [info["four_arm_lift_ticks"][arm] for arm in ARMS]
    require(info["four_arm_relay"] and lifts == sorted(set(lifts)), f"Unverified grasp order for seed {seed}")
    ticks = np.arange(info["recorded_ticks"])
    rate = float(info["video_hz"])
    require(rate in (6.25, 25.0) and 0 < len(ticks) == info["expected_ticks"] <= 2250, f"Invalid clock for seed {seed}")
    video_ticks = ticks[::int(25 / rate)]
    overview = path / "overview.mp4"
    verify_video(overview, len(video_ticks), rate, (540, 960))
    verified = {"overview.mp4": digest(overview)}
    for key in ("agent_a", "agent_b"):
        folder = path / key
        trajectory = folder / "trajectory.h5"
        with h5py.File(trajectory) as data:
            require(data.attrs["scene_seed"] == seed and data.attrs["control_hz"] == 25 and data.attrs["physics_hz"] == 250,
                    f"Mislabelled trajectory {trajectory}")
            require(np.array_equal(data["tick"][:], ticks), f"Missing control ticks in {trajectory}")
            actions = data["native/command_endpose_wxyz"][:]
            require(actions.shape == (len(ticks), 16) and np.isfinite(actions).all(), f"Invalid commands in {trajectory}")
            for start in (3, 11):
                require(np.allclose(np.linalg.norm(actions[:, start:start+4], axis=1), 1., atol=1e-4),
                        f"Invalid orientations in {trajectory}")
            for view, native_view in CAMERAS.items():
                video = folder / "videos" / f"{view}.mp4"
                require(np.array_equal(data[f"video_tick/{view}"][:], video_ticks), f"Misaligned {view} in {trajectory}")
                require(data[f"native/observation/{native_view}/extrinsic_cv"].shape == (len(video_ticks), 3, 4),
                        f"Missing camera extrinsics in {trajectory}")
                verify_video(video, len(video_ticks), rate, (240, 320))
                verified[str(video.relative_to(path))] = digest(video)
        verified[str(trajectory.relative_to(path))] = digest(trajectory)
    return {"seed": seed, "ticks": len(ticks), "video_frames": len(video_ticks), "sha256": verified,
            "four_arm_lift_ticks": dict(zip(ARMS, lifts))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected", type=int, default=200)
    parser.add_argument("--train", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    root = args.collection.resolve()
    manifest = root / "collection_report.json"
    batch = json.loads(manifest.read_text())
    accepted = [row for row in batch["episodes"] if row["accepted"]]
    seeds = [row["seed"] for row in accepted]
    require(batch["task"] == "joint_bowl_placing" and batch["requested"] == batch["accepted"] == args.expected,
            "The requested collection has not finished")
    require(len(seeds) == len(set(seeds)) == args.expected and 0 < args.train < args.expected, "Invalid seed split")
    cv2.setNumThreads(1)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        verified = list(pool.map(lambda row: audit_episode(root, row), accepted))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, {
        "collection_manifest_sha256": digest(manifest), "accepted": len(seeds),
        "attempted": len(batch["episodes"]), "trajectories": 2 * len(seeds), "videos": 7 * len(seeds),
        "train_seeds": seeds[:args.train], "holdout_seeds": seeds[args.train:], "episodes": verified,
    })
    print(f"Audited {len(seeds)} episodes and {7*len(seeds)} videos", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
