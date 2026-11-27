"""Collect expert demonstrations from the live local MA-Bench simulator."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess

import h5py
from imageio_ffmpeg import get_ffmpeg_exe
import numpy as np

from ma_bench.envs.task_builder import build_task_env
from ma_bench.registry import get_task_info


AGENTS = ("A", "B")
CAMERAS = {"head": "head_camera", "left_wrist": "left_camera", "right_wrist": "right_camera"}


class VideoWriter:
    def __init__(self, path: Path, shape: tuple[int, ...], fps: float) -> None:
        self.path = path
        self.shape = shape
        height, width, _ = shape
        self.process = subprocess.Popen(
            [get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin",
             "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
             "-framerate", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
             "-threads", "1", "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", "-n", str(path)],
            stdin=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != self.shape or frame.dtype != np.uint8:
            raise ValueError(f"Unexpected RGB frame for {self.path}: {frame.shape} {frame.dtype}")
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        try:
            self.process.stdin.close()
        finally:
            status = self.process.wait()
        if status:
            raise RuntimeError(f"FFmpeg exited {status} writing {self.path}")


class EpisodeRecorder:
    """Capture video and native trajectories from the same post-control samples."""

    def __init__(self, env, root: Path, seed: int) -> None:
        self.env = env
        self.root = root
        self.seed = seed
        self.rows = {agent: defaultdict(list) for agent in AGENTS}
        self.video_paths = {"overview": root / "overview.mp4"}
        self.encoders = {}
        self.ticks = []
        self.times = []
        for agent in AGENTS:
            video_dir = self.agent_dir(agent) / "videos"
            video_dir.mkdir(parents=True)
            self.video_paths.update({f"{agent}/{view}": video_dir / f"{view}.mp4" for view in CAMERAS})

    def agent_dir(self, agent: str) -> Path:
        return self.root / f"agent_{agent.lower()}"

    def record(self, frame) -> None:
        if frame.step != len(self.ticks):
            raise RuntimeError(f"Missing or duplicate sample at control tick {frame.step}")
        images = self.env.render(self.video_paths)
        for key, rgb in images.items():
            if key not in self.encoders:
                self.encoders[key] = VideoWriter(self.video_paths[key], rgb.shape, self.env.task_info.control_hz)
            self.encoders[key].write(rgb)

        self.ticks.append(frame.step)
        self.times.append(self.env.time_s)
        self._record_state()

    def _record_state(self) -> None:
        world = self.env.world
        canonical = self.env.canonical_state()
        for index, agent in enumerate(AGENTS):
            rows = self.rows[agent]
            left, right = (world.arms[f"{agent}.{side}"] for side in ("left", "right"))
            measured = world.robots[agent].get_qpos()
            arm_order = np.column_stack((left.arm_indices, right.arm_indices)).ravel()
            joint_order = np.concatenate((arm_order, left.finger_indices, right.finger_indices))
            rows["native/measured_qpos"].append(measured[joint_order].astype(np.float64))
            rows["action"].append(np.r_[left.target_q, left.gripper_open, right.target_q, right.gripper_open])
            for side, start, arm in (("left", 0, left), ("right", 8, right)):
                pose = canonical[index, start:start + 7].copy()
                path = f"native/endpose/{side}_endpose"
                if rows[path] and pose[3:] @ rows[path][-1][3:] < 0:
                    pose[3:] *= -1
                rows[path].append(pose)
                rows[f"native/endpose/{side}_gripper"].append(canonical[index, start + 7])
                rows[f"native/joint_action/{side}_arm"].append(arm.target_q.copy())
                rows[f"native/joint_action/{side}_gripper"].append(arm.gripper_open)

            to_world = self.env.agent_to_world(agent)
            from_world = np.linalg.inv(to_world)
            for view, name in CAMERAS.items():
                camera = world.cameras[f"{agent}/{view}"]
                prefix = f"native/observation/{name}"
                rows[f"{prefix}/cam2world_gl"].append((from_world @ camera.get_model_matrix()).astype(np.float32))
                rows[f"{prefix}/extrinsic_cv"].append((camera.get_extrinsic_matrix() @ to_world).astype(np.float32))
                rows[f"{prefix}/intrinsic_cv"].append(camera.get_intrinsic_matrix())

    def close(self) -> None:
        errors = []
        for encoder in self.encoders.values():
            try:
                encoder.close()
            except (OSError, RuntimeError) as exc:
                errors.append(str(exc))
        if self.ticks:
            self._write_trajectories()
        if errors:
            raise RuntimeError("; ".join(errors))

    def _write_trajectories(self) -> None:
        ticks = np.asarray(self.ticks, dtype=np.int64)
        for agent, rows in self.rows.items():
            commands = np.asarray(rows["action"], dtype=np.float64)
            valid = ticks < ticks[-1]
            with h5py.File(self.agent_dir(agent) / "trajectory.h5", "x") as fh:
                fh.attrs.update({
                    "scene_seed": self.seed,
                    "control_hz": int(self.env.task_info.control_hz),
                    "physics_hz": int(self.env.task_info.physics_hz),
                    "endpose_frame": "agent tabletop frame, quaternion wxyz",
                    "endpose_gripper": "measured opening fraction",
                    "source": "local scripted expert in SAPIEN PhysX",
                })
                fh["tick"] = ticks
                fh["native/source_frame"] = ticks
                fh["native/sim_time_s"] = np.asarray(self.times, dtype=np.float64)
                fh["native/to_world"] = self.env.agent_to_world(agent)
                fh["native/pointcloud"] = np.empty((len(ticks), 0), dtype=np.float64)
                fh["native/next_action"] = np.concatenate((commands[1:], commands[-1:]))
                fh["native/next_action_valid"] = valid
                for view in CAMERAS:
                    fh[f"video_tick/{view}"] = ticks
                for name, samples in rows.items():
                    if len(samples) != len(ticks):
                        raise RuntimeError(f"Incomplete {name} for agent {agent}")
                    fh.create_dataset(name, data=np.asarray(samples), compression="gzip", shuffle=True)


def write_json(path: Path, report: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def collect_episode(root: Path, episode: int, seed: int, task: str) -> dict:
    staging = root / "in_progress" / f"episode{episode}"
    staging.mkdir(parents=True)
    report = {"seed": seed, "accepted": False}
    env = recorder = None
    errors = []
    try:
        env = build_task_env(task, seed=seed, backend="sapien")
        env.reset(seed)
        recorder = EpisodeRecorder(env, staging, seed)
        env.on_control = recorder.record
        env.run_expert()
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        if recorder is not None:
            env.on_control = None
            try:
                recorder.close()
                report.update(env.acceptance())
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
            report["recorded_ticks"] = len(recorder.ticks)
            report["expected_ticks"] = env.max_steps
            report["control_hz"] = env.task_info.control_hz
            report["physics_hz"] = env.task_info.physics_hz
            report["task_spec_sha256"] = env.spec.sha256()
            report["initial_positions_m"] = {name: body.pos for name, body in env.world.initial_bodies.items()}
            report["block_colors"] = env.layout.block_colors
        if env is not None:
            env.close()

    report["errors"] = errors
    report["accepted"] = bool(report["accepted"] and not errors and report["recorded_ticks"] == report["expected_ticks"])
    category = "tasks" if report["accepted"] else "rejected"
    relative_dir = Path(category) / task / f"episode{episode}"
    report["directory"] = relative_dir.as_posix()
    write_json(staging / "episode_report.json", report)
    final = root / relative_dir
    final.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(final)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="find_common_color")
    parser.add_argument("--output", type=Path, required=True, help="New dataset directory")
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    args = parser.parse_args()
    task = get_task_info(args.task).slug
    if task != "find_common_color":
        parser.error("The local expert collector supports find_common_color (alias: color)")
    if args.num_episodes < 1 or args.seed_start < 0:
        parser.error("--num-episodes must be positive and --seed-start must be nonnegative")

    args.output.mkdir(parents=True, exist_ok=False)
    batch = {"task": task, "requested": args.num_episodes, "accepted": 0, "episodes": []}
    for episode in range(args.num_episodes):
        result = collect_episode(args.output, episode, args.seed_start + episode, task)
        batch["episodes"].append({key: result[key] for key in ("seed", "accepted", "directory", "errors")})
        batch["accepted"] += int(result["accepted"])
        write_json(args.output / "collection_report.json", batch)
        status = "accepted" if result["accepted"] else "rejected"
        print(f"seed={result['seed']} {status} ticks={result.get('recorded_ticks', 0)} {result['directory']}", flush=True)
    (args.output / "in_progress").rmdir()
    return int(batch["accepted"] != args.num_episodes)


if __name__ == "__main__":
    raise SystemExit(main())
