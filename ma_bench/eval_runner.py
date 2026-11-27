#!/usr/bin/env python3
"""Closed-loop synchronous MA-Bench evaluation runner (100 independent trials/task).

Implements the environment side of Algorithm 2 (Paper Section 4.4 & Appendix D):
  - Connects via WebSocket to `mawam.inference.policy_server`
  - Pauses simulation physics during policy inference (synchronous benchmark protocol)
  - Executes 16 control steps on chunk 0 and 32 control steps (`C=2` latent frames)
    on every subsequent chunk at 25 Hz (250 Hz physics substeps)
  - Submits stride-4 RGB frames (6.25 Hz) and executed relative actions back to the
    policy server to ground the sliding joint KV cache (`W=64`)
  - Computes Role Score S_k, Collaboration Score S_collab, joint SR, and 95% Wilson CI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ma_bench.action_codec import (
    absolute_from_relative,
    pack_executed_feedback,
    relative_from_absolute,
    unpack_action_chunk,
)
from ma_bench.collect_demos import VideoWriter, write_json
from ma_bench.envs.task_builder import build_task_env
from ma_bench.metrics import TrialMetrics, aggregate_task_metrics
from ma_bench.registry import get_task_info


class WebsocketJointClient:
    """Minimal msgpack-numpy WebSocket client talking to `mawam.inference.policy_server`."""

    def __init__(self, host: str = "127.0.0.1", port: int = 25300) -> None:
        self.url = f"ws://{host}:{port}"
        try:
            import msgpack
            from websockets.sync.client import connect
        except ImportError as exc:
            raise ImportError("Closed-loop evaluation requires websockets and msgpack") from exc
        self._msgpack = msgpack
        self._ws = connect(self.url, max_size=None, compression=None)
        self.metadata = self._receive()

    @staticmethod
    def _pack_array(value: Any) -> Dict:
        if not isinstance(value, np.ndarray) or value.dtype.kind not in "buiuf":
            raise TypeError(f"Unexpected policy request object {type(value)}")
        return {b"__ndarray__": True, b"data": value.tobytes(), b"dtype": value.dtype.str, b"shape": value.shape}

    @staticmethod
    def _unpack_array(value: Dict) -> Any:
        if b"__ndarray__" in value:
            return np.frombuffer(value[b"data"], dtype=np.dtype(value[b"dtype"])).reshape(value[b"shape"])
        if b"__npgeneric__" in value:
            return np.dtype(value[b"dtype"]).type(value[b"data"])
        return value

    def _receive(self) -> Dict[str, Any]:
        raw = self._ws.recv(timeout=300)
        if isinstance(raw, str):
            raise RuntimeError(f"Policy server error:\n{raw}")
        result = self._msgpack.unpackb(raw, object_hook=self._unpack_array, raw=False)
        if not isinstance(result, dict):
            raise TypeError("Expected a dictionary from the policy server")
        return result

    def infer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        packed = self._msgpack.packb(payload, default=self._pack_array)
        self._ws.send(packed)
        return self._receive()

    def close(self) -> None:
        self._ws.close()


def run_single_trial(
    task: str,
    seed: int,
    client: WebsocketJointClient,
    backend: str = "sapien",
    video_path: Path | None = None,
    max_control_steps: int | None = None,
) -> TrialMetrics:
    info = get_task_info(task)
    env = build_task_env(info.slug, seed=seed, backend=backend)
    writer = None
    recorded = 0
    horizon = round(info.nominal_duration_s * info.control_hz)
    if max_control_steps is not None:
        horizon = min(horizon, max_control_steps)
    try:
        obs = env.reset(seed=seed)
        if video_path is not None:
            video_path.parent.mkdir(parents=True, exist_ok=True)
            frame = env.render(["overview"])["overview"]
            writer = VideoWriter(video_path, frame.shape, info.control_hz / 4)
            writer.write(frame)
            recorded = 1
        ref_pose = env.reference_pose_wxyz
        valid_masks = [np.array([True] * 8 + [r.arms == 2] * 8, dtype=bool) for r in info.robots]
        policy = client.infer({
            "reset": True, "task": info.slug, "global_prompt": info.global_prompt,
            "local_prompts": list(info.local_prompts), "agent_ids": [r.agent_id for r in info.robots],
        })
        chunk_idx, grounded = 0, 0
        generation_times = []
        cache = {}
        execution_error = None
        failed_target = None
        done = False
        while env.step_count < horizon and not done and execution_error is None:
            first_chunk = chunk_idx == 0
            resp = client.infer({"obs": [obs.images]} if first_chunk else {})
            rel_actions = unpack_action_chunk(resp["action"], first_chunk=first_chunk)
            if rel_actions.shape[0] != info.num_agents:
                raise ValueError("Policy returned the wrong number of agents")
            generation_times.append(float(resp.get("server_seconds", 0.0)))
            chunk_idx += 1
            executed_rel = np.zeros_like(rel_actions)
            captured_obs = []
            for tick in range(rel_actions.shape[1]):
                capture = (tick + 1) % 4 == 0
                try:
                    abs_cmd = np.stack([
                        absolute_from_relative(rel_actions[k, tick], ref_pose[k], valid_masks[k])
                        for k in range(info.num_agents)
                    ])
                    if hasattr(env, "advance_control"):
                        done = env.advance_control(abs_cmd)
                        actual = env.canonical_state()
                        if capture:
                            obs = env._observation()
                    else:
                        obs, done = env.step_control(abs_cmd)
                        actual = obs.canonical_state
                except (RuntimeError, ValueError) as exc:
                    execution_error = f"tick {env.step_count}: {type(exc).__name__}: {exc}"
                    failed_target = rel_actions[:, tick].tolist()
                    break
                for k in range(info.num_agents):
                    executed_rel[k, tick] = relative_from_absolute(actual[k], ref_pose[k], valid_masks[k])
                if capture:
                    captured_obs.append(obs.images)
                    if writer is not None:
                        writer.write(env.render(["overview"])["overview"])
                        recorded += 1
                if done or env.step_count >= horizon:
                    break
            if execution_error is None and len(captured_obs) == rel_actions.shape[1] // 4:
                feedback = client.infer({
                    "compute_kv_cache": True, "obs": captured_obs,
                    "state": pack_executed_feedback(executed_rel, first_chunk=first_chunk),
                })
                cache = feedback.get("cache", {})
                grounded += 1
            if execution_error is not None and writer is not None and env.step_count % 4:
                writer.write(env.render(["overview"])["overview"])
                recorded += 1
        metrics = env.evaluator.report(execution_valid=execution_error is None)
        metrics.details.update({
            "control_ticks": env.step_count, "max_control_ticks": horizon,
            "chunks_generated": chunk_idx, "chunks_grounded": grounded,
            "server_seconds_generate": generation_times, "last_cache": cache,
            "execution_error": execution_error, "failed_relative_target": failed_target, "preset": policy.get("preset"),
            "video": str(video_path) if video_path is not None else None,
            "video_frames": recorded,
        })
        if info.slug == "joint_bowl_placing" and hasattr(env, "world"):
            metrics.details["bowl_layout"] = env.world.bodies["bowl"]["attributes"]
        return metrics
    finally:
        try:
            if writer is not None:
                writer.close()
        finally:
            env.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, help="MA-Bench task slug or alias (e.g. door, bowl, tray)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=25300)
    p.add_argument("--num-trials", type=int, default=100, help="Independent trajectory trials (paper: 100)")
    p.add_argument("--seed-start", type=int, default=100000)
    p.add_argument("--seed-file", type=Path, help="JSON file with expert-verified evaluation seeds")
    p.add_argument("--backend", choices=("sapien", "dry_run"), default="sapien")
    p.add_argument("--output", type=Path, default=Path("runs/eval_results.json"))
    p.add_argument("--video-dir", type=Path)
    p.add_argument("--max-control-steps", type=int, help="Shorten the horizon for a diagnostic run")
    args = p.parse_args()

    info = get_task_info(args.task)
    if args.num_trials < 1 or (args.max_control_steps is not None and args.max_control_steps < 1):
        p.error("Trial count and any diagnostic horizon must be positive")
    if args.video_dir is not None and args.backend != "sapien":
        p.error("Video recording requires SAPIEN")
    seeds = list(range(args.seed_start, args.seed_start + args.num_trials))
    if args.seed_file:
        seeds = json.loads(args.seed_file.read_text())["seeds"]
        if len(seeds) < args.num_trials or len(set(seeds)) != len(seeds):
            p.error("The seed file must contain enough unique seeds")
        seeds = seeds[:args.num_trials]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        p.error(f"Choose a new output path instead of replacing {args.output}")
    client = WebsocketJointClient(host=args.host, port=args.port)
    trials: List[TrialMetrics] = []
    try:
        for i, seed in enumerate(seeds):
            video = args.video_dir / f"seed{seed}.mp4" if args.video_dir is not None else None
            tm = run_single_trial(info.slug, seed, client, args.backend, video, args.max_control_steps)
            trials.append(tm)
            summary = {
                "task": info.slug, "seeds": seeds, "requested_trials": len(seeds),
                "aggregate": aggregate_task_metrics(trials), "trials": [t.to_dict() for t in trials],
            }
            write_json(args.output, summary)
            print(
                f"[{i+1:03d}/{args.num_trials}] seed={seed} "
                f"S_role={tm.S_role*100:.1f}% S_collab={tm.S_collab*100:.1f}% SR={int(tm.joint_success)}",
                flush=True,
            )
    finally:
        client.close()

    print(json.dumps(summary["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
