"""Run the live expert for each MA-Bench scene and save native simulation videos."""

from __future__ import annotations

import argparse
import json
from multiprocessing import get_context
from pathlib import Path
import traceback

from ma_bench.collect_demos import VideoWriter, write_json
from ma_bench.envs.task_builder import build_task_env
from ma_bench.registry import MABENCH_TASKS, get_task_info


def run_scene(task, seed, directory, *, record=True):
    directory.mkdir()
    env = build_task_env(task, seed=seed, backend="sapien")
    writer = None
    overview = directory / "overview.mp4"
    error = None
    details = {}
    recorded_frames = 0

    def capture(_frame):
        nonlocal writer, recorded_frames
        rgb = env.render(("overview",))["overview"]
        if writer is None:
            writer = VideoWriter(overview, rgb.shape, env.task_info.control_hz)
        writer.write(rgb)
        recorded_frames += 1

    try:
        if task == "find_common_color":
            env.reset(seed)
        else:
            env.initialize(seed)
        if record:
            env.on_control = capture
        env.run_expert()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (directory / "runtime_error.log").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        env.on_control = None
        if writer is not None:
            try:
                writer.close()
            except (OSError, RuntimeError) as exc:
                error = f"{error}; {exc}" if error else str(exc)
        if env.world is not None and env.step_count:
            try:
                details = env.acceptance() if task == "find_common_color" else env.acceptance(error=error)
            except Exception as exc:
                error = f"{error}; {exc}" if error else str(exc)
        control_steps = env.step_count
        env.close()

    video_exists = overview.is_file() and overview.stat().st_size > 0 and recorded_frames > 0
    result = {
        **details, "task": task, "seed": seed, "control_steps": control_steps,
        "control_hz": env.task_info.control_hz, "physics_hz": env.task_info.physics_hz,
        "video": overview.name if video_exists else None, "recorded_video_frames": recorded_frames,
        "runtime_error": error,
        "accepted": bool(details.get("accepted") and error is None and (video_exists or not record)),
    }
    write_json(directory / "report.json", result)
    return result


def _scene_process(task, seed, directory, record):
    report = run_scene(task, seed, directory, record=record)
    raise SystemExit(0 if report["accepted"] else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", help="A task slug or alias. Default: all seven scenes.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--no-video", action="store_true", help="Run physics and evaluation only for diagnosis")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    tasks = tuple(dict.fromkeys(get_task_info(task).slug for task in args.task)) if args.task else tuple(MABENCH_TASKS)
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {"seed": args.seed, "requested_scenes": len(tasks), "passed_scenes": 0, "scenes": []}
    context = get_context("spawn")
    for task in tasks:
        directory = args.output / task
        job = context.Process(target=_scene_process, args=(task, args.seed, directory, not args.no_video))
        job.start()
        try:
            job.join()
        except KeyboardInterrupt:
            job.terminate()
            job.join()
            raise
        report_path = directory / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {
            "accepted": False, "video": None, "runtime_error": f"Native scene process exited {job.exitcode}",
        }
        flags = report.get("metrics", {}).get("milestones", {})
        summary["scenes"].append({
            "task": task, "accepted": report["accepted"], "milestones_passed": sum(flags.values()),
            "milestones_total": MABENCH_TASKS[task].total_milestones,
            "video": f"{task}/{report['video']}" if report["video"] else None,
            "report": f"{task}/report.json" if report_path.is_file() else None,
            "runtime_error": report["runtime_error"], "native_exit_code": job.exitcode,
        })
        summary["passed_scenes"] += int(report["accepted"])
        write_json(args.output / "summary.json", summary)
        print(json.dumps(summary["scenes"][-1], ensure_ascii=False), flush=True)
    return int(summary["passed_scenes"] != len(tasks))


if __name__ == "__main__":
    raise SystemExit(main())
