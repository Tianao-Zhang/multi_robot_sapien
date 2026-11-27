"""Prepare synchronized local color demonstrations for both MAWAM training stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from shutil import copytree

import h5py
import imageio.v2 as imageio
import numpy as np
import torch
from torch.nn.functional import interpolate

from ma_bench.action_codec import ACTION_PER_FRAME
from ma_bench.registry import get_task_info
from mawam.data.prepare_pairs import build_stage2_manifest
from mawam.data.prepare_text_cache import encode_prompts
from mawam.data.stage1_dataset import CAMERA_KEYS
from mawam.upstream import add_lingbot_source


VIEWS = (("head", (256, 320)), ("left_wrist", (128, 160)), ("right_wrist", (128, 160)))
CAMERA_TICK_STRIDE = 4


def read_actions(path: Path) -> np.ndarray:
    with h5py.File(path) as record:
        ticks = record["tick"][:]
        if record.attrs["control_hz"] != 25 or not np.array_equal(ticks, np.arange(len(ticks))):
            raise ValueError(f"Expected contiguous 25 Hz ticks in {path}")
        for view, _ in VIEWS:
            if not np.array_equal(record[f"video_tick/{view}"][:], ticks):
                raise ValueError(f"Camera {view} is out of sync in {path}")
        poses = record["native/endpose"]
        actions = np.column_stack((
            poses["left_endpose"][:], poses["left_gripper"][:],
            poses["right_endpose"][:], poses["right_gripper"][:],
        )).astype(np.float32)
    if actions.shape != (len(ticks), 16) or not np.isfinite(actions).all():
        raise ValueError(f"Invalid bimanual end-effector poses in {path}")
    return actions


def encode_camera(vae, path: Path, size: tuple[int, int], ticks: int, last_tick: int) -> torch.Tensor:
    frames = []
    with imageio.get_reader(path, "ffmpeg") as video:
        if not np.isclose(video.get_meta_data()["fps"], 25):
            raise ValueError(f"Expected a 25 Hz video at {path}")
        count = 0
        for tick, rgb in enumerate(video):
            count += 1
            if tick <= last_tick and tick % CAMERA_TICK_STRIDE == 0:
                frames.append(rgb)
        if count != ticks:
            raise ValueError(f"Found {count} video frames for {ticks} ticks at {path}")

    pixels = torch.as_tensor(np.stack(frames), device=vae.device).permute(0, 3, 1, 2).float()
    pixels = interpolate(pixels, size=size, mode="bilinear", align_corners=False)
    pixels = (pixels / 255 * 2 - 1).permute(1, 0, 2, 3).unsqueeze(0).to(vae.dtype)
    with torch.inference_mode():
        latent = vae.encode(pixels).latent_dist.mode()[0].float()
        mean = torch.tensor(vae.config.latents_mean, device=latent.device)[:, None, None, None]
        std = torch.tensor(vae.config.latents_std, device=latent.device)[:, None, None, None]
        latent = ((latent - mean) / std).permute(1, 2, 3, 0).to(torch.bfloat16).cpu().contiguous()
    expected = (last_tick // ACTION_PER_FRAME + 1, size[0] // 16, size[1] // 16, 48)
    if tuple(latent.shape) != expected or not torch.isfinite(latent).all():
        raise ValueError(f"Invalid VAE latents at {path}: {tuple(latent.shape)}, expected {expected}")
    return latent


def prepare(input_root: Path, output: Path, base_model: Path, device: str) -> None:
    from modules.utils import load_vae

    batch = json.loads((input_root / "collection_report.json").read_text(encoding="utf-8"))
    task = get_task_info(batch["task"])
    accepted = [episode for episode in batch["episodes"] if episode["accepted"]]
    if task.slug != "find_common_color" or not accepted:
        raise ValueError("Provide a local find_common_color collection with accepted episodes")
    for folder in ("vae", "text_encoder", "tokenizer"):
        if not (base_model / folder).is_dir():
            raise FileNotFoundError(f"Missing LingBot-VA checkpoint component: {base_model / folder}")

    output.mkdir(parents=True, exist_ok=False)
    stage1 = output / "stage1"
    stage2 = output / "stage2"
    dtype = torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
    vae = load_vae(str(base_model / "vae"), dtype, device).eval().requires_grad_(False)
    episodes = []
    for item in accepted:
        source = (input_root / item["directory"]).resolve()
        if not source.is_relative_to(input_root / "tasks" / task.slug):
            raise ValueError(f"Invalid accepted episode path {source}")
        joint_id = source.name
        teams = [(robot, source / f"agent_{robot.role_key.lower()}") for robot in task.robots]
        actions = [read_actions(directory / "trajectory.h5") for _, directory in teams]
        ticks = len(actions[0])
        if ticks <= ACTION_PER_FRAME or any(len(seq) != ticks for seq in actions):
            raise ValueError(f"Incompatible action clocks in {source}")
        last_tick = ((ticks - 1) // ACTION_PER_FRAME) * ACTION_PER_FRAME

        for (robot, directory), seq in zip(teams, actions):
            relative = Path("episodes") / joint_id / directory.name
            (stage1 / relative).mkdir(parents=True, exist_ok=False)
            actions_path = relative / "actions.pt"
            torch.save({"raw_action16_wxyz": torch.from_numpy(seq[: last_tick + 1].copy())}, stage1 / actions_path)
            latent_paths = {}
            for key, (view, size) in zip(CAMERA_KEYS, VIEWS):
                latent_path = relative / f"{view}.pt"
                latent = encode_camera(vae, directory / "videos" / f"{view}.mp4", size, ticks, last_tick)
                torch.save({"latent": latent}, stage1 / latent_path)
                latent_paths[key] = latent_path.as_posix()
            episodes.append({
                "joint_episode_id": joint_id, "agent_id": robot.agent_id, "embodiment": robot.embodiment,
                "frames": last_tick // CAMERA_TICK_STRIDE + 1,
                "latent_frames": last_tick // ACTION_PER_FRAME + 1, "source_seed": item["seed"],
                "actions_path": actions_path.as_posix(), "latent_paths": latent_paths,
            })
        print(f"Prepared {joint_id} ({last_tick + 1} of {ticks} ticks)", flush=True)

    del vae
    encode_prompts(base_model, stage1, task.global_prompt, list(task.local_prompts), device)
    manifest = {"schema": "mawam.stage1.manifest.v1", "task": task.slug, "episodes": episodes}
    (stage1 / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    copytree(stage1 / "text", stage2 / "text")
    build_stage2_manifest(stage1, stage2, task.slug)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path, help="Raw root from ma_bench.collect_demos")
    parser.add_argument("--output", required=True, type=Path, help="New directory for stage1 and stage2")
    parser.add_argument("--base-model", default=Path("checkpoints/lingbot-va-base"), type=Path)
    parser.add_argument("--upstream-root", default=Path("third_party/LingBot-VA"), type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    add_lingbot_source(args.upstream_root)
    prepare(args.input_root.resolve(), args.output.resolve(), args.base_model.resolve(), args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
