#!/usr/bin/env python3
"""Unified closed-loop MAWAM inference server (Paper Section 4.4 & Algorithm 2).

Supports both paper inference presets without altering model weights:
  - `--preset benchmark`: 25 video steps (CFG 5.0), 50 action steps (CFG 1.0), W=64, C=2
  - `--preset realtime`:  4 video steps, 8 action steps, fused QKV + compiled RoPE +
                          cached text KVs + CUDA-graphed streaming VAE (0.72 s/chunk)

Every chunk runs Algorithm 2:
  (1) Imagine jointly: denoise all robots' future video latents together
  (2) Act: cache imagined latents and denoise all robots' actions, re-applying m_{e_k}
  (3) Execute: return native-slot actions to environment / robot controller
  (4) Ground: overwrite predicted tokens with encoded real observations & executed actions
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F

from ma_bench.action_codec import ActionCodec, EMBODIMENT_SLOTS, USED_ACTION_CHANNEL_IDS, INVERSE_USED_ACTION_CHANNEL_IDS
from ma_bench.registry import get_task_info
from mawam.config import InferenceConfig, load_inference_config
from mawam.upstream import add_lingbot_source


CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


class MawamPolicyServer:
    """Closed-loop multi-agent WAM policy implementing Algorithm 2."""

    def __init__(
        self,
        task_slug: str,
        checkpoint_dir: Path,
        base_model_dir: Path,
        text_cache_path: Path,
        infer_cfg: InferenceConfig,
        device: str = "cuda:0",
        seed: int = 0,
        action_norm: Path | None = None,
    ) -> None:
        from configs import VA_CONFIGS
        from modules.utils import WanVAEStreamingWrapper, load_vae
        from utils import FlowMatchScheduler, data_seq_to_patch, get_mesh_id

        from mawam.inference.fast_graph_engine import FastJointGraphEngine
        from mawam.inference.joint_kv_model import JointStage2InferenceModel
        from mawam.inference.streaming_vae import ThreeStateGraphedVAE

        self.task_info = get_task_info(task_slug)
        self.infer_cfg = infer_cfg
        self.device = torch.device(device)
        self.dtype = torch.bfloat16
        self.seed = int(seed)
        self.num_agents = self.task_info.num_agents
        self.chunk_frames = infer_cfg.chunk_frames
        self.action_per_frame = infer_cfg.action_per_frame
        self._data_seq_to_patch = data_seq_to_patch
        self._get_mesh_id = get_mesh_id

        self.va_cfg = copy.deepcopy(VA_CONFIGS["robotwin"])
        self.va_cfg.attn_window = infer_cfg.window_frames
        self.scheduler = FlowMatchScheduler(
            shift=infer_cfg.video_snr_shift, sigma_min=0.0, extra_one_step=True
        )
        self.action_scheduler = FlowMatchScheduler(
            shift=infer_cfg.action_snr_shift, sigma_min=0.0, extra_one_step=True
        )

        # Load frozen causal video VAE (streaming mode; CUDA-graphed in realtime preset)
        self.vae = load_vae(str(base_model_dir / "vae"), torch_dtype=self.dtype, torch_device=self.device)
        self.vae.eval().requires_grad_(False)
        self.latents_mean = torch.tensor(self.vae.config.latents_mean, device=self.device).reshape(1, -1, 1, 1, 1)
        self.latents_std = torch.tensor(self.vae.config.latents_std, device=self.device).reshape(1, -1, 1, 1, 1)

        if infer_cfg.streaming_vae_cuda_graphs:
            self.cam_streams = []
            for _ in range(self.num_agents):
                self.cam_streams.append(ThreeStateGraphedVAE(WanVAEStreamingWrapper(self.vae), self.device, 256, 320, self.dtype))
                self.cam_streams.append(ThreeStateGraphedVAE(WanVAEStreamingWrapper(self.vae), self.device, 128, 160, self.dtype))
                self.cam_streams.append(ThreeStateGraphedVAE(WanVAEStreamingWrapper(self.vae), self.device, 128, 160, self.dtype))
            self.head_vae = self.wrist_vae = None
        else:
            self.cam_streams = None
            self.head_vae = WanVAEStreamingWrapper(self.vae)
            self.wrist_vae = WanVAEStreamingWrapper(self.vae)

        # Load Stage2WanTransformer3DModel (verbatim HF checkpoint compatibility)
        transformer_dir = checkpoint_dir / "transformer" if (checkpoint_dir / "transformer").is_dir() else checkpoint_dir
        self.raw_model = JointStage2InferenceModel.from_stage2_pretrained(
            str(transformer_dir), torch_dtype=self.dtype, attn_mode="torch", torch_device=self.device
        ).eval().requires_grad_(False)

        if infer_cfg.use_fused_qkv:
            self.engine = FastJointGraphEngine(
                self.raw_model, capacity=20000, window_frames=infer_cfg.window_frames
            )
        else:
            self.engine = self.raw_model

        # Load pre-encoded frozen T5 prompts
        self.texts = torch.load(text_cache_path, map_location="cpu", weights_only=True)
        self.use_cfg = infer_cfg.video_guidance_scale > 1.0 or infer_cfg.action_guidance_scale > 1.0
        self._build_conditioning()

        # Normalization and per-agent embodiment masks m_{e_k} over 30-D slots
        if action_norm is not None:
            codec = ActionCodec.from_file(action_norm)
            q01, q99 = codec.q01, codec.q99
        else:
            q01, q99 = self.va_cfg.norm_stat["q01"], self.va_cfg.norm_stat["q99"]
        self.q01 = torch.tensor(q01, device=self.device, dtype=torch.float32).reshape(1, 30, 1, 1, 1)
        self.q99 = torch.tensor(q99, device=self.device, dtype=torch.float32).reshape(1, 30, 1, 1, 1)
        self.action_clip = float(self.task_info.action_clip)
        self.action_mask = torch.zeros((self.num_agents, 30), dtype=torch.bool, device=self.device)
        for k, robot in enumerate(self.task_info.robots):
            self.action_mask[k, list(EMBODIMENT_SLOTS[robot.embodiment])] = True

        self.frame_st_id = 0
        self.init_latent = None

    def _build_conditioning(self) -> None:
        text = self.texts["text_emb"].unsqueeze(0)
        mask = self.texts["text_mask"].unsqueeze(0)
        glob = self.texts["global_text_emb"].unsqueeze(0)
        gm = self.texts["global_text_mask"].unsqueeze(0)
        if self.use_cfg:
            empty = self.texts["empty_text_emb"]
            em = self.texts["empty_text_mask"]
            text = torch.cat([text, empty[None, None].expand(1, self.num_agents, -1, -1)])
            mask = torch.cat([mask, em[None, None].expand(1, self.num_agents, -1)])
            glob = torch.cat([glob, empty[None]])
            gm = torch.cat([gm, em[None]])
        tl = int(torch.nonzero(mask, as_tuple=False)[:, -1].max().item()) + 1
        gl = int(torch.nonzero(gm, as_tuple=False)[:, -1].max().item()) + 1
        self.condition = {
            "text_emb": text[:, :, :tl].to(self.device, self.dtype),
            "text_mask": mask[:, :, :tl].to(self.device),
            "global_text_emb": glob[:, :gl].to(self.device, self.dtype),
            "global_text_mask": gm[:, :gl].to(self.device),
            "agent_ids": torch.arange(self.num_agents, device=self.device, dtype=torch.long),
        }

    def reset(self, request: Dict[str, Any]) -> Dict[str, Any]:
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        self.frame_st_id = 0
        self.init_latent = None
        if hasattr(self.engine, "reset_episode"):
            self.engine.reset_episode()
        else:
            self.engine.clear_cache("joint")
        if self.cam_streams is not None:
            for stream in self.cam_streams:
                stream.clear_cache()
        else:
            self.head_vae.clear_cache()
            self.wrist_vae.clear_cache()
        return {"agent_ids": list(range(self.num_agents)), "frame_st_id": 0, "preset": self.infer_cfg.preset}

    def encode_obs(self, request: Dict[str, Any]):
        images = request.get("obs")
        if isinstance(images, dict):
            images = [images]
        if not images:
            return None
        streams = []
        for idx, key in enumerate(CAMERA_KEYS):
            raw = np.stack([each[key] for each in images])  # [T, N, H, W, 3]
            x = torch.from_numpy(raw).to(self.device).float().permute(1, 0, 4, 2, 3).flatten(0, 1)
            size = (256, 320) if idx == 0 else (128, 160)
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
            x = x.unflatten(0, (self.num_agents, len(images))).permute(0, 2, 1, 3, 4)
            streams.append((x / 255.0 * 2.0 - 1.0).to(self.dtype))

        if self.cam_streams is not None:
            per_agent = []
            for k in range(self.num_agents):
                h = self.cam_streams[k * 3 + 0].encode_async(streams[0][k : k + 1])
                wl = self.cam_streams[k * 3 + 1].encode_async(streams[1][k : k + 1])
                wr = self.cam_streams[k * 3 + 2].encode_async(streams[2][k : k + 1])
                per_agent.append(torch.cat([torch.cat([wl, wr], dim=-1), h], dim=-2))
            encoded = torch.cat(per_agent, dim=0)
        else:
            head = self.head_vae.encode_chunk(streams[0])
            wrists = torch.stack(streams[1:], dim=1).flatten(0, 1)
            wrist = self.wrist_vae.encode_chunk(wrists).unflatten(0, (self.num_agents, 2))
            encoded = torch.cat([torch.cat([wrist[:, 0], wrist[:, 1]], dim=-1), head], dim=-2)

        mu, _ = encoded.chunk(2, dim=1)
        lat = ((mu.float() - self.latents_mean) / self.latents_std).to(self.dtype)
        return lat.unsqueeze(0)

    def preprocess_action(self, raw_actions) -> torch.Tensor:
        raw = torch.as_tensor(raw_actions, device=self.device, dtype=torch.float32)
        pad = torch.zeros((self.num_agents, 1, *raw.shape[2:]), device=self.device)
        value = torch.cat([raw, pad], dim=1)[:, list(INVERSE_USED_ACTION_CHANNEL_IDS)].unsqueeze(-1)
        value = ((value - self.q01) / (self.q99 - self.q01 + 1e-6) * 2.0 - 1.0).clamp(
            -self.action_clip, self.action_clip
        )
        value = value * self.action_mask[:, :, None, None, None]
        return value.unsqueeze(0).to(self.dtype)

    def postprocess_action(self, value: torch.Tensor) -> np.ndarray:
        v = value[0].float()
        v = (v + 1.0) / 2.0 * (self.q99 - self.q01 + 1e-6) + self.q01
        v = v * self.action_mask[:, :, None, None, None]
        return v[:, list(USED_ACTION_CHANNEL_IDS), ..., 0].cpu().numpy()

    def _model_input(self, value: torch.Tensor, t, action: bool = False, pinned=None) -> Dict[str, Any]:
        value = value.clone()
        times = torch.full((1, self.num_agents, value.shape[3]), float(t), device=self.device)
        if pinned is not None:
            value[:, :, :, 0:1] = pinned
            times[:, :, 0] = 0
        if action:
            value = value * self.action_mask[None, :, :, None, None, None]
        f, h, w = value.shape[-3:]
        if action:
            grid = self._get_mesh_id(f, h, w, 1, 1, self.frame_st_id, action=True)
        else:
            pf, ph, pw = self.va_cfg.patch_size
            grid = self._get_mesh_id(f // pf, h // ph, w // pw, 0, 1, self.frame_st_id)
        batch = 2 if self.use_cfg else 1
        out = dict(self.condition)
        out.update(
            noisy_latents=value.expand(batch, -1, -1, -1, -1, -1),
            timesteps=times.expand(batch, -1, -1),
            grid_id=grid.to(self.device)[None, None].expand(batch, self.num_agents, -1, -1),
            frame_st_id=self.frame_st_id,
            chunk_size=self.chunk_frames,
            window_size=self.infer_cfg.window_frames,
            window_frames=self.infer_cfg.window_frames,
        )
        return out

    def generate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Algorithm 2 Steps (1) Imagine jointly -> (2) Act with mask re-application."""
        if self.frame_st_id == 0:
            self.init_latent = self.encode_obs(request)

        latents = torch.randn((1, self.num_agents, 48, self.chunk_frames, 24, 20), device=self.device, dtype=self.dtype)
        actions = torch.randn(
            (1, self.num_agents, 30, self.chunk_frames, self.action_per_frame, 1),
            device=self.device,
            dtype=self.dtype,
        )
        actions = actions * self.action_mask[None, :, :, None, None, None]

        self.scheduler.set_timesteps(self.infer_cfg.video_inference_steps)
        self.action_scheduler.set_timesteps(self.infer_cfg.action_inference_steps)
        vt = F.pad(self.scheduler.timesteps, (0, 1), value=0)
        video_pin = self.init_latent if self.frame_st_id == 0 else None
        action_pin = (
            self.preprocess_action(np.zeros((self.num_agents, 16, 1, 16), np.float32))
            if self.frame_st_id == 0
            else None
        )

        # (1) Imagine jointly: denoise all robots' future video latents and cache them at t=0
        for idx, t in enumerate(vt):
            last = idx == len(vt) - 1
            pred = self.engine.inference_forward(
                self._model_input(latents, t, pinned=video_pin),
                action_mode=False,
                cache_name="joint",
                update_cache=1 if last else 0,
            )
            if not last:
                b, n = pred.shape[:2]
                pred = self._data_seq_to_patch(
                    self.va_cfg.patch_size, pred.flatten(0, 1), self.chunk_frames, 24, 20, batch_size=b * n
                ).unflatten(0, (b, n))
                if self.infer_cfg.video_guidance_scale > 1.0:
                    pred = pred[1:] + self.infer_cfg.video_guidance_scale * (pred[:1] - pred[1:])
                else:
                    pred = pred[:1]
                latents = self.scheduler.step(pred, t, latents)
            if video_pin is not None:
                latents[:, :, :, 0:1] = video_pin

        # (2) Act: denoise all robots' actions conditioned on imagined video, re-applying m_{e_k}
        for t in self.action_scheduler.timesteps:
            pred = self.engine.inference_forward(
                self._model_input(actions, t, action=True, pinned=action_pin),
                action_mode=True,
                cache_name="joint",
                update_cache=0,
            )
            b, n = pred.shape[:2]
            pred = pred.reshape(b, n, self.chunk_frames, self.action_per_frame, 30).permute(0, 1, 4, 2, 3).unsqueeze(-1)
            pred = pred[:1] * self.action_mask[None, :, :, None, None, None]
            actions = self.action_scheduler.step(pred, t, actions)
            if action_pin is not None:
                actions[:, :, :, 0:1] = action_pin
            if self.infer_cfg.rezero_inactive_action_every_step:
                actions = actions * self.action_mask[None, :, :, None, None, None]

        return {
            "action": self.postprocess_action(actions),
            "frame_st_id": self.frame_st_id,
            "cache": self.engine.cache_summary("joint"),
        }

    def feedback(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Algorithm 2 Step (4): ground sliding KV cache with observed RGB and executed actions."""
        latent = self.encode_obs(request)
        if self.frame_st_id == 0:
            latent = torch.cat([self.init_latent, latent], dim=3) if latent is not None else self.init_latent
        action = self.preprocess_action(request["state"])
        self.engine.clear_pred_cache("joint")
        self.engine.inference_forward(self._model_input(latent, 0), action_mode=False, cache_name="joint", update_cache=2)
        self.engine.inference_forward(
            self._model_input(action, 0, action=True), action_mode=True, cache_name="joint", update_cache=2
        )
        self.frame_st_id += latent.shape[3]
        return {"frame_st_id": self.frame_st_id, "cache": self.engine.cache_summary("joint")}

    @torch.inference_mode()
    def infer(self, request: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        if request.get("reset"):
            res = self.reset(request)
        elif request.get("compute_kv_cache"):
            res = self.feedback(request)
        else:
            res = self.generate(request)
        res["server_seconds"] = round(time.perf_counter() - t0, 4)
        return res


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, help="MA-Bench task slug or alias")
    p.add_argument("--checkpoint", type=Path, required=True, help="Stage 2 checkpoint dir (from EmbodiedCity/MAWAM)")
    p.add_argument("--base-model", type=Path, required=True, help="LingBot-VA base model dir (for frozen VAE)")
    p.add_argument("--text-cache", type=Path, required=True, help="Pre-encoded frozen T5 text_cache.pt")
    p.add_argument("--upstream-root", type=Path, required=True, help="LingBot-VA (wan_va) source checkout")
    p.add_argument("--preset", choices=("benchmark", "realtime"), default="benchmark")
    p.add_argument("--config", type=Path, default=None, help="Optional YAML inference config override")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=25300)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--action-norm", type=Path, help="Per-task action quantiles bundled with a local training split")
    args = p.parse_args()

    add_lingbot_source(args.upstream_root)
    from utils.Simple_Remote_Infer.deploy.websocket_policy_server import WebsocketPolicyServer

    infer_cfg = (
        load_inference_config(args.config)
        if args.config is not None
        else (InferenceConfig.realtime() if args.preset == "realtime" else InferenceConfig.benchmark())
    )
    policy = MawamPolicyServer(
        task_slug=args.task,
        checkpoint_dir=args.checkpoint.resolve(),
        base_model_dir=args.base_model.resolve(),
        text_cache_path=args.text_cache.resolve(),
        infer_cfg=infer_cfg,
        device=args.device,
        seed=args.seed,
        action_norm=args.action_norm,
    )
    print(
        json.dumps(
            {
                "status": "serving",
                "task": args.task,
                "preset": infer_cfg.preset,
                "video_steps": infer_cfg.video_inference_steps,
                "action_steps": infer_cfg.action_inference_steps,
                "window_frames": infer_cfg.window_frames,
                "address": f"ws://{args.host}:{args.port}",
            },
            indent=2,
        ),
        flush=True,
    )
    WebsocketPolicyServer(policy, host=args.host, port=args.port).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
