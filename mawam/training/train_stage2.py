#!/usr/bin/env python3
"""MAWAM Stage 2 (N>=2) Training Entrypoint (Paper Section 4.2-4.3 & Appendix Table 6).

Trains `Stage2WanTransformer3DModel` with weight-shared joint causal self-attention,
gated global cross-attention (`gamma_b=1e-3`, 10x LR `1e-4` with zero weight decay),
local role cross-attention, and fixed Fourier physical ID + normalized role encodings
(`beta=0.01`) on synchronized multi-robot episodes.

Default paper recipe (Table 6):
  - Initialized from the task's Stage 1 checkpoint (or an existing Stage 2 checkpoint)
  - 8x NVIDIA H20 GPUs, FSDP2 bfloat16, full activation checkpointing (all 30 blocks)
  - Global batch size = 8 synchronized episodes (1 per GPU)
  - AdamW (betas=(0.9, 0.95), weight_decay=0.1), constant LR = 1e-5 (gate LR = 1e-4)
  - 3,000 updates per task, joint CFG prompt dropout = 0.1, grad_clip = 2.0
  - Flow matching: 1,000 levels, shift = 5 (video), shift = 1 (action)
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
from pathlib import Path
import random
import time

from ma_bench.action_codec import ActionCodec
from mawam.config import Stage2Config, load_training_config
from mawam.upstream import add_lingbot_source


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/training/stage2.yaml"))
    p.add_argument("--dataset-root", type=Path, required=True, help="Prepared Stage 2 paired dataset root")
    p.add_argument("--upstream-root", type=Path, required=True, help="LingBot-VA (wan_va) source checkout")
    p.add_argument("--initial-checkpoint", type=Path, required=True, help="Stage 1 (or Stage 2) checkpoint directory")
    p.add_argument("--run-root", type=Path, required=True, help="Output directory for logs and checkpoints")
    p.add_argument("--resume-directory", type=Path, default=None)
    p.add_argument("--updates", type=int, default=None, help="Override updates (paper default: 3000)")
    p.add_argument("--save-interval", type=int, default=None)
    p.add_argument("--window-frames", type=int, default=None, help="0 = full synchronized episode per sample")
    p.add_argument("--is-joint-space", action="store_true", help="Use absolute joint actions (real-robot mode)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    train_cfg = load_training_config(args.config)
    assert isinstance(train_cfg, Stage2Config)
    if args.updates is not None:
        train_cfg.updates = int(args.updates)
    if args.save_interval is not None:
        train_cfg.save_interval = int(args.save_interval)
    if args.window_frames is not None:
        train_cfg.window_frames = int(args.window_frames)

    add_lingbot_source(args.upstream_root)
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.utils.data._utils.collate import default_collate

    import train as upstream_train
    from configs import VA_CONFIGS
    from train import Trainer
    from utils import FlowMatchScheduler

    from mawam.data.stage2_dataset import Stage2Dataset
    from mawam.model.stage2_transformer import Stage2WanTransformer3DModel
    from mawam.training.checkpoint import load_training_checkpoint, save_training_checkpoint
    from mawam.training.losses import masked_compute_loss
    from mawam.training.optim import build_optimizer_groups

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")

    random.seed(train_cfg.seed + rank)
    np.random.seed(train_cfg.seed + rank)
    torch.manual_seed(train_cfg.seed + rank)
    torch.cuda.manual_seed_all(train_cfg.seed + rank)
    args.run_root.mkdir(parents=True, exist_ok=True)

    try:
        va_cfg = copy.deepcopy(VA_CONFIGS["robotwin_train"])
        va_cfg.cfg_prob = train_cfg.cfg_dropout_prob
        va_cfg.learning_rate = train_cfg.lr_shared
        va_cfg.warmup_steps = train_cfg.warmup_steps
        va_cfg.weight_decay = train_cfg.weight_decay
        va_cfg.beta1, va_cfg.beta2 = train_cfg.betas
        va_cfg.gradient_accumulation_steps = 1
        va_cfg.snr_shift = train_cfg.video_snr_shift
        va_cfg.action_snr_shift = train_cfg.action_snr_shift
        va_cfg.rank, va_cfg.local_rank, va_cfg.world_size = rank, local_rank, world
        va_cfg.empty_emb_path = str(args.dataset_root / "text" / "empty_emb.pt")

        codec = ActionCodec(
            q01=va_cfg.norm_stat["q01"],
            q99=va_cfg.norm_stat["q99"],
            clip=train_cfg.action_clip,
        )
        dataset = Stage2Dataset(
            dataset_root=args.dataset_root,
            codec=codec,
            window_frames=train_cfg.window_frames,
            cfg_prob=train_cfg.cfg_dropout_prob,
            is_joint_space=args.is_joint_space,
        )
        count = len(dataset)
        epoch_steps = (count + world - 1) // world
        schedule_cache = {}

        def _index_for(update_idx: int):
            epoch, offset = divmod(update_idx, epoch_steps)
            if epoch not in schedule_cache:
                schedule_cache.clear()
                ids = torch.randperm(
                    count, generator=torch.Generator().manual_seed(train_cfg.seed + epoch)
                ).tolist()
                ids += ids[: epoch_steps * world - len(ids)]
                schedule_cache[epoch] = ids
            return schedule_cache[epoch][offset * world + rank], epoch

        source_model = args.initial_checkpoint / "transformer"
        init_class = json.loads((source_model / "config.json").read_text(encoding="utf-8")).get("_class_name")
        if init_class == "Stage2WanTransformer3DModel":
            model, info = Stage2WanTransformer3DModel.from_pretrained(
                str(source_model), torch_dtype=torch.float32, output_loading_info=True
            )
            errs = {k: info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if info.get(k)}
            if errs:
                raise RuntimeError("Stage2 checkpoint did not load cleanly: " + repr(errs))
        else:
            model = Stage2WanTransformer3DModel.from_stage1_pretrained(
                str(source_model),
                torch_dtype=torch.float32,
                stage2_gate_init=train_cfg.stage2_gate_init,
                stage2_role_scale=train_cfg.stage2_role_scale,
                stage2_id_scale=train_cfg.stage2_id_scale,
            )

        model.train().requires_grad_(True)
        for i, block in enumerate(model.blocks):
            model.blocks[i] = checkpoint_wrapper(block, preserve_rng_state=False)
        mp = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False
        )
        for block in model.blocks:
            for mod in (block.attn1, block.attn2, block.global_attn, block.ffn):
                fully_shard(mod, mp_policy=mp, reshard_after_forward=True)
            fully_shard(block, mp_policy=mp, reshard_after_forward=True)
        fully_shard(model, mp_policy=mp, reshard_after_forward=True)

        trainer = Trainer.__new__(Trainer)
        trainer.config, trainer.device, trainer.dtype = va_cfg, device, torch.bfloat16
        trainer.patch_size = va_cfg.patch_size
        trainer.gradient_accumulation_steps = 1
        trainer.transformer = model
        param_groups = build_optimizer_groups(
            model,
            weight_decay=train_cfg.weight_decay,
            lr_shared=train_cfg.lr_shared,
            lr_global=train_cfg.lr_global,
            lr_gate=train_cfg.lr_gate,
        )
        trainer.optimizer = torch.optim.AdamW(
            param_groups, betas=train_cfg.betas, eps=train_cfg.eps, fused=True, foreach=False
        )
        trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.0)
        trainer.step = 0
        for attr, shift in (
            ("train_scheduler_latent", train_cfg.video_snr_shift),
            ("train_scheduler_action", train_cfg.action_snr_shift),
        ):
            sched = FlowMatchScheduler(shift=shift, sigma_min=0.0, extra_one_step=True)
            sched.set_timesteps(train_cfg.num_train_timesteps, training=True)
            setattr(trainer, attr, sched)

        completed = 0
        if args.resume_directory:
            restored = load_training_checkpoint(trainer, args.resume_directory, rank, world, device)
            completed = restored["global_step"]
            rng_state = restored["rng"]
            random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy"])
            torch.set_rng_state(rng_state["torch_cpu"])
            if rng_state["torch_cuda"] is not None:
                torch.cuda.set_rng_state(rng_state["torch_cuda"], device)

        for update in range(completed, train_cfg.updates):
            t0 = time.perf_counter()
            idx, epoch = _index_for(update)
            batch = default_collate([dataset[idx]])
            tbatch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            b, n = tbatch["latents"].shape[:2]
            flat_batch = {k: tbatch[k].flatten(0, 1) for k in ("latents", "actions", "actions_mask", "text_emb")}
            flat = trainer._prepare_input_dict(flat_batch)
            video_mask = tbatch["video_spatial_mask"].flatten(0, 1)

            paired = {
                key: {k: v.unflatten(0, (b, n)) if isinstance(v, torch.Tensor) else v for k, v in vals.items()}
                for key, vals in flat.items()
                if key in ("latent_dict", "action_dict")
            }
            paired.update({k: flat[k] for k in ("chunk_size", "window_size")})
            paired["latent_dict"]["text_mask"] = tbatch["text_mask"]
            for k in ("global_text_emb", "global_text_mask", "agent_ids"):
                paired[k] = tbatch[k]

            prediction = model(paired, train_mode=True)
            loss_inputs = {k: dict(v) if isinstance(v, dict) else v for k, v in flat.items()}
            loss_inputs["action_dict"]["actions_mask"] = tbatch["action_loss_mask"].flatten(0, 1)

            agent_losses = []
            for agent_idx in range(n):
                positions = torch.arange(b, device=device) * n + agent_idx
                one = {
                    key: {k: v[positions] for k, v in vals.items()}
                    for key, vals in loss_inputs.items()
                    if key in ("latent_dict", "action_dict")
                }
                lv, la = masked_compute_loss(
                    trainer,
                    one,
                    (prediction[0][:, agent_idx], prediction[1][:, agent_idx]),
                    video_mask[positions],
                    upstream_train.data_seq_to_patch,
                )
                agent_losses.extend((lv, la))
            loss_tensor = torch.stack(agent_losses)
            loss = loss_tensor.sum() / n
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            trainer.optimizer.step()
            trainer.lr_scheduler.step()
            trainer.optimizer.zero_grad(set_to_none=True)
            trainer.step = completed = update + 1
            trainer.training_cursor = {"global_step": completed, "epoch": epoch}

            averages = loss_tensor.detach().double()
            dist.all_reduce(averages, op=dist.ReduceOp.AVG)
            if rank == 0:
                row = {
                    "stage": 2,
                    "step": completed,
                    "epoch": epoch,
                    "video_loss": float(averages[::2].mean().item()),
                    "action_loss": float(averages[1::2].mean().item()),
                    "loss": float(averages.sum().item() / n),
                    "grad_norm": float(grad_norm.detach().item()),
                    "step_s": round(time.perf_counter() - t0, 3),
                }
                with (args.run_root / "metrics.jsonl").open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
                print("STAGE2_STEP", json.dumps(row), flush=True)

            if completed % 50 == 0:
                gc.collect()
            if completed % train_cfg.save_interval == 0 or completed == train_cfg.updates:
                save_training_checkpoint(
                    trainer, completed, args.run_root, rank, world, device, experiment_tag="mawam_stage2"
                )
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
