#!/usr/bin/env python3
"""MAWAM Stage 1 (N=1) Training Entrypoint (Paper Section 4.3 & Appendix Table 6).

Trains one shared LingBot-VA backbone on per-robot whole-episode trajectories
(with 1:1 embodiment balancing on heterogeneous teams, masked 30-D action slots,
and view-masked video loss for single-arm missing wrist views).

Default paper recipe (Table 6):
  - 8x NVIDIA H20 GPUs, FSDP2 bfloat16, full activation checkpointing
  - Global batch size = 8 episodes (1 per GPU)
  - AdamW (betas=(0.9, 0.95), weight_decay=0.1), constant LR = 1e-5, grad_clip = 2.0
  - 3,000 updates per task, CFG text dropout = 0.1
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
from mawam.config import Stage1Config, load_training_config
from mawam.upstream import add_lingbot_source


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/training/stage1.yaml"))
    p.add_argument("--dataset-root", type=Path, required=True, help="Prepared Stage 1 dataset root")
    p.add_argument("--upstream-root", type=Path, required=True, help="LingBot-VA (wan_va) source checkout")
    p.add_argument("--initial-checkpoint", type=Path, required=True, help="Base LingBot-VA checkpoint directory")
    p.add_argument("--run-root", type=Path, required=True, help="Output directory for logs and checkpoints")
    p.add_argument("--resume-directory", type=Path, default=None)
    p.add_argument("--updates", type=int, default=None, help="Override updates (paper default: 3000)")
    p.add_argument("--save-interval", type=int, default=None)
    p.add_argument("--is-joint-space", action="store_true", help="Use absolute joint actions (real-robot mode)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    train_cfg = load_training_config(args.config)
    assert isinstance(train_cfg, Stage1Config)
    if args.updates is not None:
        train_cfg.updates = int(args.updates)
    if args.save_interval is not None:
        train_cfg.save_interval = int(args.save_interval)

    add_lingbot_source(args.upstream_root)
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.utils.data._utils.collate import default_collate

    import train as upstream_train
    from configs import VA_CONFIGS
    from modules.model import WanTransformer3DModel
    from train import Trainer
    from utils import FlowMatchScheduler

    from mawam.data.stage1_dataset import Stage1Dataset
    from mawam.model.mask_adapter import install_stage1_mask_adapter
    from mawam.training.checkpoint import load_training_checkpoint, save_training_checkpoint

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
        va_cfg.learning_rate = train_cfg.lr
        va_cfg.warmup_steps = train_cfg.warmup_steps
        va_cfg.weight_decay = train_cfg.weight_decay
        va_cfg.beta1, va_cfg.beta2 = train_cfg.betas
        va_cfg.gradient_accumulation_steps = 1
        va_cfg.snr_shift = train_cfg.video_snr_shift
        va_cfg.action_snr_shift = train_cfg.action_snr_shift
        va_cfg.rank, va_cfg.local_rank, va_cfg.world_size = rank, local_rank, world
        va_cfg.empty_emb_path = str(args.dataset_root / "text" / "empty_emb.pt")

        norm_path = args.dataset_root / "action_norm.json"
        codec = ActionCodec.from_file(norm_path, clip=train_cfg.action_clip) if norm_path.is_file() else ActionCodec(
            q01=va_cfg.norm_stat["q01"],
            q99=va_cfg.norm_stat["q99"],
            clip=train_cfg.action_clip,
        )
        dataset = Stage1Dataset(
            dataset_root=args.dataset_root,
            codec=codec,
            cfg_prob=train_cfg.cfg_dropout_prob,
            is_joint_space=args.is_joint_space,
        )

        # Balanced 1:1 agent sampling order across heterogeneous or homogeneous embodiments
        by_agent = {}
        for idx, ep in enumerate(dataset.episodes):
            by_agent.setdefault(int(ep["agent_id"]), []).append(idx)
        agent_keys = sorted(by_agent)
        epoch_size = max(len(v) for v in by_agent.values()) * len(agent_keys)
        epoch_steps = (epoch_size + world - 1) // world

        def _index_for(update_idx: int):
            epoch, offset = divmod(update_idx, epoch_steps)
            rng = random.Random(train_cfg.seed + epoch)
            pools = {k: list(v) for k, v in by_agent.items()}
            for v in pools.values():
                rng.shuffle(v)
            interleaved = []
            max_len = max(len(v) for v in pools.values())
            for i in range(max_len):
                for k in agent_keys:
                    interleaved.append(pools[k][i % len(pools[k])])
            interleaved += [interleaved[i % epoch_size] for i in range(epoch_steps * world - epoch_size)]
            return interleaved[offset * world + rank], epoch

        model = WanTransformer3DModel.from_pretrained(
            str(args.initial_checkpoint / "transformer"), torch_dtype=torch.float32, attn_mode="flex"
        )
        model.train().requires_grad_(True)
        for i, block in enumerate(model.blocks):
            model.blocks[i] = checkpoint_wrapper(block, preserve_rng_state=False)
        mp = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False
        )
        for block in model.blocks:
            fully_shard(block, mp_policy=mp, reshard_after_forward=True)
        fully_shard(model, mp_policy=mp, reshard_after_forward=True)

        trainer = Trainer.__new__(Trainer)
        trainer.config, trainer.device, trainer.dtype = va_cfg, device, torch.bfloat16
        trainer.patch_size = va_cfg.patch_size
        trainer.gradient_accumulation_steps = 1
        trainer.transformer = model
        install_stage1_mask_adapter(trainer, upstream_train)

        trainer.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg.lr,
            betas=train_cfg.betas,
            eps=train_cfg.eps,
            weight_decay=train_cfg.weight_decay,
            fused=True,
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
            batch = trainer.convert_input_format(batch)
            inputs = trainer._prepare_input_dict(batch)
            pred = model(inputs, train_mode=True)
            v_loss, a_loss = trainer.compute_loss(inputs, pred)
            loss = v_loss + a_loss
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            trainer.optimizer.step()
            trainer.lr_scheduler.step()
            trainer.optimizer.zero_grad(set_to_none=True)
            trainer.step = completed = update + 1
            trainer.training_cursor = {"global_step": completed, "epoch": epoch}

            if rank == 0:
                row = {
                    "stage": 1,
                    "step": completed,
                    "epoch": epoch,
                    "video_loss": float(v_loss.detach().item()),
                    "action_loss": float(a_loss.detach().item()),
                    "loss": float(loss.detach().item()),
                    "grad_norm": float(grad_norm.detach().item()),
                    "step_s": round(time.perf_counter() - t0, 3),
                }
                with (args.run_root / "metrics.jsonl").open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
                print("STAGE1_STEP", json.dumps(row), flush=True)

            if completed % 50 == 0:
                gc.collect()
            if completed % train_cfg.save_interval == 0 or completed == train_cfg.updates:
                save_training_checkpoint(
                    trainer, completed, args.run_root, rank, world, device, experiment_tag="mawam_stage1"
                )
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
