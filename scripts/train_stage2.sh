#!/usr/bin/env bash
# Stage 2 (N>=2) single-node FSDP2 training launcher
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:?Set DATASET_ROOT to prepared Stage 2 paired dataset directory}"
UPSTREAM_ROOT="${UPSTREAM_ROOT:-./third_party/LingBot-VA}"
STAGE1_CKPT="${STAGE1_CKPT:?Set STAGE1_CKPT to Stage 1 checkpoint directory}"
RUN_ROOT="${RUN_ROOT:-./runs/stage2}"
NPROC="${NPROC:-8}"

torchrun --standalone --nproc_per_node="${NPROC}" \
  -m mawam.training.train_stage2 \
  --config configs/training/stage2.yaml \
  --dataset-root "${DATASET_ROOT}" \
  --upstream-root "${UPSTREAM_ROOT}" \
  --initial-checkpoint "${STAGE1_CKPT}" \
  --run-root "${RUN_ROOT}" \
  "$@"
