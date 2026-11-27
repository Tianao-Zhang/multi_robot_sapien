#!/usr/bin/env bash
# Stage 1 (N=1) single-node FSDP2 training launcher
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:?Set DATASET_ROOT to prepared Stage 1 dataset directory}"
UPSTREAM_ROOT="${UPSTREAM_ROOT:-./third_party/LingBot-VA}"
BASE_CKPT="${BASE_CKPT:-./checkpoints/lingbot-va-base}"
RUN_ROOT="${RUN_ROOT:-./runs/stage1}"
NPROC="${NPROC:-8}"

torchrun --standalone --nproc_per_node="${NPROC}" \
  -m mawam.training.train_stage1 \
  --config configs/training/stage1.yaml \
  --dataset-root "${DATASET_ROOT}" \
  --upstream-root "${UPSTREAM_ROOT}" \
  --initial-checkpoint "${BASE_CKPT}" \
  --run-root "${RUN_ROOT}" \
  "$@"
