"""Unified 30-D masked action codec and video spatial loss masks (Paper §4.1 & Appendix D.1).

Shared 30-D slot layout:
  - Arm 0 (7-D pose or 6/7-DoF joints): channels 0..6, gripper channel 28
  - Arm 1 (7-D pose or 6-DoF joints):   channels 7..13, gripper channel 29
  - Active channels per embodiment:
      * Dual-arm ALOHA (sim, 16-D):        [0..13, 28, 29]
      * Single-arm Panda / UR5 (sim, 8-D): [0..6, 28]
      * Franka FR3 (real, 8-D):            [0..6, 28]
      * AgileX Cobot Magic (real, 14-D):   [0..5, 7..12, 28, 29]

Video latent canvas per robot: (48, F, 24, 20)
  - Rows 0..7,   Cols 0..9:  left wrist  (8 x 10)
  - Rows 0..7,   Cols 10..19: right wrist (8 x 10)  [masked False in loss for single-arm]
  - Rows 8..23,  Cols 0..19: head        (16 x 20)
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

ACTION_DIM = 30
ACTION_PER_FRAME = 16
LATENT_CANVAS_HW: Tuple[int, int] = (24, 20)

# Mapping from 16-D raw bimanual vector [arm0(0..7), arm1(8..15)] into 30-D model slots:
# slot 0..6 -> raw 0..6, slot 7..13 -> raw 8..14, slot 14..27 -> pad(16), slot 28 -> raw 7, slot 29 -> raw 15
INVERSE_USED_ACTION_CHANNEL_IDS: Tuple[int, ...] = (
    0, 1, 2, 3, 4, 5, 6,
    8, 9, 10, 11, 12, 13, 14,
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    7, 15,
)
USED_ACTION_CHANNEL_IDS: Tuple[int, ...] = (
    0, 1, 2, 3, 4, 5, 6, 28,
    7, 8, 9, 10, 11, 12, 13, 29,
)

EMBODIMENT_SLOTS: Dict[str, Tuple[int, ...]] = {
    "aloha": tuple(range(0, 14)) + (28, 29),
    "aloha_dual": tuple(range(0, 14)) + (28, 29),
    "panda": tuple(range(0, 7)) + (28,),
    "panda_single": tuple(range(0, 7)) + (28,),
    "ur5": tuple(range(0, 7)) + (28,),
    "ur5_single": tuple(range(0, 7)) + (28,),
    "franka_fr3": tuple(range(0, 7)) + (28,),
    "franka_single": tuple(range(0, 7)) + (28,),
    "cobot_magic": tuple(range(0, 6)) + tuple(range(7, 13)) + (28, 29),
    "cobot_dual": tuple(range(0, 6)) + tuple(range(7, 13)) + (28, 29),
}


def embodiment_action_mask(embodiment: str) -> np.ndarray:
    """Return a `(30,)` boolean mask of active action slots for `embodiment`."""
    key = embodiment.strip().lower()
    if key not in EMBODIMENT_SLOTS:
        raise KeyError(f"Unknown embodiment {embodiment!r}; expected one of {sorted(EMBODIMENT_SLOTS)}")
    mask = np.zeros(ACTION_DIM, dtype=bool)
    mask[list(EMBODIMENT_SLOTS[key])] = True
    return mask


def video_spatial_loss_mask(embodiment: str) -> np.ndarray:
    """Return `(24, 20)` boolean spatial video loss mask (excludes black right wrist on single-arm)."""
    key = embodiment.strip().lower()
    mask = np.ones(LATENT_CANVAS_HW, dtype=bool)
    if key in ("panda", "panda_single", "ur5", "ur5_single"):
        mask[:8, 10:20] = False
    return mask


def _quat_mul_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        axis=-1,
    )


def _quat_conj_xyzw(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., :3] *= -1.0
    return out


def _quat_norm(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(n < 1e-8):
        raise ValueError("Encountered near-zero quaternion norm")
    return q / n


def relative_from_absolute(absolute_wxyz: np.ndarray, reference_wxyz: np.ndarray, valid_dims: np.ndarray | None = None) -> np.ndarray:
    """Convert raw absolute poses `(..., 16)` (wxyz) to frame-0 relative poses (xyzw)."""
    abs_arr = np.asarray(absolute_wxyz, dtype=np.float64)
    ref_arr = np.asarray(reference_wxyz, dtype=np.float64)
    if abs_arr.shape[-1] != 16 or ref_arr.shape != (16,):
        raise ValueError(f"Expected (..., 16) and (16,), got {abs_arr.shape} and {ref_arr.shape}")
    valid = np.ones(16, dtype=bool) if valid_dims is None else np.asarray(valid_dims, dtype=bool)
    out = np.zeros_like(abs_arr)
    for offset in (0, 8):
        if not valid[offset]:
            continue
        out[..., offset : offset + 3] = abs_arr[..., offset : offset + 3] - ref_arr[offset : offset + 3]
        ref_xyzw = _quat_norm(ref_arr[offset + 3 : offset + 7][[1, 2, 3, 0]])
        q_xyzw = _quat_norm(abs_arr[..., offset + 3 : offset + 7][..., [1, 2, 3, 0]])
        rel_xyzw = _quat_norm(_quat_mul_xyzw(_quat_conj_xyzw(ref_xyzw), q_xyzw))
        out[..., offset + 3 : offset + 7] = rel_xyzw
        out[..., offset + 7] = abs_arr[..., offset + 7]
    return out


def absolute_from_relative(relative_xyzw: np.ndarray, reference_wxyz: np.ndarray, valid_dims: np.ndarray | None = None) -> np.ndarray:
    """Convert frame-0 relative poses `(..., 16)` (xyzw) back to absolute poses (wxyz)."""
    rel_arr = np.asarray(relative_xyzw, dtype=np.float64)
    ref_arr = np.asarray(reference_wxyz, dtype=np.float64)
    if rel_arr.shape[-1] != 16 or ref_arr.shape != (16,):
        raise ValueError(f"Expected (..., 16) and (16,), got {rel_arr.shape} and {ref_arr.shape}")
    valid = np.ones(16, dtype=bool) if valid_dims is None else np.asarray(valid_dims, dtype=bool)
    out = np.zeros_like(rel_arr)
    for offset in (0, 8):
        if not valid[offset]:
            continue
        out[..., offset : offset + 3] = rel_arr[..., offset : offset + 3] + ref_arr[offset : offset + 3]
        ref_xyzw = _quat_norm(ref_arr[offset + 3 : offset + 7][[1, 2, 3, 0]])
        q_xyzw = _quat_norm(rel_arr[..., offset + 3 : offset + 7])
        abs_xyzw = _quat_norm(_quat_mul_xyzw(ref_xyzw, q_xyzw))
        out[..., offset + 3 : offset + 7] = abs_xyzw[..., [3, 0, 1, 2]]
        out[..., offset + 7] = rel_arr[..., offset + 7]
    return out


def unpack_action_chunk(raw_chunk: np.ndarray, first_chunk: bool) -> np.ndarray:
    """Unpack `[N, 16, C=2, 16]` into executable `[N, 16|32, 16]` control commands."""
    arr = np.asarray(raw_chunk, dtype=np.float64)
    if arr.ndim != 4 or arr.shape[1:] != (16, 2, 16) or not np.isfinite(arr).all():
        raise ValueError(f"Expected finite [N, 16, 2, 16], got {arr.shape}")
    skip = 1 if first_chunk else 0
    return arr[:, :, skip:, :].transpose(0, 2, 3, 1).reshape(arr.shape[0], -1, 16)


def pack_executed_feedback(executed_relative: np.ndarray, first_chunk: bool) -> np.ndarray:
    """Pack `[N, 16|32, 16]` executed relative actions into `[N, 16, 2, 16]` for KV grounding."""
    arr = np.asarray(executed_relative, dtype=np.float32)
    expected = 16 if first_chunk else 32
    if arr.ndim != 3 or arr.shape[1:] != (expected, 16) or not np.isfinite(arr).all():
        raise ValueError(f"Expected finite [N, {expected}, 16], got {arr.shape}")
    out = np.zeros((arr.shape[0], 16, 2, 16), dtype=np.float32)
    skip = int(first_chunk)
    out[:, :, skip:, :] = arr.reshape(arr.shape[0], 2 - skip, 16, 16).transpose(0, 3, 1, 2)
    return out


@dataclass
class ActionCodec:
    """1%/99% quantile normalizer and 30-D slot mapper matching Paper Appendix D.1."""

    q01: np.ndarray
    q99: np.ndarray
    clip: float = 5.0

    def __post_init__(self) -> None:
        self.q01 = np.asarray(self.q01, dtype=np.float64)
        self.q99 = np.asarray(self.q99, dtype=np.float64)
        if self.q01.shape != (ACTION_DIM,) or self.q99.shape != (ACTION_DIM,):
            raise ValueError("q01 and q99 must have shape (30,)")
        if not np.isfinite(self.q01).all() or not np.isfinite(self.q99).all() or np.any(self.q01 > self.q99):
            raise ValueError("Action quantiles must be finite and ordered")

    @classmethod
    def from_file(cls, path: Path, clip: float = 5.0) -> ActionCodec:
        stats = json.loads(path.read_text(encoding="utf-8"))
        return cls(q01=stats["q01"], q99=stats["q99"], clip=clip)

    def encode_slots(self, raw_rel_16: np.ndarray, embodiment: str) -> Tuple[np.ndarray, np.ndarray]:
        """Map `[T, 16]` relative actions -> normalized clipped `[T, 30]` + bool mask `[T, 30]`."""
        raw = np.asarray(raw_rel_16, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != 16:
            raise ValueError(f"Expected [T, 16], got {raw.shape}")
        padded = np.pad(raw, ((0, 0), (0, 1)))
        slots = padded[:, list(INVERSE_USED_ACTION_CHANNEL_IDS)]
        mask_1d = embodiment_action_mask(embodiment)
        mask = np.broadcast_to(mask_1d, slots.shape).copy()
        norm = (slots - self.q01[None, :]) / (self.q99[None, :] - self.q01[None, :] + 1e-6) * 2.0 - 1.0
        norm = np.clip(norm, -self.clip, self.clip)
        norm[:, ~mask_1d] = 0.0
        return norm.astype(np.float32), mask

    def decode_slots(self, norm_30: np.ndarray, embodiment: str) -> np.ndarray:
        """Map normalized `[..., 30]` back to `[..., 16]` raw relative space with inactive zeroed."""
        arr = np.asarray(norm_30, dtype=np.float64)
        if arr.shape[-1] != ACTION_DIM:
            raise ValueError(f"Expected last dim 30, got {arr.shape}")
        mask_1d = embodiment_action_mask(embodiment)
        denorm = (arr + 1.0) / 2.0 * (self.q99 - self.q01 + 1e-6) + self.q01
        denorm = np.where(mask_1d, denorm, 0.0)
        return denorm[..., list(USED_ACTION_CHANNEL_IDS)]
