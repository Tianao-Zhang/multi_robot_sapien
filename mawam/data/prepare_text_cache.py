#!/usr/bin/env python3
"""Pre-encode global instruction g, local role instructions l^k, and empty CFG prompt via frozen T5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ma_bench.registry import get_task_info
from mawam.upstream import add_lingbot_source


def encode_prompts(
    model_dir: Path,
    output_dir: Path,
    global_prompt: str,
    local_prompts: list[str],
    device: str = "cuda",
) -> Path:
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from modules.utils import load_text_encoder, load_tokenizer

    tokenizer = load_tokenizer(str(model_dir / "tokenizer"))
    encoder = load_text_encoder(str(model_dir / "text_encoder"), torch.bfloat16, device).eval()
    encoder.requires_grad_(False)

    def _encode_one(text: str):
        tokens = tokenizer(
            [prompt_clean(text)],
            padding="max_length",
            max_length=512,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        mask = tokens.attention_mask.to(device)
        with torch.inference_mode():
            emb = encoder(tokens.input_ids.to(device), mask).last_hidden_state[0]
        valid = mask[0].bool().cpu()
        emb = emb.to(device="cpu", dtype=torch.bfloat16).contiguous()
        emb[~valid] = 0
        return emb, valid

    local_embs, local_masks = [], []
    for lp in local_prompts:
        e, m = _encode_one(lp)
        local_embs.append(e)
        local_masks.append(m)
    g_emb, g_mask = _encode_one(global_prompt)
    empty_emb, empty_mask = _encode_one("")

    text_dir = output_dir / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    out_path = text_dir / "text_cache.pt"
    torch.save(
        {
            "global_prompt": global_prompt,
            "local_prompts": list(local_prompts),
            "text_emb": torch.stack(local_embs),
            "text_mask": torch.stack(local_masks),
            "global_text_emb": g_emb,
            "global_text_mask": g_mask,
            "empty_text_emb": empty_emb,
            "empty_text_mask": empty_mask,
        },
        out_path,
    )
    torch.save(empty_emb, text_dir / "empty_emb.pt")
    (text_dir / "text_meta.json").write_text(
        json.dumps(
            {
                "global_prompt": global_prompt,
                "local_prompts": list(local_prompts),
                "global_valid_tokens": int(g_mask.sum()),
                "local_valid_tokens": [int(m.sum()) for m in local_masks],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return out_path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, help="MA-Bench task slug or alias")
    p.add_argument("--base-model", type=Path, required=True, help="Path to LingBot-VA base model (contains tokenizer/ & text_encoder/)")
    p.add_argument("--output", type=Path, required=True, help="Dataset root where text/text_cache.pt will be saved")
    p.add_argument("--device", default="cuda")
    p.add_argument("--upstream-root", type=Path, default=Path("third_party/LingBot-VA"))
    args = p.parse_args()

    add_lingbot_source(args.upstream_root)
    info = get_task_info(args.task)
    out = encode_prompts(
        model_dir=args.base_model.resolve(),
        output_dir=args.output.resolve(),
        global_prompt=info.global_prompt,
        local_prompts=list(info.local_prompts),
        device=args.device,
    )
    print(f"Saved frozen T5 text cache to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
