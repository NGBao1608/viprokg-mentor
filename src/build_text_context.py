#!/usr/bin/env python3
"""Single source of truth for building the context-text tensor `T_ctx.pt`.

The recipe used to be inlined in several notebook generators. Keeping copies invites drift, and
drift here means a silently different instrument, so every caller imports `build_text_context()`
from this file.

The body of `build_text_context()` is a verbatim copy of the original recipe. Do not change it:
every published `T_ctx.pt` (MKG-W / MKG-Y / DB15K) was produced by exactly these lines, and
`build_text_context_masked.py` asserts that re-running it with `mask_spans=None` reproduces
them bit-for-bit.
"""
from __future__ import annotations

import json
from typing import Callable

import torch

# mask_spans(entity_id, raw_text) -> (masked text, number of removed tokens)
MaskFn = Callable[[int, str], "tuple[str, int]"]


def load_entity_ids(data_dir: str) -> dict[str, int]:
    """Entity name -> id, from `entity2id.tsv` (header line skipped)."""
    with open(f"{data_dir}/entity2id.tsv", encoding="utf-8") as f:
        return {r.split("\t")[0]: int(r.split("\t")[1]) for r in f.read().splitlines()[1:]}


def build_text_context(
    dataset: str,
    data_dir: str,
    textual: str,
    tok,
    model,
    *,
    mask_spans: MaskFn | None = None,
    batch: int = 64,
    device: str = "cuda",
) -> tuple[torch.Tensor, dict]:
    """Build the [N, 768] context-text tensor.

    `mask_spans=None` reproduces `T_ctx.pt`. Passing `mask_spans` masks each description first,
    which is how `T_ctx_mask.pt` is produced.

    Returns `(T, stats)`; `stats` carries the masking counters (zeros when `mask_spans is None`).
    """
    e2i = load_entity_ids(data_dir)
    with open(f"{data_dir}/entities.txt", encoding="utf-8") as f:
        N = sum(1 for _ in f)
    with open(textual, encoding="utf-8") as f:
        token_ids = json.load(f)
    T = torch.zeros(N, 768)
    # ViMMKG2 segment json is keyed by integer entity id (as in DB15K), not by QID.
    # Looking keys up through e2i would give None for every key and an all-zero tensor.
    int_keys = dataset in ("DB15K", "ViMMKG2")
    keys = [k for k in token_ids if (int(k) if int_keys else e2i.get(k)) is not None]
    stats = {"n_sua": 0, "n_xoa": 0, "vi_du": [], "che_txt": {}}

    with torch.no_grad():
        for start in range(0, len(keys), batch):
            ks = keys[start : start + batch]
            # ---- decode: verbatim from the original recipe ----
            texts = [
                tok.decode(
                    [x for x in token_ids[k] if x < tok.vocab_size], skip_special_tokens=True
                )[:2000]
                for k in ks
            ]
            if mask_spans is not None:
                new_texts = []
                for k, raw in zip(ks, texts):
                    i = int(k) if int_keys else e2i[k]
                    masked, n_removed = mask_spans(i, raw)
                    if n_removed:
                        stats["n_sua"] += 1
                        stats["n_xoa"] += n_removed
                        if len(stats["vi_du"]) < 5:
                            stats["vi_du"].append((i, raw[:150], masked[:150]))
                    stats["che_txt"][i] = masked
                    new_texts.append(masked)
                texts = new_texts
            # ---- encode + masked mean: verbatim from the original recipe ----
            enc = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(
                device
            )
            o = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (o * m).sum(1) / m.sum(1).clamp(min=1)
            for j, k in enumerate(ks):
                i = int(k) if int_keys else e2i[k]
                if i < N:
                    T[i] = pooled[j].float().cpu()
    return T, stats


def load_encoder(encoder: str, device: str = "cuda"):
    """Tokenizer + model in eval mode, same two lines the notebooks used."""
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(encoder)
    model = AutoModel.from_pretrained(encoder).to(device).eval()
    return tok, model
