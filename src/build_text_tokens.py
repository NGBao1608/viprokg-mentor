#!/usr/bin/env python3
"""Frozen BERT token embeddings for the Reader. One dataset, one variant, per call.

Output: `{out}` = dict with
    tok  : fp16 [N, L, 768]   per-token last-hidden-state, zero rows for entities with no text
    mask : bool [N, L]        True = real token (NOT the MyGO convention, see below)
    meta : {bo, bien, max_len, encoder, sha_textual, n_co_text}

Mask convention. MyGO's image mask uses `True = MISSING` (`data.py`), and getting that backwards
once already zeroed an entire modality silently. This tensor is new and consumed only by the
Reader, so it uses the ordinary reading, `True = real token`, and asserts it: an entity with
text must have `mask.any()`, and the count of all-False rows must equal the count of entities
with no description.

Size: fp16 [N, 96, 768] is 2.06 GB for N = 15 000. The Reader needs both the masked and the
original variant, so a dataset costs ~4.1 GB and three datasets 12.4 GB. The notebook therefore
builds these inside the per-dataset loop and deletes them after that dataset's runs. Nothing
here goes into a zip.

Text source:
  `--variant goc` -> the original `<dataset>-textual.json`
  `--variant che` -> the masked token json
Both are decoded with the same expression the encoders use, so the Reader sees exactly the
string the rest of the pipeline saw.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", "--bo", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--textual", required=True, help="<dataset>-textual.json (original or masked)")
    ap.add_argument("--variant", "--bien", required=True, choices=["goc", "che"])
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_len", type=int, default=96)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.encoder)
    model = AutoModel.from_pretrained(a.encoder).to(dev).eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)  # frozen: the Reader learns W_tok only

    d = Path(a.data_dir)
    e2i = {
        r.split("\t")[0]: int(r.split("\t")[1])
        for r in (d / "entity2id.tsv").read_text(encoding="utf-8").splitlines()[1:]
    }
    N = sum(1 for _ in (d / "entities.txt").open(encoding="utf-8"))
    raw = Path(a.textual).read_bytes()
    token_ids = json.loads(raw)

    T = torch.zeros(N, a.max_len, 768, dtype=torch.float16)
    M = torch.zeros(N, a.max_len, dtype=torch.bool)
    # Same key convention as `build_text_context`: DB15K keys the json by the numeric id, the
    # others by the raw entity string. Guessing this wrong silently encodes the wrong entities.
    keys = [k for k in token_ids if (int(k) if a.dataset == "DB15K" else e2i.get(k)) is not None]
    print(
        f"[T_tok] {a.dataset}/{a.variant}: N={N} - entities with text {len(keys)} - "
        f"max_len {a.max_len}",
        flush=True,
    )

    n_with_text = 0
    with torch.no_grad():
        for s0 in range(0, len(keys), a.batch):
            ks = keys[s0 : s0 + a.batch]
            texts = [
                tok.decode(
                    [x for x in token_ids[k] if x < tok.vocab_size], skip_special_tokens=True
                )[:2000]
                for k in ks
            ]
            enc = tok(
                texts,
                padding="max_length",
                truncation=True,
                max_length=a.max_len,
                return_tensors="pt",
            ).to(dev)
            out = model(**enc).last_hidden_state  # [b, L, 768]
            am = enc["attention_mask"].bool()
            for j, k in enumerate(ks):
                i = int(k) if a.dataset == "DB15K" else e2i[k]
                if i >= N:
                    continue
                T[i] = out[j].half().cpu()
                M[i] = am[j].cpu()
                n_with_text += 1
            if s0 % (a.batch * 40) == 0:
                print(f"   {s0}/{len(keys)}", flush=True)

    # assert the mask means what the docstring says, measured not assumed
    n_empty = int((~M.any(dim=1)).sum())
    n_expected = N - n_with_text
    assert (
        n_empty == n_expected
    ), f"mask convention wrong: {n_empty} all-False rows but {n_expected} entities lack text"
    assert (
        M[T.abs().sum(dim=(1, 2)) > 0].any(dim=1).all()
    ), "an entity with an embedding has an all-False mask: convention inverted"

    meta = {
        "bo": a.dataset,
        "bien": a.variant,
        "max_len": a.max_len,
        "encoder": a.encoder,
        "sha_textual": hashlib.sha256(raw).hexdigest()[:16],
        "n_co_text": n_with_text,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"tok": T, "mask": M, "meta": meta}, a.out)
    gb = T.numel() * 2 / 1024**3
    print(
        f"[T_tok] wrote {a.out} - tok {tuple(T.shape)} fp16 = {gb:.2f} GB - "
        f"mask {tuple(M.shape)} - empty rows {n_empty} - sha_textual {meta['sha_textual']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
