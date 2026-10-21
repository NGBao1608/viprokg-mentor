#!/usr/bin/env python3
"""Build `T_ctx_mask.pt`: the context-text tensor from answer-name-masked descriptions.

For every entity `e` we take its original description, delete every NAME of e's TRAIN partners
from it, then re-encode with exactly the pipeline that produced `T_ctx.pt`.

The encoding pipeline is copied, not rewritten: it lives in `build_text_context.py`
(`build_text_context`). Per entity it is:

    txt    = tok.decode([x for x in tokens[k] if x < tok.vocab_size],
                        skip_special_tokens=True)[:2000]
    enc    = tok(txt, padding=True, truncation=True, max_length=256, return_tensors="pt")
    o      = model(**enc).last_hidden_state
    m      = enc["attention_mask"].unsqueeze(-1).float()
    pooled = (o * m).sum(1) / m.sum(1).clamp(min=1)          # masked mean, dim 768

The only change here is the extra masking step applied to `txt` before re-encoding. Entities
with no text keep the zero vector, as in the original.

Partner set of e = every entity sharing a TRAIN triple with e, in either direction. Names are
normalised with the mention rule and matched longest-first, so a long name is never chopped in
half by an earlier short match.

Assert: after masking, recounting name-mention edges over TRAIN pairs gives 0.

Usage (GPU):
    python3 build_text_context_masked.py --dataset MKG-W --data_dir <dir> \
        --textual <MKG-W-textual.json> --encoder <bert-base-uncased dir> --out T_ctx_mask.pt
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

# The masking rule MUST be the rule that defines a name-mention edge, otherwise the assert below
# only tests our own invention. The canonical definition lives in:
#     label_leak_probe.py   : normalize_words(s) -> lower-cased token list
#     make_masked_text.py   : load_labels(dataset) -> entity label as a token list;
#                             MAX_LABEL_WORDS = 8 is the longest label (in words) considered
#     mention_only_baseline.py : a label matches when its token tuple occurs as a contiguous
#                             n-gram of the description's token list
# So a mention is an n-gram match over normalised tokens, not a regex over raw text. We tokenise
# the raw text keeping offsets, match n-grams on the normalised tokens, and delete the matching
# raw spans; the untouched remainder of the raw text is preserved so the encoder still sees
# ordinary prose.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_leak_probe import normalize_words  # noqa: E402,F401
from make_masked_text import MAX_LABEL_WORDS, load_labels  # noqa: E402

WORD_RE = re.compile(r"[0-9a-zA-ZÀ-ɏ]+")  # same class as normalize_words


def tokenize_with_spans(raw: str) -> tuple[list[str], list[tuple[int, int]]]:
    """Token list identical to `normalize_words(raw)`, plus each token's (start, end) in `raw`."""
    ms = list(WORD_RE.finditer(raw))
    return [m.group().lower() for m in ms], [(m.start(), m.end()) for m in ms]


def train_partners(train_path: Path) -> dict[int, set[int]]:
    """Partner set per entity from TRAIN triples, both directions."""
    d: dict[int, set[int]] = defaultdict(set)
    with train_path.open(encoding="utf-8") as f:
        for l in f:
            v = l.split()
            if len(v) < 3:
                continue
            h, t = int(v[0]), int(v[2])
            d[h].add(t)
            d[t].add(h)
    return d


def find_matches(
    words: list[str], lookup: dict[tuple, list[int]], max_len: int, allowed: set[int]
) -> list[tuple[int, int]]:
    """Contiguous token spans [i, i+n) whose tuple is a label of an ALLOWED partner.

    Longest n first at each position, so a long name is never cut in half by a short one.
    """
    out, i = [], 0
    while i < len(words):
        for n in range(min(max_len, len(words) - i), 0, -1):
            g = lookup.get(tuple(words[i : i + n]))
            if g and any(e in allowed for e in g):
                out.append((i, i + n))
                i += n
                break
        else:
            i += 1
    return out


def remove_partner_names(
    raw: str, allowed: set[int], lookup: dict[tuple, list[int]], max_len: int
) -> tuple[str, int]:
    """Delete partner-name spans from `raw`. Returns (masked text, #tokens removed)."""
    words, offsets = tokenize_with_spans(raw)
    spans = find_matches(words, lookup, max_len, allowed)
    if not spans:
        return raw, 0
    kept, prev, n_words = [], 0, 0
    for a, b in spans:
        kept.append(raw[prev : offsets[a][0]])
        prev = offsets[b - 1][1]
        n_words += b - a
    kept.append(raw[prev:])
    return re.sub(r"\s+", " ", " ".join(kept)).strip(), n_words


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", "--bo", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--textual", required=True, help="<dataset>-textual.json (MyGO wordpiece ids)")
    ap.add_argument("--encoder", required=True, help="bert-base-uncased directory")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--t_ctx",
        default="",
        help="Existing T_ctx.pt. When given (recommended), an UNMASKED build is produced first "
        "and must match it bit-for-bit before the masked tensor is written.",
    )
    ap.add_argument(
        "--skip_identity_check",
        "--bo_qua_kiem",
        action="store_true",
        help="Skip that identity check. Only for a set that has no T_ctx.pt yet.",
    )
    a = ap.parse_args()

    import torch

    import build_text_context as SC

    d = Path(a.data_dir)
    labels, e2i = load_labels(a.dataset)
    N = sum(1 for _ in (d / "entities.txt").open(encoding="utf-8"))
    lookup: dict[tuple, list[int]] = defaultdict(list)
    for i_, v in labels.items():
        if v and len(v) <= MAX_LABEL_WORDS:
            lookup[tuple(v)].append(i_)
    max_len = max((len(k) for k in lookup), default=1)
    partners = train_partners(d / "train.txt")
    print(
        f"[{a.dataset}] {N} entities - {sum(len(v) for v in partners.values()) // 2} train edges - "
        f"{len(lookup)} distinct labels - longest label {max_len} words"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model = SC.load_encoder(a.encoder, device)

    # ---- identity gate: the UNMASKED build must match the existing T_ctx.pt bit-for-bit ----
    if a.t_ctx and not a.skip_identity_check:
        T0, _ = SC.build_text_context(
            a.dataset, str(d), a.textual, tok, model, mask_spans=None, device=device
        )
        ref = torch.load(a.t_ctx, map_location="cpu", weights_only=False)
        if T0.shape != ref.shape or not torch.equal(T0, ref):
            n_diff = int((T0 != ref).any(dim=1).sum()) if T0.shape == ref.shape else -1
            # Report magnitudes, not only the number of differing rows: a row count cannot tell
            # kernel arithmetic noise (TF32/SDPA on another GPU) from a genuinely different
            # encoder or formula. BERT in eval() has no RNG, so it is not initialisation.
            detail = ""
            if T0.shape == ref.shape:
                import torch.nn.functional as _F

                d = (T0.float() - ref.float()).abs()
                cos = _F.cosine_similarity(T0.float(), ref.float(), dim=1)
                detail = (
                    f"   |delta| max {d.max():.3e} - median {d.median():.3e} - mean {d.mean():.3e}\n"
                    f"   |T| reference: median {ref.float().abs().median():.4f} - "
                    f"max {ref.float().abs().max():.4f}\n"
                    f"   per-row cosine: min {cos.min():.6f} - median {cos.median():.6f} - "
                    f"rows < 0.999: {int((cos < 0.999).sum())}\n"
                    f"   dtype new {T0.dtype} - reference {ref.dtype}\n"
                    f"   TF32 matmul={torch.backends.cuda.matmul.allow_tf32} "
                    f"cudnn={torch.backends.cudnn.allow_tf32}\n"
                )
                try:
                    import transformers as _tf

                    detail += f"   transformers {_tf.__version__} - torch {torch.__version__}\n"
                except Exception:
                    pass
            raise SystemExit(
                f"IDENTITY GATE FAILED: unmasked rebuild does not match {a.t_ctx} bit-for-bit\n"
                f"   shape {tuple(T0.shape)} vs {tuple(ref.shape)} - differing rows {n_diff}\n"
                + detail
                + "   -> the encoder or formula differs. Do NOT save the masked tensor.\n"
                "   -> cosine ~ 1 and |delta| ~ 1e-3 is kernel noise; low cosine means a truly\n"
                "      different instrument. Do not relax the gate: report these three numbers."
            )
        print(f"IDENTITY GATE OK: unmasked build is bit-identical to {a.t_ctx} ({tuple(T0.shape)})")
    elif not a.t_ctx:
        print("WARNING: --t_ctx not given -> the bit-identity gate is SKIPPED (it is required)")

    # ---- masked build ----
    def mask_spans(i_: int, raw: str) -> tuple[str, int]:
        return remove_partner_names(raw, partners.get(i_, set()), lookup, max_len)

    T, st = SC.build_text_context(
        a.dataset, str(d), a.textual, tok, model, mask_spans=mask_spans, device=device
    )
    torch.save(T, a.out)
    print(f"[{a.dataset}] entities with edited description : {st['n_sua']}")
    print(
        f"[{a.dataset}] mean tokens removed              : {st['n_xoa'] / max(1, st['n_sua']):.2f}"
    )
    for i_, g, c in st["vi_du"]:
        print(f"  --- entity {i_}\n      BEFORE: {g}\n      AFTER : {c}")

    # ---- assert: recount mention edges over train pairs on the masked descriptions = 0 ----
    masked_texts = st["che_txt"]
    remaining = 0
    for i_, parts in partners.items():
        c = masked_texts.get(i_)
        if not c:
            continue
        words, _ = tokenize_with_spans(c)
        for p0 in range(len(words)):
            for n in range(1, min(max_len, len(words) - p0) + 1):
                g = lookup.get(tuple(words[p0 : p0 + n]))
                if g and any(e in parts for e in g):
                    remaining += 1
    print(f"[{a.dataset}] mention edges REMAINING over train pairs: {remaining}")
    if remaining:
        raise SystemExit(f"ASSERT FAILED: {remaining} mention edges remain after masking")
    print(f"OK {a.dataset} T_ctx_mask {tuple(T.shape)} -> {a.out}")


if __name__ == "__main__":
    main()
