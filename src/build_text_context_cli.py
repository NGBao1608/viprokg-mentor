#!/usr/bin/env python3
"""CLI wrapper over `build_text_context.build_text_context(mask_spans=None)`: one code path for
every T_ctx tensor.

Why it exists: `T_ctx.pt` shipped inside the datasets was encoded in an older GPU image.
Re-encoding the same text in a newer image (different torch / GPU kernels) does not reproduce it
bit-for-bit (14 150 / 15 000 rows differ on MKG-W). BERT runs in `eval()` with no RNG, so the
difference is kernel-level arithmetic, not initialisation.

The danger is not the difference itself but mixing the two. Scoring the clean pass with the old
tensor and the masked pass with a new one would measure masking plus environment, not masking.
So build every text tensor with this script, in one session, from the same encoder. The only
thing that may differ between two tensors is the textual json fed in:

    original json  -> T_ctx_new.pt        (clean, this environment)
    masked json    -> T_ctx_mask_new.pt   (answer names removed)

`build_text_context_masked.py` keeps its own role (it masks raw text itself) but its unmasked
counterpart must come from here, not from the shipped file.

Not affected by any of this (checked by reading the code):
  * MyGO consumes `<dataset>-textual.json` wordpiece ids, never `T_ctx.pt`.
  * the lexical features only call `AutoTokenizer` (vocabulary lookup, no model forward), so
    BM25 / lexical features are deterministic across images.

Usage:
  python3 build_text_context_cli.py --dataset MKG-W --data_dir <d> --textual <json> \
      --encoder <d> --out <pt> [--compare_to <T_ctx.pt>]   # prints |delta| vs a reference tensor
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", "--bo", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--textual", required=True, help="<dataset>-textual.json (original or masked)")
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument(
        "--compare_to",
        "--so_sanh",
        default="",
        help="reference .pt to diff against; prints only, never exits non-zero",
    )
    a = ap.parse_args()

    import torch

    import build_text_context as SC

    tok, model = SC.load_encoder(a.encoder, "cuda" if torch.cuda.is_available() else "cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T, _st = SC.build_text_context(
        a.dataset, a.data_dir, a.textual, tok, model, mask_spans=None, batch=a.batch, device=device
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(T, a.out)
    sha = hashlib.sha256(T.contiguous().cpu().numpy().tobytes()).hexdigest()[:16]
    print(f"[T_ctx] {a.dataset} -> {a.out} shape {tuple(T.shape)} dtype {T.dtype} sha256-16 {sha}")
    # Identify the input (segment json) by hash, not by path.
    sha_in = hashlib.sha256(Path(a.textual).read_bytes()).hexdigest()[:16]
    print(f"[T_ctx] textual = {a.textual} - input hash {sha_in}")
    # Zero-row gate: an entity without a description encodes to a zero vector. If ALL rows are
    # zero the key mapping is wrong, so stop; the zero-row count is always reported separately.
    n_zero = int((T.abs().sum(1) == 0).sum())
    n_with_text = int(T.shape[0] - n_zero)
    print(
        f"[T_ctx] rows with text {n_with_text}/{T.shape[0]} - zero rows {n_zero} "
        "(entities without description)"
    )
    assert n_with_text > 0, (
        f"T_ctx is all zeros ({a.dataset}): wrong key mapping (ViMMKG2 must use integer keys, "
        "not e2i/QID). Stop; do not save."
    )

    if a.compare_to and Path(a.compare_to).exists():
        # Informational only: this number goes into Limitations, not into a gate.
        ref = torch.load(a.compare_to, map_location="cpu", weights_only=False)
        if ref.shape != T.shape:
            print(f"[diff] shape differs {tuple(T.shape)} vs {tuple(ref.shape)} - skip")
            return 0
        import torch.nn.functional as F

        A, B = T.float().cpu(), ref.float()
        d = (A - B).abs()
        n_row = int((T.cpu() != ref).any(dim=1).sum())
        # Entities with no description encode to an all-zero row. `cosine_similarity` of a zero
        # vector is 0, so including them makes `cosine min` read 0.000000 and inflates
        # `rows<0.999` even when the two tensors are bit-identical. Report zero rows separately.
        both_nonzero = (A.abs().sum(1) > 0) & (B.abs().sum(1) > 0)
        n_zero = int((~both_nonzero).sum())
        cos = (
            F.cosine_similarity(A[both_nonzero], B[both_nonzero], dim=1)
            if int(both_nonzero.sum())
            else A.new_zeros(0)
        )
        print(f"[diff] vs {a.compare_to}")
        print(f"[diff]   rows differing {n_row}/{len(ref)}")
        print(f"[diff]   |delta| max {d.max():.3e} median {d.median():.3e} mean {d.mean():.3e}")
        print(f"[diff]   |ref| median {B.abs().median():.4f} max {B.abs().max():.4f}")
        if cos.numel():
            print(
                f"[diff]   cosine over {int(both_nonzero.sum())} NON-ZERO rows: min {cos.min():.6f} "
                f"median {cos.median():.6f} rows<0.999 {int((cos < 0.999).sum())}"
            )
        print(
            f"[diff]   all-zero rows excluded from cosine: {n_zero} "
            f"(entities with no description)"
        )
        print(
            f"[diff]   tf32 matmul={torch.backends.cuda.matmul.allow_tf32} "
            f"cudnn={torch.backends.cudnn.allow_tf32}"
        )
        try:
            import transformers

            print(f"[diff]   transformers {transformers.__version__} torch {torch.__version__}")
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
