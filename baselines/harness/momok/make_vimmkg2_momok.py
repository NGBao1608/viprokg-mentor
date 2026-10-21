"""Build datasets/ViMMKG2/ in MoMoK format from ext/ViMMKG2.

- entity2id.txt / relation2id.txt: 'name id' without header; the id keeps the ext order so that
  features stay aligned.
- train/valid/test.txt: ids are written as-is ('h r t .', single spaces, as MoMoK expects).
  The split is kept unchanged, not re-split.
- img_features.pth = V.pt reshaped [N,1,32] -> [N,32] (must be 2D so that F.normalize(dim=1)
  normalises over the 32 dims).
- text_features.pth = the contextual text matrix T_ctx.pt [N,768] (not a mean-pool of the
  PhoBERT input embeddings, which would be a different instrument).
"""

from __future__ import annotations

import argparse
import hashlib
import os

import torch


def read_two_col_tsv(path: str) -> dict[str, int]:
    """Read a tsv 'name<TAB>id[<TAB>...]' with header -> {name: numeric id}."""
    d = {}
    with open(path) as f:
        next(f)  # skip header
        for line in f:
            c = line.rstrip("\n").split("\t")
            d[c[0]] = int(c[1])
    return d


def sha16(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ext", required=True, help="data/experimental/v3/ext/ViMMKG2")
    p.add_argument("--t_ctx", default=None, help="T_ctx.pt v2 [8255,768] (MENTOR path)")
    p.add_argument("--out", required=True, help="datasets/ViMMKG2 directory to create")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ent = read_two_col_tsv(os.path.join(a.ext, "entity2id.tsv"))  # QID -> id
    rel = read_two_col_tsv(os.path.join(a.ext, "relation2id.tsv"))  # PID -> id
    n_ent = max(ent.values()) + 1
    assert len(ent) == n_ent, f"entity ids are not contiguous: {len(ent)} vs {n_ent}"

    # v2 entity names can contain spaces (slice nodes VAL::...), which would break MoMoK's
    # space-split parser; the numeric id is therefore used as the "name" (space-safe, keeps the
    # id order so features stay aligned). Map ids back via ext/{entity,relation}2id.tsv.
    n_rel = max(rel.values()) + 1
    with open(os.path.join(a.out, "entity2id.txt"), "w") as f:
        for i in range(n_ent):
            f.write(f"{i} {i}\n")
    with open(os.path.join(a.out, "relation2id.txt"), "w") as f:
        for i in range(n_rel):
            f.write(f"{i} {i}\n")

    def convert(split: str) -> int:
        src = os.path.join(a.ext, f"{split}.txt")
        n = 0
        with open(src) as fi, open(os.path.join(a.out, f"{split}.txt"), "w") as fo:
            for line in fi:
                c = line.split()
                if len(c) < 3:
                    continue
                h, r, t = int(c[0]), int(c[1]), int(c[2])
                fo.write(f"{h} {r} {t} .\n")  # numeric ids, single space (MoMoK splits on ' ')
                n += 1
        return n

    counts = {s: convert(s) for s in ("train", "valid", "test")}
    print(f"[triple] {counts}")

    # --- img_features: V.pt; the dtype is checked and an integer dtype stops the run ---
    V = torch.load(os.path.join(a.ext, "V.pt"), map_location="cpu")
    if not torch.is_floating_point(V):
        raise SystemExit(
            f"V.pt dtype={V.dtype} is INTEGER: the image feature file of the A3' v2 run "
            f"(grid c) must be used instead. STOP."
        )
    img = V.float().reshape(V.shape[0], -1).contiguous()
    assert img.shape[0] == n_ent, f"V.pt {img.shape} != n_ent {n_ent}"
    torch.save(img, os.path.join(a.out, "img_features.pth"))

    # --- text_features: use T_ctx.pt (the MENTOR path), NOT a mean-pool ---
    # A mean-pool of the PhoBERT input embeddings would be a different instrument from MENTOR
    # (T_ctx is the masked mean of PhoBERT last_hidden_state, i.e. contextual).
    # Without T_ctx the run stops.
    if not a.t_ctx:
        raise SystemExit(
            "missing --t_ctx: T_ctx.pt for v2 is required (only W/Y/DB15K have one). "
            "Not running without it; provide a v2 T_ctx or the encoder."
        )
    T = torch.load(a.t_ctx, map_location="cpu").float()
    assert T.shape == (n_ent, 768), f"T_ctx shape {tuple(T.shape)} != [{n_ent},768]"
    torch.save(T.contiguous(), os.path.join(a.out, "text_features.pth"))

    # provenance: the json sha must go into the run's hash_dau_vao
    json_sha = sha16(os.path.join(a.ext, "ViMMKG2-textual-phobert-seg.json"))
    assert json_sha == "479405e3a71e82f0", f"json sha {json_sha} != 479405e3a71e82f0"
    print(f"[feat] img={tuple(img.shape)} ({V.dtype}) text={tuple(T.shape)} (from T_ctx)")
    print(
        "[sha] img_features=%s text_features=%s"
        % (
            sha16(os.path.join(a.out, "img_features.pth")),
            sha16(os.path.join(a.out, "text_features.pth")),
        )
    )
    print(
        "[sha sources] V.pt=%s T_ctx=%s json=%s"
        % (sha16(os.path.join(a.ext, "V.pt")), sha16(a.t_ctx), json_sha)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
