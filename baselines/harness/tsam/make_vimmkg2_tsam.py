"""Build the TSAM datasets for ViProKG with numeric ids (avoids space-split problems).

data/ViMMKG2/: entities/relations/entity2id (space)/relation2id/train/valid/test (TAB), numeric ids.
tokens/ViMMKG2-visual.json  = ViMMKG2-visual.json (codebook ids, keys are id strings).
tokens/ViMMKG2-textual.json = ViMMKG2-textual-phobert-seg.json (PhoBERT ids, keys are id strings).
tokens/visual.pth  = BEiT codebook [8192,32] + 1 pad row -> [8193,32] (pad index 8192).
tokens/textual.pth = PhoBERT table [64001,768].
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil

import torch


def sha16(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--ext", required=True, help="data/experimental/v3/ext/ViMMKG2")
    p.add_argument("--tokens_v2", required=True, help="directory containing ViMMKG2-visual.json")
    p.add_argument("--codebook", required=True, help="BEiT codebook visual.pth [8192,32]")
    p.add_argument("--phobert", required=True, help="PhoBERT table textual-phobert.pth [64001,768]")
    a = p.parse_args()
    data_dir = os.path.join(a.repo, "data", "ViMMKG2")
    tokens_dir = os.path.join(a.repo, "tokens")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(tokens_dir, exist_ok=True)

    def count_rows(path):  # 'name<TAB>id[...]' with header -> row count
        n = 0
        with open(path) as f:
            next(f)
            for _ in f:
                n += 1
        return n

    n_ent = count_rows(f"{a.ext}/entity2id.tsv")
    n_rel = count_rows(f"{a.ext}/relation2id.tsv")

    with open(f"{data_dir}/entities.txt", "w") as f:
        f.write("\n".join(str(i) for i in range(n_ent)) + "\n")
    with open(f"{data_dir}/relations.txt", "w") as f:
        f.write("\n".join(str(j) for j in range(n_rel)) + "\n")
    with open(f"{data_dir}/entity2id.txt", "w") as f:
        for i in range(n_ent):
            f.write(f"{i} {i}\n")
    with open(f"{data_dir}/relation2id.txt", "w") as f:
        for j in range(n_rel):
            f.write(f"{j} {j}\n")
    for split in ("train", "valid", "test"):
        with open(f"{a.ext}/{split}.txt") as fi, open(f"{data_dir}/{split}.txt", "w") as fo:
            for line in fi:
                c = line.split()
                if len(c) >= 3:
                    fo.write(f"{c[0]}\t{c[1]}\t{c[2]}\n")

    shutil.copy(f"{a.tokens_v2}/ViMMKG2-visual.json", f"{tokens_dir}/ViMMKG2-visual.json")
    shutil.copy(f"{a.ext}/ViMMKG2-textual-phobert-seg.json", f"{tokens_dir}/ViMMKG2-textual.json")

    cb = torch.load(a.codebook, map_location="cpu").float()  # [8192,32]
    assert cb.shape[1] == 32, f"codebook dim {cb.shape} != 32"
    cb = torch.cat([cb, torch.zeros(1, cb.shape[1])], 0)  # +pad row → [8193,32]
    torch.save(cb, f"{tokens_dir}/visual.pth")
    pt = torch.load(a.phobert, map_location="cpu").float()  # [64001,768]
    assert pt.shape == (64001, 768), f"phobert {pt.shape} != [64001,768]"
    torch.save(pt, f"{tokens_dir}/textual.pth")

    json_sha = sha16(f"{a.ext}/ViMMKG2-textual-phobert-seg.json")
    assert json_sha == "479405e3a71e82f0", f"sha json {json_sha} != 479405e3a71e82f0"
    print(f"[tsam-v2] data/ViMMKG2 (ent={n_ent} rel={n_rel}) + tokens")
    print(
        f'[sha] visual.json={sha16(tokens_dir+"/ViMMKG2-visual.json")} textual.json(seg)={json_sha} '
        f'visual.pth={sha16(tokens_dir+"/visual.pth")} textual.pth={sha16(tokens_dir+"/textual.pth")}'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
