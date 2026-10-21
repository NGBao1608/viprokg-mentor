"""TSAM v2 round-trip gate: the merged textual tokens for v2 must neither leak nor contain garbage.

Runs get_entity_textual_tokens('ViMMKG2', max) with the patched merge_tokens, then asserts:
  (G1) every merged text id (padding removed) of an entity is in that entity's segmented json
       id set (0 garbage),
  (G2) the maximum merged id is <= 64000,
  (G3) the number of entities with text equals the number of entities in the segmented json (8207),
  (G4) with the pre-seeded 64001 vocabulary, position == id, so the merged id is the real
       PhoBERT id (no shift).
A failure returns rc=1, in which case v2 is dropped.
"""

from __future__ import annotations

import json
import os
import sys

MAX_TXT = 24  # as in the MKG-W run


def main() -> int:
    repo = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else "."
    os.chdir(repo)
    sys.path.insert(0, repo)
    import torch

    torch.Tensor.cuda = lambda self, *a, **k: self  # type: ignore
    torch.cuda.is_available = lambda: False  # type: ignore
    from merge_tokens import get_entity_textual_tokens, load_ent_map

    seg = json.load(open("tokens/ViMMKG2-textual.json"))
    ent_map = load_ent_map("ViMMKG2")  # name (id string) -> int id
    # json id set by numeric entity id
    seg_by_id = {ent_map[k]: set(v) for k, v in seg.items() if k in ent_map}

    idx, mask = get_entity_textual_tokens("ViMMKG2", MAX_TXT)
    idx = idx.numpy()
    mask = mask.numpy()

    n_text = 0
    garbage = 0
    maxid = 0
    n_checked = 0
    for i in range(idx.shape[0]):
        toks = [int(idx[i][j]) for j in range(MAX_TXT) if not mask[i][j]]  # drop padding by mask
        if not toks:
            continue
        n_text += 1
        json_ids = seg_by_id.get(i, set())
        for tok in toks:
            maxid = max(maxid, tok)
            if tok not in json_ids:  # garbage: merged id not in the entity's segmented json
                garbage += 1
        n_checked += len(toks)
    n_json = len([k for k in seg if k in ent_map])

    print(f"[circle] entities with text (merged) = {n_text} | segmented json = {n_json}")
    print(f"[circle] max id merge = {maxid} (≤64000? {maxid <= 64000})")
    print(f"[circle] garbage tokens (id not in the entity json) = {garbage}/{n_checked}")
    ok = (n_text == n_json) and (maxid <= 64000) and (garbage == 0)
    print(("ROUND-TRIP GATE PASSED" if ok else "ROUND-TRIP GATE FAILED: drop v2"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
