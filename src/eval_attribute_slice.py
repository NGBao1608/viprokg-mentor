"""Visual attribute-slice evaluation: predict the tail restricted to the value set of each relation.

Unlike the gold evaluation, candidates are not all N entities but only the k value nodes of the
relation (k = 4-7), so this is a choose-1-of-k task that can be compared directly with the random
(1/k) and majority baselines. The head-prediction direction is not measured.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch


def load_v2_package(
    v2_dir: str,
) -> Tuple[List[Tuple[int, int, int]], Dict[int, List[int]], Dict[int, str]]:
    """Read the numeric v2 package: test triples (h r t), candidates (r, t1,t2,...), relation names."""
    test = []
    with open(os.path.join(v2_dir, "v2_test_ids.txt")) as f:
        for line in f:
            a = line.split()
            if len(a) == 3:
                test.append((int(a[0]), int(a[1]), int(a[2])))
    cand: Dict[int, List[int]] = {}
    with open(os.path.join(v2_dir, "v2_cand.tsv")) as f:
        next(f)
        for line in f:
            r, ts = line.rstrip("\n").split("\t")
            cand[int(r)] = [int(x) for x in ts.split(",")]
    names: Dict[int, str] = {}
    with open(os.path.join(v2_dir, "v2_rel_names.tsv")) as f:
        next(f)
        for line in f:
            r, name = line.rstrip("\n").split("\t")
            names[int(r)] = name
    return test, cand, names


@torch.no_grad()
def eval_v2(
    model,
    T: torch.Tensor,
    V: torch.Tensor,
    v2_dir: str,
    use_text: bool = True,
    use_image: bool = True,
    H: Optional[torch.Tensor] = None,
    Hmask: Optional[torch.Tensor] = None,
    T2: Optional[torch.Tensor] = None,
    dump: str = "",
) -> Dict[str, object]:
    """Return {'overall': {...}, 'theo_quan_he': {name: {...}}} with Hits@1 and MRR over k candidates."""
    model.eval()
    dev = next(model.parameters()).device
    test, cand, names = load_v2_package(v2_dir)
    enc = model.encode(
        T.to(dev),
        V.to(dev),
        H=None if H is None else H.to(dev),
        Hmask=None if Hmask is None else Hmask.to(dev),
        use_text=use_text,
        use_image=use_image,
        T2=None if T2 is None else T2.to(dev),
    )
    ent_all = enc["h_e"]

    by_rel: Dict[int, List[int]] = defaultdict(list)  # r -> list of ranks
    detail = []
    for h, r, t in test:
        ids = cand[r]
        cols = torch.tensor(ids, device=dev)
        # score exactly the k candidates (not all N): choose 1 of k
        s = model.tucker.score_tail(
            ent_all[h].unsqueeze(0), torch.tensor([r], device=dev), ent_all[cols]
        )[
            0
        ]  # [k]
        vi = ids.index(t)
        gold = s[vi].item()
        rank = int((s > gold).sum().item()) + (int((s == gold).sum().item()) + 1) // 2
        by_rel[r].append(rank)
        detail.append((h, r, t, rank, len(ids)))

    def aggregate(hs: List[int]) -> Dict[str, float]:
        x = torch.tensor(hs, dtype=torch.float)
        return {
            "n": len(hs),
            "Hits@1": float((x <= 1).float().mean()),
            "MRR": float((1.0 / x).mean()),
        }

    all_ranks = [v for hs in by_rel.values() for v in hs]
    out = {
        "overall": aggregate(all_ranks),
        "theo_quan_he": {
            names.get(r, str(r)): aggregate(hs)
            for r, hs in sorted(by_rel.items(), key=lambda kv: names.get(kv[0], ""))
        },
    }
    if dump:
        os.makedirs(os.path.dirname(dump) or ".", exist_ok=True)
        with open(dump, "w") as f:
            f.write("h_id\tr_id\tt_id\thang\tk\n")
            for row in detail:
                f.write("\t".join(map(str, row)) + "\n")
    return out


def print_summary(result: Dict[str, object], label: str = "") -> None:
    """Compact print for the Kaggle log."""
    o = result["overall"]
    print(
        f"  [v2{(' ' + label) if label else ''}] n={o['n']}  Hits@1 {o['Hits@1']:.4f}  MRR {o['MRR']:.4f}",
        flush=True,
    )
    for name, m in result["theo_quan_he"].items():
        print(
            f"      {name:26} n={m['n']:3}  H@1 {m['Hits@1']:.4f}  MRR {m['MRR']:.4f}", flush=True
        )
