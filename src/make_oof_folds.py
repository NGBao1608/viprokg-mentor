"""Split the training triples into K=3 folds (by triple). No GPU.

Writes `<ext>/<dataset>/folds_k3.json` = {"k": 3, "seed": 42, "n": n, "fold": [...]}
where `fold[i]` is the fold id (0..K-1) of the i-th triple in `train.txt`.

Checks: every triple belongs to exactly one fold, and the fold sizes sum to the number of
training triples.
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inspect_validation_dump as M  # noqa: E402

DATASETS = ["MKG-W", "MKG-Y", "DB15K"]
K, SEED = 3, 42


def main() -> None:
    for dataset in DATASETS:
        d = os.path.join(M.EXT_DIR, dataset)
        tr = M.load_triples_text(os.path.join(d, "train.txt"))
        n = len(tr)
        idx = list(range(n))
        random.Random(SEED).shuffle(idx)
        fold = [0] * n
        for rank, i in enumerate(idx):
            fold[i] = rank % K
        form_counts = Counter(fold)
        assert sum(form_counts.values()) == n, "fold sizes do not sum to the number of triples"
        assert set(form_counts) == set(range(K)), "missing fold"
        p = os.path.join(d, f"folds_k{K}.json")
        json.dump({"k": K, "seed": SEED, "n": n, "fold": fold}, open(p, "w", encoding="utf-8"))
        print(
            f"OK {dataset:7} n={n:6} · fold "
            + " · ".join(f"k{k}={form_counts[k]}" for k in range(K))
            + f" · each triple in exactly 1 fold -> {p}"
        )


if __name__ == "__main__":
    sys.exit(main())
