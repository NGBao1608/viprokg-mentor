#!/usr/bin/env python3
"""Load the ViProKG graph, gate metadata and offline text/visual features for ViProNet.

Reads the MyGO layout under data/final/ViMMKG (entities.txt = numeric ids 0..N-1, and
train/valid/test.txt as numeric (h r t) triples). Gold files are never modified. Gate metadata comes
from image_index.tsv.
"""

from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import torch

from config import DATA_DIR, IMAGE_KINDS, TIER_PRIOR, load_tier_prior


def _read_triples(path: str) -> List[Tuple[int, int, int]]:
    out = []
    with open(path) as f:
        for line in f:
            h, r, t = line.rstrip("\n").split("\t")
            out.append((int(h), int(r), int(t)))
    return out


class ViMMKGData:
    """ViProNet dataset: graph + filter dicts + gate metadata.

    Text (T) / visual (V) features are loaded separately through load_features (they can be dummy
    features for a CPU dry run).
    """

    def __init__(self, data_dir: str = DATA_DIR) -> None:
        self.dir = data_dir
        with open(os.path.join(data_dir, "entities.txt")) as f:
            self.num_ent = sum(1 for _ in f)
        with open(os.path.join(data_dir, "relations.txt")) as f:
            self.num_rel = sum(1 for _ in f)

        self.train = _read_triples(os.path.join(data_dir, "train.txt"))
        self.valid = _read_triples(os.path.join(data_dir, "valid.txt"))
        self.test = _read_triples(os.path.join(data_dir, "test.txt"))

        # filter dicts for standard filtered ranking (drop the other true answers)
        self.filter_head: Dict[Tuple[int, int], set] = defaultdict(set)  # (r,t) -> {true h}
        self.filter_tail: Dict[Tuple[int, int], set] = defaultdict(set)  # (h,r) -> {true t}
        for h, r, t in self.train + self.valid + self.test:
            self.filter_head[(r, t)].add(h)
            self.filter_tail[(h, r)].add(t)

        self._load_gate_meta()

    def _load_gate_meta(self) -> None:
        """Per-entity gate features: has_image, prior_tier, onehot(kind), log1p(n_share),
        default_view. Indexed by numeric id (= column 'id' of image_index.tsv)."""
        N = self.num_ent
        self.has_image = torch.zeros(N, dtype=torch.bool)
        self.prior_tier = torch.zeros(N, dtype=torch.float)  # 0.97/0.66/0.73, 0 if no image
        self.tier = torch.zeros(N, dtype=torch.long)  # 1/2/3, 0 if no image
        self.kind_onehot = torch.zeros(N, len(IMAGE_KINDS), dtype=torch.float)
        self.log_nshare = torch.zeros(N, dtype=torch.float)
        self.default_view = torch.zeros(N, dtype=torch.bool)
        kind_idx = {k: i for i, k in enumerate(IMAGE_KINDS)}

        # prefer the 13-cell `prior_tang.tsv`; otherwise fall back to the hard-coded TIER_PRIOR
        prior_path = os.path.join(self.dir, "prior_tang.tsv")
        tier_prior_table = load_tier_prior(prior_path) if os.path.exists(prior_path) else None
        unique_label = tier_prior_table.pop("_unique_label", {}) if tier_prior_table else {}
        print(
            f"[gate] prior = "
            f"{'prior_tang.tsv (%d cells)' % len(tier_prior_table) if tier_prior_table else 'hard-coded TIER_PRIOR (3 cells)'}",
            flush=True,
        )

        path = os.path.join(self.dir, "image_index.tsv")
        with open(path) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if r.get("has_image") != "1":
                    continue
                i = int(r["id"])
                if i >= N:
                    continue
                tier_raw = (r.get("tier") or "").strip()
                tier = int(tier_raw) if tier_raw in ("1", "2", "3") else 0
                self.has_image[i] = True
                self.tier[i] = tier
                if tier_prior_table is not None:
                    scored = (r.get("da_cham") or "0").strip()
                    key = (tier_raw, unique_label.get(tier_raw, scored) if scored != "0" else "0")
                    self.prior_tier[i] = tier_prior_table.get(key, 0.0)
                else:
                    self.prior_tier[i] = TIER_PRIOR.get(tier, 0.0)
                k = r.get("image_kind", "")
                if k in kind_idx:
                    self.kind_onehot[i, kind_idx[k]] = 1.0
                self.log_nshare[i] = math.log1p(int(r.get("n_share_file", "0") or 0))
                self.default_view[i] = r.get("in_default_view") == "1"

    def load_features(self, t_path: str, v_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load T [N x 768] (PhoBERT) and V (CLIP or BEiT-v2 tokens) cached offline.

        Returns (T, V). Entities missing a modality are zero vectors (handled at encode time).
        Shapes are asserted.
        """
        T = torch.load(t_path, map_location="cpu")
        V = torch.load(v_path, map_location="cpu")
        if V.dim() == 3:  # [N, k, d]: several images per entity -> masked mean -> [N, d]
            V = self._merge_multi_image(V, v_path)
        assert T.shape[0] == self.num_ent, f"T shape {T.shape} != N={self.num_ent}"
        assert V.shape[0] == self.num_ent, f"V shape {V.shape} != N={self.num_ent}"
        return T.float(), V.float()

    def _merge_multi_image(self, V: torch.Tensor, v_path: str) -> torch.Tensor:
        """Merge [N, k, d] into [N, d] by masked mean; the model is unchanged.

        The mask is read from `<v_path stem>_mask.pt` (MyGO convention: True = NO image in that
        slot). An entity whose mask is all True becomes the zero vector, matching how entities
        without an image are represented in `load_features`. Without a mask: plain mean.
        """
        m_path = v_path[:-3] + "_mask.pt" if v_path.endswith(".pt") else v_path + "_mask.pt"
        if os.path.exists(m_path):
            M = torch.load(m_path, map_location="cpu").bool()  # True = missing
            assert M.shape == V.shape[:2], f"mask {tuple(M.shape)} != V {tuple(V.shape[:2])}"
            keep = (~M).float().unsqueeze(-1)  # [N,k,1]
            n = keep.sum(1).clamp(min=1.0)  # [N,1]
            out = (V.float() * keep).sum(1) / n  # [N,d]
            out[(~M).sum(1) == 0] = 0.0  # entity without image -> 0
            n_with_image = int(((~M).any(1)).sum())
        else:
            out = V.float().mean(1)
            n_with_image = V.shape[0]
        print(
            f"[features] merged V {tuple(V.shape)} -> {tuple(out.shape)} "
            f"(masked mean, {n_with_image}/{V.shape[0]} entities have images; "
            f"mask={os.path.basename(m_path)})",
            flush=True,
        )
        return out

    def has_text_mask(self, T: torch.Tensor) -> torch.Tensor:
        """Entities with text = non-zero T vector (used by L_align)."""
        return T.abs().sum(dim=1) > 0
