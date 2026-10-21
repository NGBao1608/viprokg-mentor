"""Out-of-fold gate mixing the structure score S and the lexical score L (no GPU needed).

Design:
  input  = 6 query-level features + emb(r)
           [margin_L, margin_S, query_has_mention, log_deg, direction, has_lexical_candidates]
  model  = 2-layer MLP -> softmax over {g_S, g_L}
  score  = g_S * z(S) + g_L * z(L), z = per-query z-score (same as TuckerLexicalFusion._z)
  loss   = cross-entropy on the gold, masking every other true answer in the full train set
           (including the held-out fold)

No feature reads the gold. `query_has_mention` is a query-level feature: 1 if the entity on the
query side has at least one mention edge (derived from text) to any entity. The fold-wise
selection criterion uses the pair-level indicator (h, t) of the triple instead; the two must
never be mixed up (the group sizes differ by a large factor).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def per_query_zscore(x: torch.Tensor) -> torch.Tensor:
    """Row-wise z-score ignoring -inf cells. Same as `TuckerLexicalFusion._z`."""
    finite = torch.isfinite(x)
    m = torch.where(finite, x, torch.zeros_like(x)).sum(1, keepdim=True) / finite.sum(
        1, keepdim=True
    )
    v = torch.where(finite, (x - m) ** 2, torch.zeros_like(x)).sum(1, keepdim=True)
    sd = (v / finite.sum(1, keepdim=True)).sqrt().clamp(min=1e-6)
    return torch.where(finite, (x - m) / sd, x)


def gate_features(
    S: torch.Tensor,
    L: torch.Tensor,
    source: torch.Tensor,
    query_has_mention: torch.Tensor,
    deg: torch.Tensor,
    direction: torch.Tensor,
    has_lexical_candidates: torch.Tensor,
) -> torch.Tensor:
    """Six query-level features, [B, 6]. The gold is deliberately NOT an argument.

    `margin(x)` = top1 - top2 of the score matrix itself (how confident the model is); it uses
    no label. `query_has_mention` is query-level, not the pair-level selection criterion.
    """

    def margin(x: torch.Tensor) -> torch.Tensor:
        t2 = torch.topk(torch.nan_to_num(x, neginf=-1e4), 2, dim=1).values
        return (t2[:, 0] - t2[:, 1]).clamp(min=0)

    return torch.stack(
        [
            margin(L),
            margin(S),
            query_has_mention,
            torch.log1p(deg[source]),
            direction,
            has_lexical_candidates,
        ],
        dim=1,
    )


class Gate(nn.Module):
    """2-layer MLP -> softmax {g_S, g_L}. `dim_r` is 16 or 64 (fixed grid)."""

    def __init__(self, num_rel: int, hidden: int = 32, dim_r: int = 16) -> None:
        super().__init__()
        self.Er = nn.Embedding(num_rel * 2, dim_r)
        self.mlp = nn.Sequential(nn.Linear(6 + dim_r, hidden), nn.ReLU(), nn.Linear(hidden, 2))
        nn.init.xavier_normal_(self.Er.weight)

    def forward(self, feats: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """`feats` [B,6], `r` [B] -> `g` [B,2]."""
        return F.softmax(self.mlp(torch.cat([feats, self.Er(r)], dim=-1)), dim=-1)


def mix_scores(g: torch.Tensor, S: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Gate-weighted mixture after per-query z-scoring."""
    return g[:, :1] * per_query_zscore(S) + g[:, 1:2] * per_query_zscore(L)


def train_gate(gate: nn.Module, batches, lr: float, num_epochs: int = 30) -> nn.Module:
    """Cross-entropy on the gold; `batches` yields `(feats, r, S, L, gold, mask_other)`.

    `mask_other` [B,N] bool marks every other true answer in the full train set (including the
    held-out fold); those cells are set to -inf before the loss, like the validation filter.
    """
    opt = torch.optim.Adam(gate.parameters(), lr=lr)
    for _ in range(num_epochs):
        for feats, r, S, L, gold, mask_other in batches:
            g = gate(feats, r)
            sc = mix_scores(g, S, L)
            sc = sc.masked_fill(mask_other, float("-inf"))
            opt.zero_grad()
            F.cross_entropy(sc, gold).backward()
            opt.step()
    return gate
