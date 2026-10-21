"""Scan directions (T1, T2, S1, S2, L1, L2, H1, H2, M1, ...) collected in one place.

`model.py` and `train.py` stay almost unchanged; each direction plugs in through three hooks:
  - model additions: extra modules (S1, S2, H2, M1)
  - loss functions:  replacement training losses (L1, L2)
  - schedule:        replacement learning-rate schedule (H1)
T1/T2 need no code: only `--t_path` changes (T.pt built with a different context encoder).

Per-direction hyper-parameters are the defaults of the original papers, declared in
`VARIANT_DEFAULTS` and written to `config.json`. They are not tuned during the screening round.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

VARIANT_CODES = [
    "T1",
    "T2",
    "S1",
    "S2",
    "L1",
    "L2",
    "H1",
    "H2",
    "H2T",
    "M1",
    "G1",
    "G2",
    "X1",
    "X2",
    "L3",
    "P1",
    "LSR",
]

# defaults taken from the original papers - NOT tuned
VARIANT_DEFAULTS: Dict[str, dict] = {
    "T1": {"nguon": "SimKGC/KG-BERT", "doi": "only T.pt changes (last-layer BERT pooling, mean)"},
    "T2": {"nguon": "retrieval (bge/e5)", "doi": "only T.pt changes"},
    "S1": {
        "nguon": "NodePiece ICLR'22",
        "so_lang_gieng": 8,
        "dim_attn": 256,
        "doi": "emb = alpha(deg)*free + (1-alpha)*attn-pool{(r, neighbour)}",
    },
    "S2": {
        "nguon": "CompGCN ICLR'20",
        "so_lop": 2,
        "comp": "sub",
        "drop": 0.1,
        "doi": "message passing before TuckER",
    },
    "L1": {
        "nguon": "degree-bias WWW'23",
        "mu": 0.5,
        "doi": "CE weight proportional to 1/freq(gold)^mu",
    },
    "L2": {
        "nguon": "SimKGC ACL'22",
        "tau": 0.05,
        "so_hard": 64,
        "smoothing": 0.1,
        "doi": "hard negative in-batch + label smoothing",
    },
    "H1": {
        "nguon": "n/a",
        "num_epoch": 600,
        "lich": "cosine",
        "patience": 50,
        "doi": "only the learning-rate schedule changes",
    },
    "H2": {
        "nguon": "dual-encoder",
        "beta_init": 0.1,
        "doi": "score = TuckER + beta*cos(W(text_h + r), text_t), beta learned",
    },
    "H2T": {
        "nguon": "text-only probe",
        "beta_init": 0.1,
        "doi": "text only: score = beta*cos(W(text_q + r), text_c), no TuckER term",
    },
    "M1": {
        "nguon": "MyGO AAAI'25",
        "so_token": 8,
        "so_lop": 1,
        "so_head": 2,
        "doi": "image token sequence + small transformer replacing 32-d mean pooling",
    },
    "G1": {
        "nguon": "CompGCN + mention edges",
        "so_lop": 2,
        "comp": "sub",
        "drop": 0.1,
        "doi": "like S2 but graph = train + mention edges (new relation r_men)",
    },
    "G2": {
        "nguon": "pseudo-text",
        "doi": "only T.pt changes - text generated from TRAIN for entities without a description",
    },
    "X1": {
        "nguon": "dual-encoder prompt",
        "beta_init": 0.1,
        "dim": 256,
        "doi": "q = MLP([T[source]; emb(r)]) instead of adding Wr (H2)",
    },
    "X2": {
        "nguon": "BM25 (Robertson k1=1.5 b=0.75)",
        "gamma_init": 0.1,
        "top_k": 100,
        "doi": "score += gamma*bm25(text[source], text[c]) on the top-100",
    },
    "L3": {
        "nguon": "ConvE 1-N scoring",
        "doi": "multi-label BCE over ALL true tails of (h,r) in train",
    },
    "LSR": {
        "nguon": "lexical-semantic router v1",
        "dim_L": 64,
        "dim_R": 32,
        "top_k": 100,
        "doi": "s = g_S(q)*S + g_L(q)*L + g_M(q)*M; g initialised ~ (1,0,0); per-query z-score",
    },
    "P1": {
        "nguon": "self-training",
        "so_vong": 2,
        "margin": 1.0,
        "max_moi_cap": 1,
        "doi": "2 rounds; round 2 adds pseudo-triples with deg<=2 filtered by margin",
    },
}


# ---------------------------------------------------------------- S1 NodePiece-lite
class NeighborEncoder(nn.Module):
    """S1 - attention pooling over (relation, neighbour) of an entity, mixed with a free embedding by degree."""

    def __init__(self, num_ent: int, num_rel: int, dim: int, k: int = 8) -> None:
        super().__init__()
        self.k = k
        self.R = nn.Embedding(num_rel * 2 + 1, dim, padding_idx=num_rel * 2)
        self.q = nn.Parameter(torch.randn(dim) * 0.02)
        self.alpha = nn.Parameter(torch.zeros(3))  # alpha per degree bin {<=2, 3-10, >10}
        nn.init.xavier_normal_(self.R.weight)

    def forward(
        self,
        E_s: torch.Tensor,
        nb_ent: torch.Tensor,
        nb_rel: torch.Tensor,
        nb_mask: torch.Tensor,
        degree_bin: torch.Tensor,
    ) -> torch.Tensor:
        """nb_* [N,k]; nb_mask True = empty slot; degree_bin [N] in {0,1,2}."""
        msg = E_s[nb_ent] + self.R(nb_rel)  # [N,k,d]
        w = (msg * self.q).sum(-1).masked_fill(nb_mask, float("-inf"))
        w = torch.softmax(w, dim=1).nan_to_num(0.0).unsqueeze(-1)
        ctx = (w * msg).sum(1)  # [N,d]
        a = torch.sigmoid(self.alpha)[degree_bin].unsqueeze(-1)
        return a * E_s + (1 - a) * ctx


# ---------------------------------------------------------------- S2 CompGCN-lite
class CompGCN(nn.Module):
    """S2 - `sub` message passing (h - r), 1-2 layers, on the train graph with inverse relations."""

    def __init__(self, num_rel: int, dim: int, so_lop: int = 2, drop: float = 0.1) -> None:
        super().__init__()
        self.so_lop = so_lop
        self.R = nn.Embedding(num_rel * 2, dim)
        self.W = nn.ModuleList([nn.Linear(dim, dim) for _ in range(so_lop)])
        self.ln = nn.ModuleList([nn.LayerNorm(dim) for _ in range(so_lop)])
        self.drop = nn.Dropout(drop)
        nn.init.xavier_normal_(self.R.weight)

    def forward(
        self,
        E: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        rel: torch.Tensor,
        deg: torch.Tensor,
    ) -> torch.Tensor:
        x = E
        for i in range(self.so_lop):
            m = x[src] - self.R(rel)  # comp = sub (CompGCN)
            agg = torch.zeros_like(x).index_add_(0, dst, m) / deg.unsqueeze(-1).clamp(min=1)
            x = self.ln[i](x + self.drop(torch.relu(self.W[i](agg))))
        return x


# ---------------------------------------------------------------- M1 MyGO image tokens
class ImageTokenEncoder(nn.Module):
    """M1 - sequence of image tokens (codebook ids) + a small transformer, replacing 32-d mean pooling."""

    def __init__(
        self, num_codes: int, dim: int, k: int = 8, so_lop: int = 1, so_head: int = 2
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(num_codes + 1, dim, padding_idx=num_codes)
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, so_head, dim * 2, 0.1, batch_first=True)
        self.tr = nn.TransformerEncoder(layer, so_lop)
        self.k = k

    def forward(self, tok: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """tok [N,k] token ids (pad = num_codes); mask [N,k] True = empty."""
        x = torch.cat([self.cls.expand(tok.shape[0], -1, -1), self.emb(tok)], dim=1)
        m = torch.cat(
            [torch.zeros(tok.shape[0], 1, dtype=torch.bool, device=tok.device), mask], dim=1
        )
        return self.tr(x, src_key_padding_mask=m)[:, 0]


# ---------------------------------------------------------------- sparse lexical matrix
def compress_sparse_lexical(L: torch.Tensor, max_nonconst: int | None = None) -> dict:
    """Compress the lexical score matrix L LOSSLESSLY (checked by a reconstruction assert).

    `LexicalBlock.diem` returns 0 outside the BM25 top-100, so after z-scoring EVERY cell outside the
    top-100 carries the same constant (measured: at most 100 cells differ from the constant per
    row, 86 on average at N=15 000). Storing full rows wastes about half of the out-of-fold dump size.

    Returns `{"L_idx": int32[Q,k], "L_val": fp16[Q,k], "L_hang": fp16[Q]}`;
    `decompress_sparse_lexical` rebuilds the matrix bit-exactly. Spare slots (rows with fewer than k
    non-constant cells) point at positions whose value equals the constant, so the scatter is a no-op.
    """
    row_const = L.mode(dim=1).values  # per-row constant
    differs = L != row_const.unsqueeze(1)
    k = int(max(1, differs.sum(1).max().item()))
    # "Sparse" is a property relative to N, not an absolute number: the limit is k <= N/8, which
    # still catches a dense L (k ~ N) without blocking candidate unions of ~200 cells.
    if max_nonconst is None:
        max_nonconst = max(128, L.shape[1] // 8)
    assert (
        k <= max_nonconst
    ), f"L has {k} non-constant cells/row > {max_nonconst} (= N/8, N={L.shape[1]}): L is no longer sparse"
    idx = differs.to(torch.float32).topk(k, dim=1).indices.to(torch.int32)
    val = L.gather(1, idx.long())
    res = {"L_idx": idx, "L_val": val, "L_hang": row_const}
    assert torch.equal(
        decompress_sparse_lexical(res, L.shape[1]), L
    ), "sparse L compression does not reconstruct exactly"
    return res


def decompress_sparse_lexical(d: dict, N: int) -> torch.Tensor:
    """Rebuild L [Q,N] from the sparse form of `compress_sparse_lexical` (bit-exact)."""
    L = d["L_hang"].unsqueeze(1).expand(-1, N).clone()
    return L.scatter_(1, d["L_idx"].long(), d["L_val"])


def multi_positive_target(
    s: torch.Tensor, h, r, gold, answers_lookup: dict, sm: float
) -> torch.Tensor:
    """Multi-positive target (KvsAll inside the softmax), [B, N].

    `q = eps/N` uniformly PLUS `(1-eps)/|A|` on `A = N(h, r)` (every true answer of that very query in
    TRAIN, including the gold). With `|A| = 1` the formula coincides bit-exactly with PyTorch's
    `label_smoothing` (`q[gold] = 1-eps+eps/N`).

    This is a configuration change, not a contribution: KvsAll already exists in LibKGE and ConvE
    uses multi-label BCE. Motivation: the single-gold loss scores 56.7% / 68.2% / 65.0% of train
    triples as negatives although they are true answers of the same query.

    `answers_lookup` is built from TRAIN only (fold-filtered if applicable) and looked up by `(h, r)` of the
    query; it never reads the gold of valid/test.
    """
    q = torch.full_like(s, sm / s.shape[1])
    hs, rs = h.tolist(), r.tolist()
    for i in range(len(hs)):
        a = answers_lookup.get((hs[i], rs[i]))
        if not a:
            q[i, gold[i]] += 1.0 - sm
        else:
            q[i, torch.as_tensor(a, device=s.device, dtype=torch.long)] += (1.0 - sm) / len(a)
    return q


def structured_smoothing_target(
    s: torch.Tensor,
    h,
    r,
    gold,
    sm: float,
    che_do: str,
    A_r: dict | None = None,
    NB: dict | None = None,
    answers_lookup: dict | None = None,
) -> tuple[torch.Tensor, int]:
    """Target with STRUCTURED label smoothing, [B, N] (plus the number of fallback cases).

    Standard `label_smoothing` spreads the mass `eps` uniformly over all N. Here it is spread over a
    meaningful set:

    | `che_do` | mass `eps` spread over |
    |---|---|
    | `deu` | all `N` - bit-identical to PyTorch (reference / check) |
    | `Ar` | `A(r)` = entities that were answers of relation r (right direction) in TRAIN |
    | `tuvung` | mention neighbours of the gold (graph built from TEXT) |
    | `Ar+tuvung` | `eps/2` for each kind |

    The mass `1 - eps` goes to the gold (or uniformly over `N(h,r)` if `answers_lookup` is not None, i.e.
    combined with the multi-positive target). The gold is excluded from the smoothing set. An empty
    set falls back to `deu`; the number of such cases is counted and reported.

    `A(r)` is built from TRAIN only; mention neighbours come from TEXT only. Both enter only the loss
    target, never the scoring function, so evaluation does not see them.
    """
    N = s.shape[1]
    q = torch.full_like(s, sm / N) if che_do == "deu" else torch.zeros_like(s)
    n_fb = 0
    hs, rs, gs = h.tolist(), r.tolist(), gold.tolist()

    # ---- mass 1 - eps (gold, or uniform over N(h,r) when combined with multi-positive)
    if answers_lookup is not None:
        for i in range(len(hs)):
            a = answers_lookup.get((hs[i], rs[i])) or [gs[i]]
            q[i, torch.as_tensor(a, device=s.device, dtype=torch.long)] += (1.0 - sm) / len(a)
    else:
        q.scatter_add_(
            1, gold.view(-1, 1), torch.full((len(hs), 1), 1.0 - sm, device=s.device, dtype=q.dtype)
        )
    if che_do == "deu":
        return q, 0

    # ---- mass eps, VECTORISED: padded tables [*, Kmax] + masks instead of a Python loop.
    # Semantics unchanged (checked bit-exactly against the loop version).
    def _pad_lists(table: dict, row_keys: list) -> tuple[torch.Tensor, torch.Tensor]:
        """[B, Kmax] indices + mask, looked up by `row_keys` (one key per batch row)."""
        ds = [table.get(k) or [] for k in row_keys]
        km = max((len(x) for x in ds), default=0)
        if km == 0:
            return (
                torch.zeros(len(ds), 0, dtype=torch.long, device=s.device),
                torch.zeros(len(ds), 0, dtype=torch.bool, device=s.device),
            )
        ix = torch.zeros(len(ds), km, dtype=torch.long, device=s.device)
        mk = torch.zeros(len(ds), km, dtype=torch.bool, device=s.device)
        for i, x in enumerate(ds):
            if x:
                ix[i, : len(x)] = torch.as_tensor(x, device=s.device, dtype=torch.long)
                mk[i, : len(x)] = True
        return ix, mk

    parts = []
    if che_do in ("Ar", "Ar+tuvung"):
        parts.append(_pad_lists(A_r or {}, rs))
    if che_do in ("tuvung", "Ar+tuvung"):
        parts.append(_pad_lists(NB or {}, gs))
    # remove the gold from every smoothing set
    parts = [(ix, mk & (ix != gold.view(-1, 1))) for ix, mk in parts if ix.shape[1] > 0]
    if not parts:
        return q + sm / N, len(hs)
    n_each = [mk.sum(1) for _, mk in parts]  # [B] number of elements per part
    n_parts = sum((n > 0).long() for n in n_each)  # [B] number of NON-EMPTY parts
    empty_mask = n_parts == 0  # fall back to `deu`
    n_fb = int(empty_mask.sum().item())
    n_parts_c = n_parts.clamp(min=1)
    for (ix, mk), n in zip(parts, n_each):
        w = torch.where(
            n > 0,
            sm / n_parts_c.to(q.dtype) / n.clamp(min=1).to(q.dtype),
            torch.zeros_like(q[:, 0]),
        )
        q.scatter_add_(
            1,
            ix,
            torch.where(
                mk,
                w.view(-1, 1).expand_as(ix.to(q.dtype)),
                torch.zeros(1, device=s.device, dtype=q.dtype),
            ),
        )
    if n_fb:
        q[empty_mask] = q[empty_mask] + sm / N
    return q, n_fb


def structured_smoothing_target_loop(
    s: torch.Tensor,
    h,
    r,
    gold,
    sm: float,
    che_do: str,
    A_r: dict | None = None,
    NB: dict | None = None,
    answers_lookup: dict | None = None,
) -> tuple[torch.Tensor, int]:
    """Slow loop version, kept as the reference for the vectorised `structured_smoothing_target`.

    Not used for training; it exists only to show that the vectorised version has the same semantics.
    """
    N = s.shape[1]
    q = torch.full_like(s, sm / N) if che_do == "deu" else torch.zeros_like(s)
    n_fb = 0
    hs, rs, gs = h.tolist(), r.tolist(), gold.tolist()
    for i in range(len(hs)):
        if answers_lookup is not None:
            a = answers_lookup.get((hs[i], rs[i])) or [gs[i]]
            q[i, torch.as_tensor(a, device=s.device, dtype=torch.long)] += (1.0 - sm) / len(a)
        else:
            q[i, gs[i]] += 1.0 - sm
        if che_do == "deu":
            continue
        parts = []
        if che_do in ("Ar", "Ar+tuvung"):
            parts.append([e for e in (A_r.get(rs[i]) or ()) if e != gs[i]])
        if che_do in ("tuvung", "Ar+tuvung"):
            parts.append([e for e in (NB.get(gs[i]) or ()) if e != gs[i]])
        parts = [p for p in parts if p]
        if not parts:
            q[i] += sm / N
            n_fb += 1
            continue
        w = sm / len(parts)
        for p in parts:
            q[i, torch.as_tensor(p, device=s.device, dtype=torch.long)] += w / len(p)
    return q, n_fb


def loss_freq_weighted(
    model,
    ent_all,
    h,
    r,
    t,
    smoothing: float,
    freq: torch.Tensor,
    mu: float = 0.5,
    tail_only: bool = False,
    answers_lookup: dict | None = None,
    ls: dict | None = None,
    w_extra: torch.Tensor | None = None,
) -> torch.Tensor:
    """L1 - CE weighted by 1/freq(gold)^mu (counters degree bias, WWW'23).

    `tail_only=True` means the tail branch ONLY, used when `reciprocal=1`: the batch is already
    doubled to `(t, r+R, h)`, so every row is one direction and scoring `score_head` as well would be
    redundant AND wrong. `TuckerLexicalFusion.score_head` adds `+ num_rel` to `r_idx`; with
    reciprocal relations `r_idx` already reaches `2R-1`, so it would become `3R-1` and overflow
    `nn.Embedding(2R)` (IndexError).

    `w_extra` [B] is an EXTRA per-triple factor (group importance weighting), None by default,
    multiplied into the frequency weight `w_t`. The final normalisation `/ w.sum()` is unchanged, so
    `w_extra = 1` everywhere is bit-identical to `w_extra = None` (the caller normalises
    mean(w_extra) = 1 per batch).
    """
    s_tail = model.tucker.score_tail(ent_all[h], r, ent_all)
    if tail_only:
        w_t = freq[t].clamp(min=1).pow(-mu)
        if w_extra is not None:  # group importance weighting
            w_t = w_t * w_extra
        if ls is not None:  # structured label smoothing (including `deu`, for checks)
            q, _nfb = structured_smoothing_target(
                s_tail,
                h,
                r,
                t,
                smoothing,
                ls["che_do"],
                ls.get("A_r"),
                ls.get("NB"),
                answers_lookup,
            )
            ls["n_fallback"] = ls.get("n_fallback", 0) + _nfb
            lt = -(q * F.log_softmax(s_tail, dim=1)).sum(1)
        elif answers_lookup is None:
            lt = F.cross_entropy(s_tail, t, label_smoothing=smoothing, reduction="none")
        else:  # multi-positive target (KvsAll), see multi_positive_target
            q = multi_positive_target(s_tail, h, r, t, answers_lookup, smoothing)
            lt = -(q * F.log_softmax(s_tail, dim=1)).sum(1)
        return (w_t * lt).sum() / w_t.sum()
    s_head = model.tucker.score_head(r, ent_all[t], ent_all)
    w_t = freq[t].clamp(min=1).pow(-mu)
    w_h = freq[h].clamp(min=1).pow(-mu)
    lt = F.cross_entropy(s_tail, t, label_smoothing=smoothing, reduction="none")
    lh = F.cross_entropy(s_head, h, label_smoothing=smoothing, reduction="none")
    return ((w_t * lt).sum() / w_t.sum() + (w_h * lh).sum() / w_h.sum()) / 2


def loss_hard_negatives(
    model, ent_all, h, r, t, smoothing: float, so_hard: int = 64, tau: float = 0.05
) -> torch.Tensor:
    """L2 - keep only the `so_hard` hardest candidates + the gold, CE on the subset, temperature tau (SimKGC)."""
    out = 0.0
    for s_all, gold in (
        (model.tucker.score_tail(ent_all[h], r, ent_all), t),
        (model.tucker.score_head(r, ent_all[t], ent_all), h),
    ):
        s = s_all / tau
        che = s.clone()
        che.scatter_(
            1, gold.unsqueeze(1), float("-inf")
        )  # exclude the gold from the hard-negative pick
        idx = che.topk(min(so_hard, s.shape[1] - 1), dim=1).indices
        cand = torch.cat([gold.unsqueeze(1), idx], dim=1)  # gold in column 0
        out = out + F.cross_entropy(
            s.gather(1, cand),
            torch.zeros(s.shape[0], dtype=torch.long, device=s.device),
            label_smoothing=smoothing,
        )
    return out / 2


# ---------------------------------------------------------------- H1 learning-rate schedule
def cosine_schedule_h1(opt, num_epoch: int):
    """H1 - cosine annealing; early-stop patience is handled by train.py."""
    return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epoch)


# ---------------------------------------------------------------- H2 late fusion
class TextHead(nn.Module):
    """H2 - independent text head: additive score beta * cos(W([text_h ; r]), W2(text_t))."""

    def __init__(self, dim_txt: int, num_rel: int, dim: int = 256) -> None:
        super().__init__()
        self.Wr = nn.Embedding(num_rel * 2, dim)
        self.W1 = nn.Linear(dim_txt, dim)
        self.W2 = nn.Linear(dim_txt, dim)
        self.beta = nn.Parameter(torch.tensor(0.1))
        nn.init.xavier_normal_(self.Wr.weight)

    def diem(self, T: torch.Tensor, src_idx: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """src_idx [B] source entity indices; returns the [B,N] additive score."""
        a = F.normalize(self.W1(T[src_idx]) + self.Wr(r), dim=-1)
        b = F.normalize(self.W2(T), dim=-1)
        return self.beta * (a @ b.t())


# ---------------------------------------------------------------- X1: relation-prompt text head
class TextPromptHead(nn.Module):
    """X1 - `q = MLP([T[source] ; emb(r)])`, `c = MLP2(T[.])`, score `+= beta * cos(q, c)`.

    Unlike `TextHead` (H2) it CONCATENATES and applies an MLP (multiplicative interaction) instead
    of ADDING `Wr`. The `.diem` interface is unchanged, so `TuckerTextAdd` and all gold-invariance
    guards apply as they are.
    """

    def __init__(self, dim_txt: int, num_rel: int, dim: int = 256) -> None:
        super().__init__()
        self.Er = nn.Embedding(num_rel * 2, dim)
        self.W1 = nn.Sequential(nn.Linear(dim_txt + dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.W2 = nn.Sequential(nn.Linear(dim_txt, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.beta = nn.Parameter(torch.tensor(0.1))
        nn.init.xavier_normal_(self.Er.weight)

    def diem(self, T: torch.Tensor, src_idx: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        a = F.normalize(self.W1(torch.cat([T[src_idx], self.Er(r)], dim=-1)), dim=-1)
        b = F.normalize(self.W2(T), dim=-1)
        return self.beta * (a @ b.t())


# ---------------------------------------------------------------- X2: sparse BM25 head
class BM25Head(nn.Module):
    """X2 - score `+= gamma * normalised_bm25(source, c)` for c in the top-K, 0 outside the top-K.

    `idx`/`val` [N,K] are built locally (`build_lexical_data.py`) from entity descriptions and touch
    no triple. `r` does not enter the formula: BM25 is a purely lexical signal, deliberately
    independent of the relation.
    """

    def __init__(
        self,
        idx: torch.Tensor,
        val: torch.Tensor,
        gamma_init: float = 0.1,
        num_ent: int | None = None,
    ) -> None:
        super().__init__()
        # Rows with fewer than K neighbours are padded with `idx = 0, val = 0`. Scattering them
        # directly would write 0 into candidate 0, which genuinely is a BM25 neighbour of 39-103
        # entities per dataset; and `scatter_` with DUPLICATE indices is non-deterministic in PyTorch.
        # Fix: map padding cells to a sentinel index N, score into the extra column, then cut it off.
        n = int(num_ent if num_ent is not None else idx.max().item() + 1)
        idx = idx.long().clone()
        idx[val <= 0] = n  # every padding cell -> sentinel column
        self.N = n
        self.register_buffer("idx", idx, persistent=False)
        self.register_buffer("val", val, persistent=False)
        self.gamma = nn.Parameter(torch.tensor(gamma_init))

    def diem(self, T: torch.Tensor, src_idx: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(src_idx.shape[0], self.N + 1, device=T.device, dtype=self.val.dtype)
        out.scatter_(1, self.idx[src_idx], self.val[src_idx])
        return self.gamma * out[:, : self.N]


# ---------------------------------------------------------------- L3: multi-label BCE
def loss_bce_all_tails(model, ent_all, h, r, t, nhan_mt, smoothing: float = 0.0) -> torch.Tensor:
    """L3 - BCE over ALL true tails of (h,r) in TRAIN (ConvE-style 1-N scoring).

    `nhan_mt` is a function (h, r) -> 0/1 tensor [B, N], prebuilt from train in `build_aux_data`.

    `smoothing`: the original ConvE uses label smoothing together with BCE. Without it the loss
    collapses to ~0 within 10 epochs and MRR collapses too: with 15 000 classes and ~1 positive,
    predicting ~0 everywhere already drives BCE to 0 (the model learns not to answer). Smoothing
    lifts negative labels to `s/N`, removing that trivial solution. This is a hyper-parameter change,
    so results must be reported as a separate round, not merged into the screening table.
    """
    s = model.tucker.score_tail(ent_all[h], r, ent_all)
    y = nhan_mt(h, r)
    if smoothing > 0:
        y = y * (1.0 - smoothing) + smoothing / y.shape[1]
    return F.binary_cross_entropy_with_logits(s, y)


# ---------------------------------------------------------------- LSR: lexical-semantic router
class LexicalBlock(nn.Module):
    """Block L - LEARNED lexical matcher over the BM25 top-100 (features built offline).

    `feat[N,K,7]` and `idx[N,K]` are looked up by the SOURCE entity, so nothing depends on the gold
    (a structural invariant). Outside the top-K the score is 0, using a sentinel index N so padding
    cells cannot overwrite candidate 0 (same bug as the first version of `BM25Head`).
    """

    def __init__(self, idx: torch.Tensor, feat: torch.Tensor, num_rel: int, dim: int = 64) -> None:
        super().__init__()
        n, k, f = feat.shape
        self.N, self.K = n, k
        ix = idx.long().clone()
        ix[feat[:, :, 0] <= 0] = n  # sentinel
        self.register_buffer("idx", ix, persistent=False)
        self.register_buffer("feat", feat.float(), persistent=False)
        self.register_buffer("co", (feat[:, :, 0] > 0).float(), persistent=False)
        self.Er = nn.Embedding(num_rel * 2, dim)
        self.mlp = nn.Sequential(nn.Linear(f + dim, dim), nn.ReLU(), nn.Linear(dim, 1))
        nn.init.xavier_normal_(self.Er.weight)

    def diem(self, src_idx: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """[B,N] lexical score; 0 outside the top-K."""
        f = self.feat[src_idx]  # [B,K,7]
        e = self.Er(r).unsqueeze(1).expand(-1, self.K, -1)  # [B,K,dim]
        v = self.mlp(torch.cat([f, e], dim=-1)).squeeze(-1)  # [B,K]
        v = v * self.co[src_idx]  # padding cells -> 0
        out = torch.zeros(src_idx.shape[0], self.N + 1, device=v.device, dtype=v.dtype)
        out.scatter_(1, self.idx[src_idx], v)
        return out[:, : self.N]


class LexicalUnionBlock(nn.Module):
    """Lexical block over the UNION(BM25 top-100 + top-100 of S), 10 features.

    Unlike `LexicalBlock` the lookup axis is the QUERY `(source, r_idx)` through the table
    `tra[N, 2R]`, because the top-S depends on the relation as well.

    Gold invariance is structural: `diem(src_idx, r)` only receives `(source, r_idx)`; the gold is not
    a parameter, so permuting the gold cannot change the candidate set or the scores. The lookup key
    must never be `(h, r, t)` - that would be a leak.

    A query missing from the table (`tra = -1`) uses a padding row of sentinels, so its score is 0.
    """

    def __init__(
        self, tra: torch.Tensor, idx: torch.Tensor, feat: torch.Tensor, num_rel: int, dim: int = 64
    ) -> None:
        super().__init__()
        q, k, f = feat.shape
        self.N = int(tra.shape[0])
        self.K, self.Q = k, q
        ix = idx.long().clone()
        ix[ix < 0] = self.N  # sentinel
        # `device=` is required: new tensors must live on the same device as the inputs (`idx/feat`
        # were moved to the device in train.py). Without it the code runs on CPU but fails on GPU.
        ix = torch.cat(
            [ix, torch.full((1, k), self.N, dtype=torch.long, device=ix.device)]
        )  # padding row
        ft = torch.cat([feat.float(), torch.zeros(1, k, f, device=feat.device, dtype=torch.float)])
        self.register_buffer("tra", tra.long(), persistent=False)
        self.register_buffer("idx", ix, persistent=False)
        self.register_buffer("feat", ft, persistent=False)
        # has a candidate <=> at least one of the last three source flags is on
        self.register_buffer("co", (ft[:, :, -3:].sum(-1) > 0).float(), persistent=False)
        self.Er = nn.Embedding(num_rel * 2, dim)
        self.mlp = nn.Sequential(nn.Linear(f + dim, dim), nn.ReLU(), nn.Linear(dim, 1))
        nn.init.xavier_normal_(self.Er.weight)

    def diem(self, src_idx: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """[B,N] lexical score over the UNION; 0 outside the candidate set of that very query."""
        q = self.tra[src_idx, r]
        q = torch.where(q < 0, torch.full_like(q, self.Q), q)  # padding row
        f = self.feat[q]  # [B,K,10]
        e = self.Er(r).unsqueeze(1).expand(-1, self.K, -1)
        v = self.mlp(torch.cat([f, e], dim=-1)).squeeze(-1) * self.co[q]
        out = torch.zeros(src_idx.shape[0], self.N + 1, device=v.device, dtype=v.dtype)
        out.scatter_(1, self.idx[q], v)
        return out[:, : self.N]


class GateRouter(nn.Module):
    """Router - `g(q) = softmax(MLP[query features])`, initialised so that g ~ (1, 0, 0).

    Initialisation: last layer `weight = 0`, `bias = (b, 0, 0)` with large b, so softmax ~ (1,0,0) and
    the score at epoch 0 equals the base model. Training starts from the base and learns to open L/M.
    """

    def __init__(
        self, num_rel: int, dim: int = 32, b0: float = 6.0, init_mode: str = "nen"
    ) -> None:
        """`init_mode`: "nen" -> g ~ (1,0,0); "deu" -> g ~ (1/3,1/3,1/3).

        The uniform start tests whether initialising near the base creates a flat region from which
        the router cannot escape (g_L stayed at 0.017-0.047 from epoch 50 to 300, g_M = 0.000).
        """
        super().__init__()
        self.Er = nn.Embedding(num_rel * 2, dim)
        self.mlp = nn.Sequential(nn.Linear(6 + dim, dim), nn.ReLU(), nn.Linear(dim, 3))
        nn.init.xavier_normal_(self.Er.weight)
        nn.init.zeros_(self.mlp[-1].weight)
        with torch.no_grad():
            self.mlp[-1].bias.copy_(
                torch.zeros(3) if init_mode == "deu" else torch.tensor([b0, 0.0, 0.0])
            )

    def forward(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """`x` [B,6] = [log1p(deg), log1p(|A|), has_text, len_text/512, has_BM25, direction]."""
        return F.softmax(self.mlp(torch.cat([x, self.Er(r)], dim=-1)), dim=-1)


# ------------------------------------------------------------------ Reader
class Reader(nn.Module):
    """Late interaction over the CANDIDATE's description tokens, conditioned on (h, r).

    score_R(t) = (1/K) * sum_k  max_j  <slot_k(h, r), W_tok * tok_j(t)>

    `slot_k` are K query-side probes produced from `[e_h ; e_r]`; `W_tok` projects frozen BERT
    token states 768 -> `d_proj`. Nothing on the candidate side is learned beyond `W_tok`, and
    the token states themselves are frozen, so this adds `K*d_proj` probes and one linear map.

    No term may read the gold. Every input here is query-side (`e_h`, `e_r`) or candidate-side
    (`tok`), and the candidate axis is the FULL entity set, scored identically for every candidate.
    There is no path by which the gold index enters, and `TuckerLexicalFusion.score_tail` keeps its
    `assert torch.equal(h, ent_all[self._h])` guard.

    Cost. The full product is B*K*N*L*d, which cannot be materialised: for B=512, K=8,
    N=15000, L=96, d=128 the intermediate [B*K, N*L] alone is ~23 GB in fp16. Candidates are
    therefore chunked (`chunk`), the max over tokens is taken per chunk, and only [B*K, chunk]
    survives (~400 MB at chunk=512).
    """

    def __init__(
        self,
        dim_ent: int,
        dim_rel: int,
        K: int = 8,
        d_proj: int = 128,
        pool: str = "token",
        lam_init: float = 0.3,
    ) -> None:
        super().__init__()
        self.K, self.d_proj, self.pool = K, d_proj, pool
        self.W_tok = nn.Linear(768, d_proj, bias=False)
        self.slot = nn.Sequential(
            nn.Linear(dim_ent + dim_rel, 256), nn.GELU(), nn.Linear(256, K * d_proj)
        )
        # lambda is LEARNED but starts at 0.3, so step 0 is close to the structure-only model
        # instead of a random mixture.
        self.lam = nn.Parameter(torch.tensor(float(lam_init)))

    def score_subset(
        self,
        e_h: torch.Tensor,
        e_r: torch.Tensor,
        tok: torch.Tensor,
        mask: torch.Tensor,
        idx: torch.Tensor,
        chunk: int = 512,
    ) -> torch.Tensor:
        """-> [B, M]. Score ONLY the `M` candidates that `idx` [B, M] points to (negative sampling).

        Why it exists. `diem()` scores ALL N candidates and is called at EVERY training step (1-vs-all
        CE), so `B*K*N*L*d` = 1.51 TFLOP/batch on MKG-W, i.e. 1.1-1.3 h/run for training alone. With
        `M = 256` that becomes 0.026 TFLOP, `N/M = 58x` cheaper.
        `W_tok` runs only on the UNION of indices used in the batch, not on all N: running it on all
        N would make `N*L*768*d` = 0.28 TFLOP/batch the dominant term and the gain would drop to ~6x.
        The real union size is logged (it depends on how negatives are drawn).
        """
        B, M = idx.shape
        sl = self.slot(torch.cat([e_h, e_r], dim=-1)).view(B, self.K, self.d_proj)
        union_idx, inverse = torch.unique(idx.reshape(-1), return_inverse=True)  # [U], [B*M]
        t_ = tok[union_idx]  # [U, L, 768]
        m_ = mask[union_idx]  # [U, L]
        U, L = t_.shape[0], t_.shape[1]
        P = self.W_tok(t_.to(sl.dtype).reshape(U * L, 768)).view(U, L, self.d_proj)
        row_of = inverse.view(B, M)
        res = e_h.new_empty(B, M)
        # Chunk over QUERIES so that [B_c, K, M, L] does not blow up memory.
        # Chunk by an ELEMENT BUDGET, not by a proportional formula: a formula like
        # `min(B, chunk*512/M)` gives 1024 at M = 256, i.e. no chunking at all, and the tensor
        # `Pi = [c=512, M=256, L=96, d=128]` is 1.6e9 elements = 6.0 GB fp32 per batch. FLOPs are
        # cheap, but allocating and gathering 6 GB per batch dominates (measured 18.8 min/run vs ~4
        # min of FLOPs). A budget of 2^27 elements ~ 512 MB fp32 gives c = 42 at M=256 (~528 MB).
        ELEMENT_BUDGET = 1 << 27
        b_chunk = max(1, min(B, ELEMENT_BUDGET // max(1, M * L * self.d_proj)))
        for s0 in range(0, B, b_chunk):
            s1 = min(B, s0 + b_chunk)
            Pi = P[row_of[s0:s1]]  # [c, M, L, d]
            mi = m_[row_of[s0:s1]]  # [c, M, L]
            x = torch.einsum("ckd,cmld->ckml", sl[s0:s1], Pi)
            x = x.masked_fill(~mi.unsqueeze(1), -1e4)
            if self.pool == "mean":
                mm = mi.unsqueeze(1).to(x.dtype)
                x = x.masked_fill(~mi.unsqueeze(1), 0.0).sum(-1) / mm.sum(-1).clamp(min=1.0)
            else:
                x = x.max(dim=-1).values  # [c, K, M]
            co = mi.any(dim=-1).to(x.dtype)  # [c, M] - 0 for entities without text
            res[s0:s1] = x.mean(dim=1) * co
        self.last_union_size = int(U)  # for logging: real size of the index union
        return res

    def diem(
        self,
        e_h: torch.Tensor,
        e_r: torch.Tensor,
        tok: torch.Tensor,
        mask: torch.Tensor,
        chunk: int = 512,
    ) -> torch.Tensor:
        """-> [B, N]. `tok` [N,L,768] fp16 buffer, `mask` [N,L] bool, True = real token."""
        B = e_h.shape[0]
        # Permanent guard: the input width of `slot` must match the declared width. A wrong width
        # measures something other than intended, and the default `F.linear` message
        # ("64x48 and 784x256") does not say who declared it wrong.
        _got, _expected = e_h.shape[-1] + e_r.shape[-1], self.slot[0].in_features
        assert _got == _expected, (
            f"Reader: slot declares {_expected} inputs but got {_got} "
            f"(e_h {e_h.shape[-1]} + e_r {e_r.shape[-1]}) - wrong dim_ent in enable_reader"
        )
        sl = self.slot(torch.cat([e_h, e_r], dim=-1)).view(B * self.K, self.d_proj)
        res = e_h.new_empty(B, tok.shape[0])
        for s0 in range(0, tok.shape[0], chunk):
            t = tok[s0 : s0 + chunk]  # [c, L, 768]
            m = mask[s0 : s0 + chunk]  # [c, L]
            c, L = t.shape[0], t.shape[1]
            P = self.W_tok(t.to(sl.dtype).reshape(c * L, 768))  # [c*L, d]
            x = sl @ P.t()  # [B*K, c*L]
            x = x.view(B * self.K, c, L)
            # padding must not win the max; -inf would poison rows with no token at all, so
            # a large finite floor is used and all-padding rows are zeroed right after.
            x = x.masked_fill(~m.unsqueeze(0), -1e4)
            if self.pool == "mean":
                # mean-pool ablation: average the real tokens instead of taking the max
                mm = m.unsqueeze(0).to(x.dtype)
                x = x.masked_fill(~m.unsqueeze(0), 0.0).sum(-1) / mm.sum(-1).clamp(min=1.0)
            else:
                x = x.max(dim=-1).values  # [B*K, c]
            x = x.view(B, self.K, c).mean(dim=1)  # average the K probes
            co = m.any(dim=-1).to(x.dtype)  # 0 for entities w/o text
            res[:, s0 : s0 + chunk] = x * co
        return res


class TuckerLexicalFusion(nn.Module):
    """Wraps TuckER: `s = g_S*z(S) + g_L*z(L) + g_M*z(M)`, z = per-QUERY z-score."""

    def __init__(
        self,
        goc,
        lexical_block: "LexicalBlock",
        text_head: "TextPromptHead",
        T: torch.Tensor,
        R: "GateRouter",
        meta: torch.Tensor,
        num_rel: int,
        tat: str = "",
    ) -> None:
        super().__init__()
        self.goc, self.L, self.M, self.R = goc, lexical_block, text_head, R
        self.register_buffer("T", T, persistent=False)
        self.register_buffer("meta", meta, persistent=False)  # [N,4] deg, |A| replaced by 0, text..
        self.num_rel = num_rel
        self.tat = tat  # "R" | "L" | "M" | "RM" (S+L only) | ""
        self.g_L = 0.15  # FIXED weight when tat="RM" (trained at 0.15)
        self.luu_SL = None  # where the separate S and L matrices are collected, for the g_L sweep
        self.khoa_SL = []  # explicit (h,r,t,direction) key of every collected row
        self.hai_chieu = False  # True -> also collect the tail direction (default: head only)
        self.thu_SL = None  # TAG set by evaluate.py: collect only when == "head" / "tail".
        # Relying on call ORDER is wrong - the invariance check calls
        # twice more in the first batch (2 921 rows instead of 2 665).
        self.g_tb = None  # for logging
        # --- Product-of-Experts with a frozen lexical bias model (Clark et al. 2019), OFF by default.
        # `poe = (alpha, tau)` or None. When set, `_score_mix` returns PoE TRAINING logits *only
        # while self.training is True*; evaluation is untouched, so the usual g-sweep keeps working.
        self.poe = None
        # --- Reader (`--reader {che,goc}`), OFF by default ------------------------------
        # Plain attributes, NOT a submodule. Instantiating `Reader` here would draw from the RNG at
        # construction time even with the flag off, which shifts every later draw and breaks
        # bit-identity with the published path. The module is built LAZILY by `enable_reader()`.
        self.reader = None
        # The full-N scoring pass at train time must SKIP the Reader (its term enters only
        # `CE_sub`). The flag is toggled by `train.py` around that pass; eval always leaves it False.
        self.reader_skip = False
        # When skipping the Reader, KEEP the S-mix (still with grad) so `train.py` can add `CE_sub`
        # without a second full-N pass, and `CE_full` remains exactly the baseline loss (L1).
        self.last_smix = None
        self.reader_tok = None
        self.reader_mask = None
        self.reader_chunk = 512

    def enable_reader(
        self,
        tok: torch.Tensor,
        mask: torch.Tensor,
        dim_ent: int,
        K: int = 8,
        d_proj: int = 128,
        pool: str = "token",
        lam: float = 0.3,
        chunk: int = 512,
    ) -> "Reader":
        """Build the Reader and attach its frozen token buffers. Call ONLY when enabled.

        `dim_ent` is REQUIRED and must be the width of `ent_all` (`encode()["h_e"]`), i.e. of the
        `vec` that `_score_mix` hands to `Reader.diem`. An earlier version GUESSED it with
        `self.goc.E.weight.shape[1] if hasattr(self.goc, "E") else self.T.shape[1]`: the base model
        has no `.E`, so every call silently fell back to the TEXT width 768 and built `slot` as
        `Linear(768+dim_rel, ...)`. Nothing caught it, because every earlier check only exercised
        `--reader none`; it surfaced as `mat1 and mat2 shapes cannot be multiplied` in the first run
        with the flag ON.
        """
        assert mask.dtype == torch.bool, "reader mask must be bool (True = real token)"
        assert tok.shape[:2] == mask.shape, f"tok {tuple(tok.shape)} vs mask {tuple(mask.shape)}"
        dim_e = int(dim_ent)
        dim_r = self.goc.R.weight.shape[1]
        self.reader = Reader(dim_e, dim_r, K=K, d_proj=d_proj, pool=pool, lam_init=lam).to(
            mask.device
        )
        self.reader_tok, self.reader_mask, self.reader_chunk = tok, mask, chunk
        return self.reader

    @staticmethod
    def _z(x: torch.Tensor) -> torch.Tensor:
        finite = torch.isfinite(x)
        m = torch.where(finite, x, torch.zeros_like(x)).sum(1, keepdim=True) / finite.sum(
            1, keepdim=True
        )
        v = torch.where(finite, (x - m) ** 2, torch.zeros_like(x)).sum(1, keepdim=True)
        sd = (v / finite.sum(1, keepdim=True)).sqrt().clamp(min=1e-6)
        return torch.where(finite, (x - m) / sd, x)

    def lexical_score_z(self, src_idx, r_idx):
        """z-scored lexical score L, the SAME expression `_score_mix` uses (do not rewrite it)."""
        return self._z(self.L.diem(src_idx, r_idx))

    def _score_mix(self, vec, r_idx, ent_all, src_idx):
        s = self._z(self.goc.score_tail(vec, r_idx, ent_all))
        skip_L = self.tat in ("L",)
        skip_M = self.tat in ("M", "RM")
        lex_z = torch.zeros_like(s) if skip_L else self._z(self.L.diem(src_idx, r_idx))
        text_z = torch.zeros_like(s) if skip_M else self._z(self.M.diem(self.T, src_idx, r_idx))
        if self.luu_SL is not None and self.thu_SL in ("head", "tail"):
            # Collect S and L SEPARATELY (fp16, max subtracted). The tag `thu_SL` is the direction
            # being scored (set by evaluate.py); both directions may be collected, each row carrying
            # its direction in `luu_SL["huong"]` (never infer the direction from call order).
            for key, x in (("S", s), ("L", lex_z)):
                mx = x.max(dim=1, keepdim=True).values
                self.luu_SL[key].append((x - mx).half().cpu())
                self.luu_SL[key + "_max"].append(mx.squeeze(1).float().cpu())
            if "huong" in self.luu_SL:
                self.luu_SL["huong"] += [self.thu_SL] * s.shape[0]
        if self.poe is not None and self.training:
            # PoE training logits: log p_S + alpha * log p_L, L DETACHED.
            # alpha = 0 reduces to log_softmax(S) alone, i.e. the base with g_L = 0 at train time.
            _alpha, _tau = self.poe
            return F.log_softmax(s, dim=1) + _alpha * F.log_softmax(lex_z.detach() / _tau, dim=1)
        if self.tat == "RM":  # S + g_L*L only, no router, no M
            g = torch.tensor([1.0, self.g_L, 0.0], device=s.device).expand(s.shape[0], 3)
        elif self.tat == "R":  # ablation: fixed g
            g = torch.tensor([0.5, 0.25, 0.25], device=s.device).expand(s.shape[0], 3)
        else:
            x = torch.cat(
                [
                    self.meta[src_idx],
                    torch.full(
                        (src_idx.shape[0], 1), float(r_idx[0] >= self.num_rel), device=s.device
                    ),
                ],
                dim=-1,
            )
            g = self.R(x, r_idx)
        self.g_tb = g.detach()
        out = g[:, :1] * s + g[:, 1:2] * lex_z + g[:, 2:3] * text_z
        if self.reader is not None and self.reader_skip:
            self.last_smix = out
        if self.reader is not None and not self.reader_skip:
            # logit = z(S-mix) + lambda * z(R). Both sides are z-scored per query, so lambda is
            # a scale between two comparable quantities rather than between raw magnitudes.
            e_r = self.goc.R(r_idx)
            rz = self._z(
                self.reader.diem(
                    vec, e_r, self.reader_tok, self.reader_mask, chunk=self.reader_chunk
                )
            )
            out = self._z(out) + self.reader.lam * rz
        return out

    def score_tail(self, h, r_idx, ent_all):
        assert torch.equal(
            h, ent_all[self._h]
        ), "TuckerLexicalFusion: source vector does not match the indices set by dat_chi_so"
        return self._score_mix(h, r_idx, ent_all, self._h)

    def score_head(self, r_idx, t, ent_all):
        return self._score_mix(ent_all[self._t], r_idx + self.num_rel, ent_all, self._t)

    def dat_chi_so(self, h_idx, t_idx) -> None:
        self._h, self._t = h_idx, t_idx


# ---------------------------------------------------------------- H2: TuckER wrapper
class TuckerTextAdd(nn.Module):
    """Wraps TuckER and adds a score from a text head. Keeps the `score_tail/score_head` interface
    unchanged, so both training and evaluation use it without touching `evaluate.py`."""

    def __init__(
        self, goc, dau: "TextHead", T: torch.Tensor, num_rel: int, text_only: bool = False
    ) -> None:
        super().__init__()
        self.goc = goc
        self.dau = dau
        self.register_buffer("T", T, persistent=False)
        self.num_rel = num_rel
        self.text_only = text_only  # text-only: drop the structure score entirely

    def score_tail(self, h, r_idx, ent_all):
        # GUARD: the declared source indices must match the vector being scored. Without it, an
        # index leak in the H2 evaluation survived several reviews unnoticed.
        assert torch.equal(h, ent_all[self._h]), (
            "TuckerTextAdd: source vector does not match the indices set by dat_chi_so - "
            "call dat_chi_so(source, target) right before every scoring call"
        )
        d = self.dau.diem(self.T, self._h, r_idx)
        return d if self.text_only else self.goc.score_tail(h, r_idx, ent_all) + d

    def score_head(self, r_idx, t, ent_all):
        d = self.dau.diem(self.T, self._t, r_idx + self.num_rel)
        return d if self.text_only else self.goc.score_head(r_idx, t, ent_all) + d

    def dat_chi_so(self, h_idx, t_idx) -> None:
        """Train/eval must declare the source entity indices before every scoring call (TuckER only gets vectors)."""
        self._h, self._t = h_idx, t_idx


# ---------------------------------------------------------------- auxiliary data
def build_aux_data(variant: str, data, dev, so_lang_gieng: int = 8) -> dict:
    """Build the auxiliary tensors of each direction from the run's own **train** split (never valid/test)."""
    N, R = data.num_ent, data.num_rel
    res: dict = {}
    if variant == "L1":
        freq = torch.zeros(N)
        for h, _r, t in data.train:
            freq[h] += 1
            freq[t] += 1
        res["freq"] = freq.to(dev)
    if variant == "S1":
        k = so_lang_gieng
        nb_e = torch.full((N, k), 0, dtype=torch.long)
        nb_r = torch.full((N, k), R * 2, dtype=torch.long)  # R*2 = padding_idx
        nb_m = torch.ones(N, k, dtype=torch.bool)
        count = [0] * N
        deg = torch.zeros(N, dtype=torch.long)
        for h, r, t in data.train:
            deg[h] += 1
            deg[t] += 1
            for a, b, rr in ((h, t, r), (t, h, r + R)):
                if count[a] < k:
                    nb_e[a, count[a]] = b
                    nb_r[a, count[a]] = rr
                    nb_m[a, count[a]] = False
                    count[a] += 1
        degree_bin = torch.where(deg <= 2, 0, torch.where(deg <= 10, 1, 2))
        res.update(
            nb_ent=nb_e.to(dev),
            nb_rel=nb_r.to(dev),
            nb_mask=nb_m.to(dev),
            degree_bin=degree_bin.to(dev),
        )
    if variant in ("S2", "G1"):
        src, dst, rel = [], [], []
        for h, r, t in data.train:
            src += [h, t]
            dst += [t, h]
            rel += [r, r + R]
        deg = torch.zeros(N)
        for d_ in dst:
            deg[d_] += 1
        if variant == "G1":
            # mention edges: NEW relation r_men = 2R (forward) and 2R+1 (inverse).
            # Derived from TEXT, never from valid/test triples. NOT appended to data.train, so the
            # filter dict and negative sampling stay unchanged.
            n0 = len(src)
            for a, b in getattr(data, "canh_nhac_ten", []):
                src += [a, b]
                dst += [b, a]
                rel += [2 * R, 2 * R + 1]
            deg = torch.zeros(N)
            for d_ in dst:
                deg[d_] += 1
            print(
                f"[G1] {len(src) - n0} directed mention edges added to {n0} train edges", flush=True
            )
        res.update(
            src=torch.tensor(src, device=dev),
            dst=torch.tensor(dst, device=dev),
            rel=torch.tensor(rel, device=dev),
            deg=deg.to(dev),
        )
    if variant == "L3":
        # multi-label matrix built SPARSELY from train, looked up by (h, r)
        tra: Dict[tuple, list] = {}
        for h, r_, t in data.train:
            tra.setdefault((h, r_), []).append(t)

        def nhan_mt(hb: torch.Tensor, rb: torch.Tensor) -> torch.Tensor:
            y = torch.zeros(hb.shape[0], N, device=hb.device)
            for i, (a, b) in enumerate(zip(hb.tolist(), rb.tolist())):
                v = tra.get((a, b))
                if v:
                    y[i, torch.tensor(v, device=hb.device)] = 1.0
            return y

        res["nhan_mt"] = nhan_mt
    return res
