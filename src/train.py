#!/usr/bin/env python3
"""Train ViProNet. Loss = L_rank + lambda_1 * L_align + lambda_2 * L_gate.

Usage: python3 train.py --gate_mode gate_prior_init --seed 42 \
        --t_path feats/T.pt --v_path feats/V.pt --exp main --dump_ranks result/ranks_main.tsv
Every run requires --seed; metrics are logged at every evaluation.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from config import (
    ALIGN_TEMP,
    BATCH_SIZE,
    DATA_DIR,
    DIM_REL,
    DIM_STR,
    DIM_TXT,
    DIM_VIS,
    GATE_MODES,
    LAMBDA_ALIGN,
    LAMBDA_GATE,
    LR,
    NUM_EPOCH,
    TIER_PRIOR,
    VALID_EPOCH,
)
from data import ViMMKGData
from evaluate import evaluate
from model import ViProNet


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def l_rank(model, ent_all, h, r, t, smoothing: float) -> torch.Tensor:
    """1-vs-all in both directions, label-smoothed cross-entropy (standard TuckER, no negative sampling)."""
    s_tail = model.tucker.score_tail(ent_all[h], r, ent_all)  # [B,N]
    s_head = model.tucker.score_head(r, ent_all[t], ent_all)  # [B,N]
    loss = F.cross_entropy(s_tail, t, label_smoothing=smoothing) + F.cross_entropy(
        s_head, h, label_smoothing=smoothing
    )
    return loss / 2


def l_rank_tail(model, ent_all, h, r, t, smoothing: float) -> torch.Tensor:
    """Tail direction only (reciprocal mode: head prediction is the tail prediction of r^-1 in the augmented batch)."""
    s_tail = model.tucker.score_tail(ent_all[h], r, ent_all)  # [B,N]
    return F.cross_entropy(s_tail, t, label_smoothing=smoothing)


def sampled_ce_shared_pool(
    T, ent_all, h, r, t, smoothing: float, n_neg: int, n_hard: int, n_pool: int
) -> "tuple[torch.Tensor, int, float]":
    """Sampled softmax with in-batch shared negatives:
    CE_sub(z(S-mix)_sub + lam * z(R)_sub) over a pool shared by the whole batch.

    Called right after the base loss branch, with `T.reader_skip = True` during that branch:
    the base branch scored all N candidates and `_cham` kept the S-mix in `T.last_smix` (still with
    grad), so nothing is scored twice and `CE_full` stays exactly the baseline loss.

    Pool `P` = gold ids of the batch + hard negatives of the whole batch + shared random ids,
    capped at `n_pool`. Per-query subset: `M = 1 + n_neg` candidates, gold at column 0, `n_hard`
    hard negatives = top z(S) of that query's own scores restricted to `P`, the rest random
    from `P`; the CE target is therefore 0.

    The pool is capped because `W_tok` runs on the union of indices of the batch: drawing random
    negatives per query (512 x 127 draws over ~15 000 entities) would cover almost all entities and
    make `W_tok` the dominant cost. Capping the pool at 2 048 bounds |union| <= 2 048 by
    construction, independent of the model.
    z(S-mix)_sub is gathered AFTER z over all N; z(R)_sub is normalised over the M columns only.
    The two use different denominators and therefore different scales: this is a modelling
    choice, not an implementation detail.
    """
    s_full = T.last_smix
    if s_full is None:
        raise SystemExit(
            "`last_smix` is empty: the base loss branch did not run through `_cham` with "
            "`reader_skip=True`. Check the call order."
        )
    T.last_smix = None
    sz = T._z(s_full)
    B, N = sz.shape
    n_hard = min(n_hard, n_neg)
    n_rand = n_neg - n_hard
    with torch.no_grad():
        gold = torch.unique(t)
        # Cap only the ADDED part, never truncate after the union: `unique` sorts ascending, so
        # truncating would drop the LARGE ids, possibly including gold, and `searchsorted(P, t)`
        # would return |P| (index out of bounds). Building under a budget keeps gold always
        # present and |P| <= n_pool by construction.
        if n_pool < int(gold.numel()):
            raise SystemExit(
                f"--reader_npool {n_pool} < number of distinct gold ids in the batch "
                f"{int(gold.numel())}. The pool must contain all gold ids: set "
                f"`--reader_npool` >= `--batch_size`."
            )
        dmax = sz.detach().max(dim=0).values  # [N]
        dmax[gold] = float("-inf")  # gold already has its own slot
        budget = max(0, n_pool - int(gold.numel()))
        n_r = min(n_rand, budget)
        n_fill_hard = min(max(0, budget - n_r), N - int(gold.numel()))
        hard_ids = dmax.topk(n_fill_hard).indices if n_fill_hard > 0 else gold[:0]
        rand_ids = torch.randint(0, N, (n_r,), device=sz.device) if n_r else gold[:0]
        P = torch.unique(torch.cat([gold, hard_ids, rand_ids]))  # [U] ascending, U <= n_pool
        U = int(P.numel())
        # position of gold inside P (P is sorted ascending, so searchsorted is exact)
        pos_gold = torch.searchsorted(P, t)
        szP = sz.detach().gather(1, P.view(1, -1).expand(B, -1))  # [B, U]
        szP.scatter_(1, pos_gold.view(-1, 1), float("-inf"))
        k_take = min(n_hard, max(0, U - 1))
        pos_hard = szP.topk(k_take, dim=1).indices if k_take else pos_gold.view(-1, 0)
        shortfall = n_hard - k_take  # filled with random ids if P is too small
        n_extra = n_rand + shortfall
        pos_rand = (
            torch.randint(0, U, (B, n_extra), device=sz.device) if n_extra else pos_gold.view(-1, 0)
        )
        pos = torch.cat([pos_gold.view(-1, 1), pos_hard, pos_rand], dim=1)  # [B, M] positions in P
        idx = P[pos]  # [B, M] global ids
    assert U <= n_pool, f"|P| = {U} > --reader_pool {n_pool}"
    s_sub = sz.gather(1, idx)
    # `e_r` is taken exactly as in `_cham`: `self.goc.R(r_idx)`
    r_sub = T._z(
        T.reader.score_subset(
            ent_all[h], T.goc.R(r), T.reader_tok, T.reader_mask, idx, chunk=T.reader_chunk
        )
    )
    target0 = torch.zeros(B, dtype=torch.long, device=sz.device)
    return (
        F.cross_entropy(s_sub + T.reader.lam * r_sub, target0, label_smoothing=smoothing),
        U,
        shortfall / max(1, n_hard),
    )


def mine_coanswer_groups(train, min_size: int = 2):
    """Co-answer groups: entities that are the HEAD of triples sharing (r, tail). Mined offline from train."""
    from collections import defaultdict

    g = defaultdict(set)
    for h, r, t in train:
        g[(r, t)].add(h)
    return [list(v) for v in g.values() if len(v) >= min_size]  # list[list[int]]


def l_repel(h_e, groups, n_groups: int, max_members: int, temp: float) -> torch.Tensor:
    """InfoNCE repulsion: each step samples n_groups groups and pushes the h_e of the members of a
    group apart (minimise logsumexp of off-diagonal similarity) so the scorer can tell the gold
    head apart inside a co-answer cluster."""
    if not groups:
        return torch.zeros((), device=h_e.device)
    gi = torch.randint(0, len(groups), (n_groups,))  # global RNG (set_seed already called)
    losses = []
    for j in gi.tolist():
        mem = groups[j]
        if len(mem) > max_members:
            sel = torch.randperm(len(mem))[:max_members].tolist()
            mem = [mem[k] for k in sel]
        if len(mem) < 2:
            continue
        m = torch.tensor(mem, device=h_e.device)
        H = F.normalize(h_e[m], dim=1)  # [k,d]
        S = (H @ H.t()) / temp  # [k,k]
        S.fill_diagonal_(float("-inf"))
        losses.append(torch.logsumexp(S, dim=1).mean())  # push each member away from its peers
    return torch.stack(losses).mean() if losses else torch.zeros((), device=h_e.device)


def l_align(e_s, t_e, ent_ids, has_text, temp: float) -> torch.Tensor:
    """In-batch InfoNCE pulling (t_e, e_s) of the same entity together; only entities with text."""
    ids = torch.unique(ent_ids)
    ids = ids[has_text[ids]]
    if ids.numel() < 2:
        return torch.zeros((), device=e_s.device)
    a = F.normalize(t_e[ids], dim=1)
    b = F.normalize(e_s[ids], dim=1)
    logits = (a @ b.t()) / temp  # [K,K]
    labels = torch.arange(ids.numel(), device=e_s.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2


def l_gate(g, prior_tier, has_image) -> torch.Tensor:
    """||g - prior_tier||^2 over entities with an image (gate anchor, prevents the image channel collapsing)."""
    m = has_image.bool()
    if m.sum() == 0:
        return torch.zeros((), device=g.device)
    return ((g[m] - prior_tier[m]) ** 2).mean()


def _enable_sl_capture(model, args) -> None:
    """Enable capture of the S/L score matrices for the NEXT `evaluate` call (valid and test)."""
    if hasattr(getattr(model, "tucker", None), "luu_SL"):
        model.tucker.luu_SL = {"S": [], "L": [], "S_max": [], "L_max": [], "huong": []}
        model.tucker.khoa_SL = []
        model.tucker.hai_chieu = bool(args.dump_SL_2c)


def _write_sl_dump(model, args, path: str, split_name: str) -> None:
    """Write the captured S/L matrices to `path`; `split_name` is 'valid' or 'test'.

    Shared verbatim by both splits so they are guaranteed to have the same format.
    """
    import scan_variants as SV

    _b = model.tucker.luu_SL
    _k = model.tucker.khoa_SL
    _S = torch.cat(_b["S"])
    assert (
        len(_k) == _S.shape[0] == len(_b["huong"])
    ), f"dump_SL[{split_name}]: keys {len(_k)} != rows {_S.shape[0]} != labels {len(_b['huong'])}"
    assert [x[3] for x in _k] == _b[
        "huong"
    ], f"dump_SL[{split_name}]: direction labels differ from keys"
    _g = {
        "split": split_name,
        "huong": "ca_hai" if args.dump_SL_2c else "head",
        "g_L_train": args.lsr_gL,
        "ghi_chu": "S,L da tru max moi query; diem goc = X + X_max. "
        "L luu THUA (L_idx/L_val/L_hang) — HU.giai_nen_L_thua(). "
        "keys = (h,r,t,huong) TUONG MINH tung hang.",
        "keys": _k,
        "S": _S,
        "S_max": torch.cat(_b["S_max"]),
        "L_max": torch.cat(_b["L_max"]),
        **SV.compress_sparse_lexical(torch.cat(_b["L"])),
    }
    torch.save(_g, path)
    print(
        f"[dump_SL] {split_name} -> {path} S{tuple(_S.shape)} "
        f"direction={_g['huong']} ({_b['huong'].count('head')} head / "
        f"{_b['huong'].count('tail')} tail)",
        flush=True,
    )
    model.tucker.luu_SL = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--t_path", required=True, help="T.pt [N x 768] PhoBERT")
    ap.add_argument(
        "--v_path",
        required=True,
        help="V.pt [N, k, 32] BEiT-v2 codebook tokens (k=3 for the three external datasets and "
        "ViProKG). Every V.pt in use is BEiT-v2; switching to CLIP would break instrument "
        "consistency with the three datasets and prevent reuse of the v1 tokens.",
    )
    ap.add_argument("--gate_mode", required=True, choices=GATE_MODES)
    ap.add_argument("--seed", type=int, required=True)  # required
    # switch modalities off for the no_image / structure_only configurations (int 0/1)
    ap.add_argument("--use_text", type=int, default=1, choices=[0, 1])
    ap.add_argument("--use_image", type=int, default=1, choices=[0, 1])
    ap.add_argument("--exp", default="vipronet")
    ap.add_argument("--num_epoch", type=int, default=NUM_EPOCH)
    ap.add_argument("--valid_epoch", type=int, default=VALID_EPOCH)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--lambda_align", type=float, default=LAMBDA_ALIGN)
    ap.add_argument("--lambda_gate", type=float, default=LAMBDA_GATE)
    ap.add_argument("--smoothing", type=float, default=0.1)
    ap.add_argument(
        "--lr_decay",
        type=float,
        default=1.0,
        help="ExponentialLR gamma per epoch (e.g. 0.995); 1.0 = off",
    )
    ap.add_argument(
        "--text_mode",
        default="mean",
        choices=["mean", "attn", "tokfuse"],
        help="mean | attn (attention pooling) | tokfuse (transformer token fusion)",
    )
    ap.add_argument(
        "--h_path",
        default="",
        help="token states .pt {H[N,k,768] fp16, mask[N,k]} for attn/tokfuse",
    )
    ap.add_argument(
        "--reciprocal",
        type=int,
        default=0,
        choices=[0, 1],
        help="learn separate inverse relations r^-1 (augment (t,r+R,h)); head prediction = tail prediction of r^-1",
    )
    ap.add_argument(
        "--entity_bias",
        type=int,
        default=0,
        choices=[0, 1],
        help="entity bias b_e in the scorer",
    )
    # name channel: syllable-level encoder of the plain entity label (not from the bio)
    ap.add_argument(
        "--use_name",
        type=int,
        default=0,
        choices=[0, 1],
        help="enable the name channel (4th fusion input)",
    )
    ap.add_argument(
        "--name_path",
        default="name_ids.pt",
        help="name channel: .pt {ids[N,L], mask[N,L], vocab}",
    )
    ap.add_argument("--dim_name", type=int, default=128, help="name channel: vector width")
    # second text channel (T_ft) alongside T_base
    ap.add_argument(
        "--dual_text",
        type=int,
        default=0,
        choices=[0, 1],
        help="two text channels [T_base;T_ft]",
    )
    ap.add_argument("--t2_path", default="", help="path of the second text .pt (T_ft)")
    # co-answer repulsion: push apart h_e of entities sharing (r, tail) so the scorer separates gold better
    ap.add_argument(
        "--lambda_repel", type=float, default=0.0, help="lambda_3 in {0.01,0.05}; 0 = off"
    )
    ap.add_argument("--repel_groups", type=int, default=64, help="groups sampled per step")
    ap.add_argument("--repel_max", type=int, default=32, help="max members per group per step")
    ap.add_argument(
        "--repel_temp", type=float, default=0.5, help="InfoNCE temperature of the repulsion"
    )
    ap.add_argument(
        "--dim_rel", type=int, default=DIM_REL
    )  # default 64; capacity ablation {64,256}
    ap.add_argument(
        "--dim",
        type=int,
        default=DIM_STR,
        help="structural embedding width `d` (default 256 = config.DIM_STR). "
        "Only exposes an existing ViProNet.__init__ parameter, adds no logic.",
    )
    ap.add_argument(
        "--corrupt_tier", type=int, default=0, choices=[0, 1]
    )  # shuffle tiers (corrupt-provenance control)
    ap.add_argument("--dump_ranks", default=None)
    # visual attribute slice (predict the tail within the value set of a relation) + mask new value nodes from gold
    ap.add_argument(
        "--v2_dir",
        default="",
        help="directory of the numeric attribute-slice package (v2_test_ids.txt, v2_cand.tsv, ...)",
    )
    ap.add_argument("--v2_out", default="", help=".json file receiving the attribute-slice results")
    ap.add_argument(
        "--v2_dump",
        default="",
        help=".tsv file receiving the PER-CASE ranks of the attribute slice",
    )
    ap.add_argument(
        "--max_ent_gold",
        type=int,
        default=0,
        help="rank gold test answers only among ids < this value (0 = no masking)",
    )
    ap.add_argument(
        "--dump_topk", default=None, help="write the top-5 predicted heads per test triple"
    )
    ap.add_argument("--dump_full", default=None, help="full TEST dump (rank/score_gold/top10)")
    ap.add_argument("--dump_full_valid", default=None, help="full VALIDATION dump (for dissection)")
    ap.add_argument(
        "--dump_scores_rel",
        default=None,
        help=".pt with the FULL scores of every candidate (valid, head)",
    )
    ap.add_argument(
        "--scores_rel_pid", default="", help="pid of the relation whose scores are dumped"
    )
    # every run has this option; enable it when post-hoc analysis is needed
    ap.add_argument(
        "--dump_scores_full",
        default=None,
        help=".pt with the full fp16 score matrix [n_query, N] on VALIDATION",
    )
    ap.add_argument(
        "--scores_full_direction",
        "--scores_full_huong",
        default="head",
        choices=["head", "tail", "ca_hai"],
    )
    # Full-score dump for TEST. `dump_scores_full` is only wired into the VALID evaluation branch
    # (see below), so there would otherwise be no way to get top-K + scores on test.
    ap.add_argument(
        "--dump_scores_full_test",
        default=None,
        help="like --dump_scores_full but for the TEST split (uses the same "
        "--scores_full_direction)",
    )
    # g_L used WHEN DUMPING scores (valid+test). Default None keeps the current g_L (trained at
    # --lsr_gL, default 0.15) and does not change the trajectory of earlier runs. It allows
    # dumping scores at g* (e.g. g*=0.25) while the model was TRAINED at the default 0.15: train at
    # 0.15, sweep to pick g*, test at g*. Only affects the dump pass, never training.
    ap.add_argument(
        "--scores_g",
        type=float,
        default=None,
        help="set model.tucker.g_L to this value right before dumping valid/test scores "
        "(restored afterwards). None = keep as is (no-op, earlier runs unchanged).",
    )
    # variant scan. "" = the original ViProNet (reference `text`).
    ap.add_argument(
        "--variant",
        "--huong",
        default="",
        help="T1 T2 S1 S2 L1 L2 H1 H2 H2T M1 (empty = original)",
    )
    ap.add_argument("--vis_token", default="", help="M1: .pt {tok[N,k] long, mask[N,k] bool}")
    # ===================== answer-name masking screening flags =====================
    # All OFF by default. With `--egen_mode none` every path below is skipped and the run is
    # bit-identical to the previous code (regression gate).
    ap.add_argument(
        "--egen_mode",
        default="none",
        choices=["none", "mask", "poe", "reweight", "shuffle"],
        help="Screening direction. none=off (default). "
        "mask=partner-name masking (pair with --text_path_override); "
        "poe=product-of-experts with frozen lexical bias; "
        "reweight=group importance weighting; "
        "shuffle=description shuffling as augmentation.",
    )
    ap.add_argument(
        "--egen_param",
        type=float,
        default=0.0,
        help="alpha for poe / beta for reweight / p for shuffle. "
        "0 must reproduce the baseline bit-for-bit.",
    )
    ap.add_argument(
        "--text_path_override",
        default="",
        help="Replaces --t_path at load time (used by --egen_mode mask with "
        "T_ctx_mask.pt). Recorded on the command line, which is what "
        "config.json stores.",
    )
    # ===== stress test: mask the answer name on TEST =====
    # Empty = OFF, the old path is bit-identical. When set: after loading best-valid, score TEST a
    # SECOND time in the SAME process with the masked text tensor. No checkpoint is saved/loaded
    # and nothing is retrained.
    # WARNING: this flag only swaps **T**. The **L** block reads its buffers from `L_feat.pt`
    # (the lexical block) and does NOT receive T, so L is NOT masked. For any configuration with
    # `g_L > 0` the measured drop is a LOWER BOUND. The code prints this warning on every run.
    ap.add_argument(
        "--t_path_masked_test",
        "--t_path_che_test",
        default="",
        help="T with the test answer name masked; empty = off",
    )
    ap.add_argument(
        "--dump_test_masked",
        "--dump_test_che",
        default="",
        help="TSV of per-key ranks of the MASKED test pass",
    )
    # Mask L as well: swap the lexical block's buffers inside the process and restore them in
    # `finally`. Empty = off = bit-identical.
    ap.add_argument(
        "--l_feat_masked_test",
        "--l_feat_che_test",
        default="",
        help="L_feat built on the MASKED text (build_masked_chain.py)",
    )
    # `--dump_test_masked` only writes the RANK at the g used for training, with no g* column,
    # while the downstream table scripts read **`sweep_gL_test_che.tsv`**, a file nothing used to
    # produce. Even when the masked pass ran, every row would report "ran, masked pass MISSING"
    # and all `rot_*` columns would be empty. The g grid must equal the original pass's
    # `--sweep_gL`.
    ap.add_argument(
        "--sweep_test_masked_out",
        "--sweep_test_che_out",
        default="",
        help="TSV of TEST ranks of the MASKED pass on the SAME --sweep_gL grid (name the file "
        "`sweep_gL_test_che.tsv`: that is the name the table scripts read)",
    )
    # ===================== Reader (late interaction) - all OFF by default =====================
    # With `--reader none` not one line below runs and the model is bit-identical to the baseline.
    ap.add_argument(
        "--reader",
        default="none",
        choices=["none", "che", "goc"],
        help="late interaction over candidate description tokens. "
        "`che` = masked text (rule E v2), `goc` = original text.",
    )
    ap.add_argument(
        "--reader_tok",
        default="",
        help=".pt from build_text_tokens.py {tok fp16 [N,L,768], mask bool [N,L], meta}",
    )
    ap.add_argument("--reader_K", type=int, default=8, choices=[1, 4, 8])
    ap.add_argument("--reader_pool", default="token", choices=["token", "mean"])
    ap.add_argument("--reader_dim", type=int, default=128)
    ap.add_argument("--reader_lam", type=float, default=0.3, help="lambda init (learned)")
    # 0 = OLD PATH (all N candidates, bit-identical to 1-vs-all). > 0: the Reader scores only
    # `1 + n_neg` candidates at train time; **eval still scores all N, unchanged**.
    ap.add_argument(
        "--reader_nneg",
        type=int,
        default=0,
        help="negatives per query at training (e.g. 255). 0 = all N as before",
    )
    ap.add_argument(
        "--reader_nkho",
        type=int,
        default=128,
        help="how many of them are HARD negatives taken from the detached top z(S)",
    )
    # Named `--reader_npool` because `--reader_pool` is already the pooling-type flag
    # (`token`/`mean`) and a duplicate would raise `argparse.ArgumentError`.
    ap.add_argument(
        "--reader_npool",
        type=int,
        default=2048,
        help="cap on |P|, the shared in-batch pool",
    )
    ap.add_argument(
        "--reader_chunk",
        type=int,
        default=512,
        help="candidates per chunk; the full [B*K, N*L] product cannot be held",
    )
    ap.add_argument(
        "--mention_edges",
        "--canh_nhac_ten",
        default="",
        help="name-mention edges .tsv (a\tb), built locally by build_lexical_data.py",
    )
    ap.add_argument("--bm25", default="", help=".npz {idx[N,K] int32, val[N,K] float32}")
    ap.add_argument("--lsr_L", default="", help="L_feat.pt")
    ap.add_argument(
        "--lsr_g_init",
        default="nen",
        choices=["nen", "deu"],
        help="router initialisation: nen = g~(1,0,0) | deu = g~(1/3,1/3,1/3)",
    )
    ap.add_argument(
        "--lsr_gL",
        type=float,
        default=0.15,
        help="FIXED weight of the L block when --lsr_off RM (trained at 0.15)",
    )
    ap.add_argument(
        "--lsr_l1",
        type=int,
        default=0,
        choices=[0, 1],
        help="use loss L1 (CE weighted by 1/freq^0.5) instead of plain CE",
    )
    ap.add_argument(
        "--fold_json",
        default="",
        help="folds_k3.json (make_oof_folds.py)",
    )
    ap.add_argument(
        "--fold_k",
        type=int,
        default=-1,
        help="DROP this fold from train (K-1 folds remain), then score that fold",
    )
    ap.add_argument(
        "--oof_out",
        default="",
        help=".pt of OUT-OF-FOLD scores for the triples of fold k, BOTH directions",
    )
    ap.add_argument(
        "--in_drop",
        type=float,
        default=-1.0,
        help="INPUT dropout of TuckER (code default 0.2). -1 = keep. "
        "Only exposes a parameter already in model.py, adds no logic.",
    )
    ap.add_argument(
        "--hid_drop",
        type=float,
        default=-1.0,
        help="INNER dropout of TuckER (code default 0.3). -1 = keep.",
    )
    ap.add_argument(
        "--structured_ls",
        "--ls_cau_truc",
        default="",
        choices=["", "deu", "Ar", "tuvung", "Ar+tuvung"],
        help="spread the label-smoothing mass over a STRUCTURED set instead of uniformly over N. "
        "`Ar` = answers ever seen for relation r (TRAIN) | `tuvung` = name-mention neighbours "
        "of gold (TEXT) | `Ar+tuvung` = eps/2 each. `deu` = standard PyTorch (reference). "
        "Gold is excluded from the smoothing set.",
    )
    ap.add_argument(
        "--multi_positive",
        "--da_duong",
        type=int,
        default=0,
        choices=[0, 1],
        help="softmax target = UNIFORM over N(h,r) of TRAIN (KvsAll) instead of one-hot. "
        "Label smoothing unchanged. Fixes the loss scoring 56-68 %% of the true answers as "
        "negatives. KvsAll already exists in LibKGE, so it is not a contribution.",
    )
    ap.add_argument(
        "--oof_topS",
        default="",
        help=".pt of the top-100 ids of S for ALL triples of fold k (both directions, int32 ids "
        "only, so light). The candidates of L at TRAIN time must be the OUT-OF-FOLD top-100, "
        "because S memorises train.",
    )
    ap.add_argument(
        "--lsr_L_hop",
        default="",
        help="L_feat_hop.pt: the L block scores the UNION(BM25 + top-S), looked up by QUERY "
        "(source, r_idx). Replaces --lsr_L for the L block; --lsr_L is STILL needed (for the "
        "has_BM25 flag in the router meta).",
    )
    ap.add_argument(
        "--topS_vt_out",
        default="",
        help=".pt of the top-K ids of S for **valid + test**, produced by the run trained on the "
        "FULL train set (eval uses the top-S of the full model)",
    )
    ap.add_argument(
        "--insample_out",
        default="",
        help="IN-SAMPLE scores as a control: scores a subsample of fold (fold_k+1)%%3, i.e. a "
        "fold that IS in train. Same format as oof.",
    )
    ap.add_argument(
        "--topS_k", type=int, default=100, help="number of top candidates of S to store"
    )
    ap.add_argument(
        "--oof_sample",
        type=int,
        default=0,
        help="score out-of-fold on a SUBSAMPLE of n triples of fold k "
        "(random.Random(--seed)). 0 = whole fold. Reason: a full-row dump of 3 folds x 3 "
        "datasets is 13.4 GB, more than the 8.2 GB disk.",
    )
    ap.add_argument(
        "--dump_SL_2c",
        type=int,
        default=0,
        choices=[0, 1],
        help="--dump_SL captures BOTH directions (default head only, which mismatches the "
        "two-direction evaluation scope). Adds (h,r,t,direction) keys.",
    )
    ap.add_argument(
        "--sweep_gL",
        default="",
        help="sweep g_L AT EVAL TIME, e.g. '0,0.05,0.15,0.6'. Writes one TSV: keys + one rank "
        "column per g. BOTH DIRECTIONS (dump_SL has head only, which mismatches the "
        "evaluation scope).",
    )
    ap.add_argument("--sweep_out", default="", help="TSV path of --sweep_gL")
    ap.add_argument(
        "--sweep_test_out",
        default="",
        help="TSV of TEST ranks for the same g grid. Cheaper than re-running a session to score "
        "at g* (chosen on valid). The collection step must read ONLY the g* column; the "
        "other columns go into no table.",
    )
    ap.add_argument(
        "--dump_SL",
        default="",
        help=".pt of the S and L matrices separately on valid/head (fp16, max subtracted)",
    )
    ap.add_argument(
        "--dump_check_test",
        "--dump_kiem_test",
        default="",
        help="TSV with three rows (ours / MyGO verbatim / optimistic) + fingerprint of the "
        "filter set, on TEST, at g = --lsr_gL. ONE dump serves the equivalence check of the "
        "ranking function, the filter-set check and the reverse tie-break check. "
        "READ-ONLY: changes no reported rank.",
    )
    ap.add_argument(
        "--dump_SL_test",
        default="",
        help=".pt of the S and L matrices on **TEST** (same format as --dump_SL). Needed to apply "
        "a gate to test: `--dump_SL` is valid only, and `--sweep_test_out` has ranks only, "
        "no scores.",
    )
    ap.add_argument(
        "--lsr_off",
        "--lsr_tat",
        default="",
        choices=["", "R", "L", "M", "RM"],
        help="ablation: switch off the router / lexical block / semantic block",
    )
    ap.add_argument(
        "--gen_pseudo",
        "--sinh_pseudo",
        default="",
        help="round 1: .tsv of pseudo-triples for entities with deg<=2 (margin from VARIANT_DEFAULTS)",
    )
    ap.add_argument(
        "--extra_train",
        "--them_train",
        default="",
        help="round 2: .tsv of pseudo-triples appended to TRAIN (train only, filters untouched)",
    )
    ap.add_argument("--patience", type=int, default=0, help="early stop (0 = off)")
    ap.add_argument(
        "--dump_gate", default=None, help="save the learned g (id,tier,g) for gate diagnostics"
    )
    args = ap.parse_args()

    set_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[MENTOR] gate={args.gate_mode} seed={args.seed} device={dev} "
        f"lambda1={args.lambda_align} lambda2={args.lambda_gate}",
        flush=True,
    )

    data = ViMMKGData(args.data_dir)
    # `mask` mode: load a different text tensor (partner names removed). The SAME file is used for
    # train and for scoring, on purpose: masking only at train time would score with a different
    # instrument than it trained with.
    _t_path = args.text_path_override or args.t_path
    if args.text_path_override:
        print(f"[screen] text override: {args.t_path} -> {_t_path}", flush=True)
    T, V = data.load_features(_t_path, args.v_path)
    T, V = T.to(dev), V.to(dev)
    has_text = data.has_text_mask(T).to(dev)
    T2 = None  # second text channel
    # Flag COMBINATION, not a lone flag: `--t2_path` alone does nothing. The tensor is only loaded
    # when `--dual_text 1`, and the concat branch of `model.encode` is also switched by
    # `dual_text`. The old code silently ignored `--t2_path`, so such a run reproduced the
    # baseline bit for bit with no flag raised. Both directions now stop.
    if args.t2_path and not args.dual_text:
        raise SystemExit(
            "`--t2_path` given without `--dual_text 1`: T2 would NOT be loaded and the concat "
            "branch would stay off, so this run would be bit-identical to the baseline. Add "
            "`--dual_text 1`."
        )
    if args.dual_text and not args.t2_path:
        raise SystemExit("`--dual_text 1` without `--t2_path`: no second text channel.")
    if args.dual_text:
        T2 = torch.load(args.t2_path, map_location="cpu").to(dev)
        # permanent shape guard: T2 must have the same number of entities and text width as T.
        assert T2.dim() == 2 and T2.shape[0] == T.shape[0] and T2.shape[1] == T.shape[1], (
            f"T2 {tuple(T2.shape)} != T {tuple(T.shape)}: the second text channel must have the same "
            f"(n_entity, dim_text); a row mismatch means a wrong entity mapping and every later "
            f"number is meaningless"
        )
        print(f"[dual_text] T2 {tuple(T2.shape)} <- {args.t2_path}", flush=True)
    # token states (attn/tokfuse); mean does not need them
    H = Hmask = None
    if args.text_mode != "mean":
        if not args.h_path:
            raise SystemExit(f"text_mode={args.text_mode} needs --h_path (token states .pt)")
        blob = torch.load(args.h_path, map_location="cpu")
        H = blob["H"].float().to(dev)
        Hmask = blob["mask"].bool().to(dev)
        print(
            f"[MENTOR] H {tuple(H.shape)} mask {tuple(Hmask.shape)} "
            f"(real tokens avg {float(Hmask.float().sum(1).mean()):.1f}/{H.shape[1]})",
            flush=True,
        )
    # name_ids/name_mask (every entity, [N,L]); vocab size to build the NameEncoder
    name_ids = name_mask = None
    name_vocab = 0
    if args.use_name:
        blob = torch.load(args.name_path, map_location="cpu")
        name_ids = blob["ids"].long().to(dev)
        name_mask = blob["mask"].bool().to(dev)
        name_vocab = int(blob["vocab"])
        assert name_ids.shape[0] == data.num_ent, f"name_ids {name_ids.shape} != N={data.num_ent}"
        n_named = int((name_mask.sum(1) > 0).sum())
        print(
            f"[name] name_ids {tuple(name_ids.shape)} vocab={name_vocab} "
            f"with-name {n_named}/{data.num_ent}",
            flush=True,
        )
    meta = {
        "has_image": data.has_image,
        "prior_tier": data.prior_tier,
        "tier": data.tier,
        "kind_onehot": data.kind_onehot,
        "log_nshare": data.log_nshare,
    }
    if args.corrupt_tier:
        # Corrupt-provenance control: shuffle provenance labels among entities WITH an image
        # (distribution kept, entity<->tier link destroyed). If gate_prior_init drops to the blind
        # level, the gate is using the REAL signal.
        idx = data.has_image.nonzero(as_tuple=True)[0]
        gen = torch.Generator().manual_seed(args.seed)
        perm = idx[torch.randperm(idx.numel(), generator=gen)]
        for k in ["prior_tier", "tier", "kind_onehot", "log_nshare"]:
            v = meta[k].clone()
            v[idx] = meta[k][perm]
            meta[k] = v
        print(
            "[corrupt_tier] shuffled tier/provenance among entities with images (seed %d)"
            % args.seed,
            flush=True,
        )
    # Round 2: append pseudo-triples to TRAIN. valid/test and the filter dicts are untouched
    # (filters are built from the original train+valid+test in data.py; pseudo-triples are only
    # extra training data).
    if args.extra_train:
        _pseudo = [
            tuple(int(x) for x in l.split("\t")[:3])
            for l in open(args.extra_train, encoding="utf-8").read().splitlines()[1:]
            if l.strip()
        ]
        print(
            f"[pseudo] appending {len(_pseudo)} pseudo-triples to train ({len(data.train)} -> "
            f"{len(data.train) + len(_pseudo)})",
            flush=True,
        )
        data.train = list(data.train) + _pseudo

    # FOLD FILTER: must happen BEFORE the reciprocal augmentation, otherwise every triple is
    # already doubled and dropping by index would drop one direction only. `filter_dict` is NOT
    # changed: it is built from the FULL train + valid + test in data.py, so ranks stay comparable
    # with all earlier tables.
    _fold_triples = None
    if args.fold_json and args.fold_k >= 0:
        _fj = json.load(open(args.fold_json, encoding="utf-8"))
        assert len(_fj["fold"]) == len(
            data.train
        ), f"folds json has {len(_fj['fold'])} entries, train has {len(data.train)}"
        _fj_train = list(
            data.train
        )  # ORIGINAL train (before filtering) to fetch the in-sample fold
        _kept, _fold_triples = [], []
        for _i, _t3 in enumerate(data.train):
            (_kept if _fj["fold"][_i] != args.fold_k else _fold_triples).append(_t3)
        assert len(_kept) + len(_fold_triples) == len(data.train), "fold filter lost triples"
        print(
            f"[fold] fold k={args.fold_k}: train {len(data.train)} -> {len(_kept)} "
            f"(keeping {len(_fold_triples)} triples to score out-of-fold)",
            flush=True,
        )
        data.train = _kept
        # fold (k+1)%K is still IN train: score it as an IN-SAMPLE control
        _k_in = (args.fold_k + 1) % int(_fj["k"])
        _in_triples = [t for i, t in enumerate(_fj_train) if _fj["fold"][i] == _k_in]
        _fold_triples_full = list(_fold_triples)  # top-S needs the WHOLE fold k; the
        # subsample only applies to the (heavy) S/L dump
        if args.oof_sample and args.oof_sample < len(_fold_triples):
            import random as _rd

            _idx = sorted(_rd.Random(args.seed).sample(range(len(_fold_triples)), args.oof_sample))
            _fold_triples = [_fold_triples[_i] for _i in _idx]
            print(
                f"[fold] out-of-fold subsample: {args.oof_sample} triples "
                f"(random.Random({args.seed}).sample)",
                flush=True,
            )
        if args.insample_out and args.oof_sample and args.oof_sample < len(_in_triples):
            import random as _rd2

            _ix2 = sorted(
                _rd2.Random(args.seed + 1).sample(range(len(_in_triples)), args.oof_sample)
            )
            _in_triples = [_in_triples[_i] for _i in _ix2]
        if args.insample_out:
            print(
                f"[fold] in-sample: fold k={_k_in} (IN train), "
                f"{len(_in_triples)} triples, control for the out-of-fold scores",
                flush=True,
            )

    # Reciprocal mode: double the relations, augment (t, r+R, h); head prediction becomes tail
    # prediction of r^-1
    recip = bool(args.reciprocal)
    R0 = data.num_rel
    if recip:
        train_list = data.train + [(t, r + R0, h) for (h, r, t) in data.train]
        num_rel_eff = 2 * R0
        print(
            f"[MENTOR] RECIPROCAL: rel {R0}->{num_rel_eff}, train {len(data.train)}->{len(train_list)}",
            flush=True,
        )
    else:
        train_list = data.train
        num_rel_eff = R0
    # Lookup A = N(h, r) for the multi-positive target. Built from `train_list`, i.e. TRAIN (fold
    # filtered if any) AFTER the reciprocal augmentation, so the (h, r) keys cover BOTH
    # directions. Never reads valid/test.
    ANSWERS_BY_HR = None
    if args.multi_positive:
        from collections import defaultdict as _dd

        _m = _dd(list)
        for _h, _r, _t in train_list:
            _m[(_h, _r)].append(_t)
        ANSWERS_BY_HR = dict(_m)
        _sz = sum(len(v) for v in ANSWERS_BY_HR.values()) / len(ANSWERS_BY_HR)
        _nh = sum(1 for v in ANSWERS_BY_HR.values() if len(v) > 1)
        print(
            f"[multi_positive] {len(ANSWERS_BY_HR)} (h,r) keys | mean |A| {_sz:.2f} | "
            f"{_nh/len(ANSWERS_BY_HR):.1%} keys with |A|>1 | max |A| "
            f"{max(len(v) for v in ANSWERS_BY_HR.values())}",
            flush=True,
        )

    # Structured smoothing set. A(r) from TRAIN (after reciprocal, so the direction is right);
    # name-mention neighbours from TEXT. Both only enter the loss TARGET, never the scoring function.
    LS = None
    if args.structured_ls:
        from collections import defaultdict as _dd2

        _ar = _dd2(set)
        for _h, _r, _t in train_list:
            _ar[_r].add(_t)
        _A_r = {k: sorted(v) for k, v in _ar.items()}
        _NB = {}
        if args.structured_ls in ("tuvung", "Ar+tuvung"):
            if not args.mention_edges:
                raise SystemExit("--structured_ls tuvung needs --mention_edges")
            _nb = _dd2(set)
            with open(args.mention_edges, encoding="utf-8") as _f:
                next(_f)
                for _l in _f:
                    if _l.strip():
                        _a, _b2 = _l.split("\t")[:2]
                        _nb[int(_a)].add(int(_b2))
                        _nb[int(_b2)].add(int(_a))
            _NB = {k: sorted(v) for k, v in _nb.items()}
        LS = {"che_do": args.structured_ls, "A_r": _A_r, "NB": _NB, "n_fallback": 0}
        print(
            f"[structured_ls] structured_ls={args.structured_ls} | mean |A(r)| "
            f"{sum(len(v) for v in _A_r.values())/max(len(_A_r),1):.1f} "
            f"(max {max((len(v) for v in _A_r.values()), default=0)}) | "
            f"name-mention neighbours: {len(_NB)} entities, mean |NB| "
            f"{sum(len(v) for v in _NB.values())/max(len(_NB),1):.1f}",
            flush=True,
        )

    model = ViProNet(
        data.num_ent,
        num_rel_eff,
        meta,
        args.gate_mode,
        dim=args.dim,
        dim_rel=args.dim_rel,
        text_mode=args.text_mode,
        entity_bias=bool(args.entity_bias),
        use_name=bool(args.use_name),
        name_vocab=name_vocab,
        dim_name=args.dim_name,
        dual_text=bool(args.dual_text),
    ).to(dev)
    # Expose the TuckER dropouts (parameters that already exist in model.py; only the value of
    # the two built `nn.Dropout` modules changes, no new logic). -1 = keep default 0.2 / 0.3.
    if args.in_drop >= 0 or args.hid_drop >= 0:
        if args.in_drop >= 0:
            model.tucker.in_drop.p = args.in_drop
        if args.hid_drop >= 0:
            model.tucker.hid_drop.p = args.hid_drop
        print(
            f"[dropout] TuckER dropout: in={model.tucker.in_drop.p} "
            f"hid={model.tucker.hid_drop.p}",
            flush=True,
        )

    prior_tier = meta["prior_tier"].to(
        dev
    )  # prior (possibly corrupted): L_gate is anchored consistently with the gate
    has_image = data.has_image.float().to(dev)

    train = torch.tensor(train_list, dtype=torch.long)
    repel_groups = mine_coanswer_groups(data.train) if args.lambda_repel > 0 else None
    if repel_groups is not None:
        sizes = [len(g) for g in repel_groups]
        print(
            f"[MENTOR] repel: {len(repel_groups)} co-answer groups (size>=2, "
            f"max {max(sizes)}, mean {sum(sizes)/len(sizes):.1f}) lambda3={args.lambda_repel}",
            flush=True,
        )
    # ---------------- plug in a variant (wraps encode/tucker, so eval uses it without touching evaluate.py)
    VARIANT, aux = args.variant, {}
    if VARIANT == "G1":
        if not args.mention_edges:
            raise SystemExit("G1 needs --mention_edges (built by build_lexical_data.py)")
        data.canh_nhac_ten = [
            tuple(int(x) for x in l.split("\t"))
            for l in open(args.mention_edges, encoding="utf-8").read().splitlines()[1:]
            if l.strip()
        ]
    if VARIANT:
        import scan_variants as SV

        assert (
            VARIANT in SV.VARIANT_CODES
        ), f"--variant {VARIANT} is invalid; choose {SV.VARIANT_CODES}"
        aux = SV.build_aux_data(VARIANT, data, dev)
        if VARIANT == "LSR" and args.lsr_l1:
            aux.update(SV.build_aux_data("L1", data, dev))  # needs `freq` for loss L1
        print(f"[variant] {VARIANT} | paper defaults: {SV.VARIANT_DEFAULTS[VARIANT]}", flush=True)
        if VARIANT == "S1":
            model.them_s1 = SV.NeighborEncoder(
                data.num_ent, data.num_rel, model.dim, SV.VARIANT_DEFAULTS["S1"]["so_lang_gieng"]
            ).to(dev)
            _orig_encode = model.encode

            def _enc_s1(*a_, **k_):
                e = _orig_encode(*a_, **k_)
                e["h_e"] = model.them_s1(
                    e["h_e"], aux["nb_ent"], aux["nb_rel"], aux["nb_mask"], aux["degree_bin"]
                )
                return e

            model.encode = _enc_s1
        elif VARIANT in ("S2", "G1"):
            # G1 = S2 but the graph also gets name-mention edges, so +1 relation pair (r_men, r_men^-1)
            _nr = data.num_rel + (1 if VARIANT == "G1" else 0)
            model.them_s2 = SV.CompGCN(
                _nr,
                model.dim,
                SV.VARIANT_DEFAULTS[VARIANT]["so_lop"],
                SV.VARIANT_DEFAULTS[VARIANT]["drop"],
            ).to(dev)
            _orig_encode = model.encode

            def _enc_s2(*a_, **k_):  # noqa: E306
                e = _orig_encode(*a_, **k_)
                e["h_e"] = model.them_s2(e["h_e"], aux["src"], aux["dst"], aux["rel"], aux["deg"])
                return e

            model.encode = _enc_s2
        elif VARIANT == "M1":
            blob = torch.load(args.vis_token, map_location="cpu")
            tok, mk = blob["tok"].to(dev), blob["mask"].to(dev)
            n_codes = int(blob.get("so_ma", int(tok.max()) + 1))
            model.them_m1 = SV.ImageTokenEncoder(
                n_codes,
                DIM_VIS,
                SV.VARIANT_DEFAULTS["M1"]["so_token"],
                SV.VARIANT_DEFAULTS["M1"]["so_lop"],
                SV.VARIANT_DEFAULTS["M1"]["so_head"],
            ).to(dev)
            _orig_encode = model.encode

            def _enc_m1(T_, V_, **k_):
                return _orig_encode(
                    T_, model.them_m1(tok, mk), **k_
                )  # V replaced by the transformer output

            model.encode = _enc_m1
        elif VARIANT in ("H2", "H2T"):
            model.them_h2 = SV.TextHead(DIM_TXT, data.num_rel).to(dev)
            model.tucker = SV.TuckerTextAdd(
                model.tucker, model.them_h2, T, data.num_rel, text_only=(VARIANT == "H2T")
            ).to(dev)
        elif VARIANT == "LSR":
            if not args.lsr_L:
                raise SystemExit("LSR needs --lsr_L (lexical_channel_features.py)")
            _b = torch.load(args.lsr_L, map_location="cpu")
            if args.lsr_L_hop:  # candidates = UNION(BM25 + top-S)
                _bh = torch.load(args.lsr_L_hop, map_location="cpu")
                assert _bh["feat"].shape[-1] == 10, "L_feat_hop must have 10 features"
                assert int(_bh["N"]) == data.num_ent and int(_bh["R"]) == data.num_rel, (
                    f"L_feat_hop: N={_bh['N']}/R={_bh['R']} differs from the data "
                    f"N={data.num_ent}/R={data.num_rel}"
                )
                model.lsr_L = SV.LexicalUnionBlock(
                    _bh["tra"].to(dev),
                    _bh["idx"].to(dev),
                    _bh["feat"].to(dev),
                    data.num_rel,
                    SV.VARIANT_DEFAULTS["LSR"]["dim_L"],
                ).to(dev)
                _d = _bh.get("dem_nguon", {})
                _tg = max(sum(_d.values()), 1)
                print(
                    f"[lexical] L block = UNION | {_bh['n_truy_van']} queries | K={_bh['K']} | "
                    f"source: BM25 only {_d.get('chi_bm25',0)/_tg:.1%} | "
                    f"S only {_d.get('chi_S',0)/_tg:.1%} | both {_d.get('ca_hai',0)/_tg:.1%}",
                    flush=True,
                )
            else:
                model.lsr_L = SV.LexicalBlock(
                    _b["idx"].to(dev),
                    _b["feat"].to(dev),
                    data.num_rel,
                    SV.VARIANT_DEFAULTS["LSR"]["dim_L"],
                ).to(dev)
            model.lsr_M = SV.TextPromptHead(
                DIM_TXT, data.num_rel, SV.VARIANT_DEFAULTS["X1"]["dim"]
            ).to(dev)
            model.lsr_R = SV.GateRouter(
                data.num_rel, SV.VARIANT_DEFAULTS["LSR"]["dim_R"], init_mode=args.lsr_g_init
            ).to(dev)
            # meta[N,5] = [log1p(deg), log1p(mean |A|), has_text, len_text/512, has_BM25]
            _deg = torch.zeros(data.num_ent)
            for _h, _r, _t in data.train:
                _deg[_h] += 1
                _deg[_t] += 1
            _ct = (T.abs().sum(1) > 0).float()
            _cb = (_b["feat"][:, :, 0].float().sum(1) > 0).float().to(dev)
            _meta = torch.stack(
                [
                    torch.log1p(_deg).to(dev),
                    torch.log1p(_deg).to(dev) * 0,
                    _ct.to(dev),
                    (_ct * 0.5).to(dev),
                    _cb,
                ],
                dim=-1,
            )
            model.tucker = SV.TuckerLexicalFusion(
                model.tucker,
                model.lsr_L,
                model.lsr_M,
                T,
                model.lsr_R,
                _meta,
                data.num_rel,
                tat=args.lsr_off,
            ).to(dev)
            model.tucker.g_L = args.lsr_gL
            _t = args.lsr_off  # explicit label following the logic of TuckerLexicalFusion._cham
            _kb = "S"
            if _t != "L":
                _kb += " + L"
            if _t not in ("M", "RM"):
                _kb += " + M"
            if _t == "RM":
                _kb += f" | g FIXED (1, {args.lsr_gL}, 0), NO R"
            elif _t == "R":
                _kb += " | g FIXED (0.5, 0.25, 0.25), NO R"
            else:
                _kb += " + R learned"
            print(f"[LSR] active blocks: {_kb}", flush=True)
        elif VARIANT == "X1":
            model.them_x1 = SV.TextPromptHead(
                DIM_TXT, data.num_rel, SV.VARIANT_DEFAULTS["X1"]["dim"]
            ).to(dev)
            model.tucker = SV.TuckerTextAdd(model.tucker, model.them_x1, T, data.num_rel).to(dev)
        elif VARIANT == "X2":
            if not args.bm25:
                raise SystemExit("X2 needs --bm25 (built by build_lexical_data.py)")
            import numpy as _np

            _b = _np.load(args.bm25)
            model.them_x2 = SV.BM25Head(
                torch.tensor(_b["idx"]).to(dev),
                torch.tensor(_b["val"]).to(dev),
                SV.VARIANT_DEFAULTS["X2"]["gamma_init"],
                num_ent=data.num_ent,
            ).to(dev)
            model.tucker = SV.TuckerTextAdd(model.tucker, model.them_x2, T, data.num_rel).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    if VARIANT == "H1":
        import scan_variants as SV

        sched = SV.cosine_schedule_h1(opt, args.num_epoch)
        print(f"[variant] H1: cosine T_max={args.num_epoch} | patience={args.patience}", flush=True)
    else:
        sched = None
    sched = (
        sched
        if VARIANT == "H1"
        else (
            torch.optim.lr_scheduler.ExponentialLR(opt, gamma=args.lr_decay)
            if args.lr_decay < 1.0
            else None
        )
    )  # lr decay per epoch
    ut, ui = bool(args.use_text), bool(args.use_image)
    # L_gate only applies to init-prior with an image channel; L_align only with a text channel
    use_gate_loss = (args.gate_mode == "gate_prior_init") and args.lambda_gate > 0 and ui
    use_align = args.lambda_align > 0 and ut
    print(
        f"  use_text={ut} use_image={ui} | L_align={use_align} L_gate={use_gate_loss}", flush=True
    )

    # ======================= screening setup (no-op when --egen_mode none) =======================
    EG = args.egen_mode
    eg_edges = None  # set of frozenset({h, t}) carrying a name-mention edge
    eg_n_swapped = 0  # shuffle: how many rows were actually swapped
    eg_n_batch = 0
    if EG in ("reweight", "shuffle"):
        # "mention pair" = the PAIR (h, t) has a name-mention edge, BOTH directions. Read from the
        # prebuilt file; never recomputed here.
        if not args.mention_edges:
            raise SystemExit(
                f"--egen_mode {EG} needs --mention_edges (name-mention edge file of the dataset)"
            )
        with open(args.mention_edges, encoding="utf-8") as _f:
            eg_edges = {
                frozenset((int(a), int(b)))
                for a, b in (l.split("\t") for l in _f.read().splitlines()[1:] if l)
            }
        _tr = train.tolist()
        _n_kc = sum(1 for a, _r, b in _tr if frozenset((a, b)) not in eg_edges)
        print(
            f"[screen {EG}] mention edges {len(eg_edges)} pairs | train without mention edge "
            f"{_n_kc}/{len(_tr)} = {_n_kc / max(1, len(_tr)):.4f}",
            flush=True,
        )
    # ===== Reader setup. Built LAZILY so `--reader none` draws no RNG (see scan_variants.py) =====
    if args.reader != "none":
        if VARIANT != "LSR":
            raise SystemExit("--reader needs --variant LSR (the Reader wraps TuckerLexicalFusion)")
        if not args.reader_tok:
            raise SystemExit("--reader needs --reader_tok (produced by build_text_tokens.py)")
        _rt = torch.load(args.reader_tok, map_location="cpu", weights_only=False)
        _mt = _rt.get("meta", {})
        if _mt.get("bien") != args.reader:
            raise SystemExit(
                f"--reader {args.reader} but the file is variant {_mt.get('bien')!r}. "
                f"Using the wrong variant measures something other than intended."
            )
        _tk = _rt["tok"].to(dev)
        _mk = _rt["mask"].to(dev)
        # `dim_ent` = width of `ent_all` (= `encode()["h_e"]`), i.e. the width of `vec` that `_cham`
        # passes to the Reader. MEASURED with one no-grad encode instead of assuming `args.dim`
        # (call the consuming function itself, do not trust docs).
        # The probe must be RNG-NEUTRAL: `encode` has dropout, and in train mode it DRAWS RNG and
        # would shift the whole random sequence of the run. Save/restore the state and use eval()
        # so the measurement does not change the training trajectory.
        _rng = torch.get_rng_state()
        _rng_c = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        _was_training = model.training
        model.eval()
        with torch.no_grad():
            _de = int(
                model.encode(T, V, use_text=args.use_text, use_image=args.use_image)["h_e"].shape[1]
            )
        if _was_training:
            model.train()
        torch.set_rng_state(_rng)
        if _rng_c is not None:
            torch.cuda.set_rng_state_all(_rng_c)
        if _de != args.dim:
            print(
                f"[Reader] warning: ent_all width {_de} != --dim {args.dim}; using {_de}",
                flush=True,
            )
        _rd = model.tucker.enable_reader(
            _tk,
            _mk,
            dim_ent=_de,
            K=args.reader_K,
            d_proj=args.reader_dim,
            pool=args.reader_pool,
            lam=args.reader_lam,
            chunk=args.reader_chunk,
        )
        opt.add_param_group({"params": list(_rd.parameters())})
        print(
            f"[Reader] variant={args.reader} dim_ent={_de} K={args.reader_K} pool={args.reader_pool} "
            f"d={args.reader_dim} lam0={args.reader_lam} chunk={args.reader_chunk} | "
            f"tok {tuple(_tk.shape)} | sha_textual {_mt.get('sha_textual')} | "
            f"new parameters {sum(p.numel() for p in _rd.parameters())}",
            flush=True,
        )

    if EG == "poe":
        if VARIANT != "LSR":
            raise SystemExit(
                "--egen_mode poe needs --variant LSR (takes L from TuckerLexicalFusion)"
            )
        model.tucker.poe = (args.egen_param, 1.0)  # tau = 1
        print(
            f"[screen poe] alpha={args.egen_param} tau=1.0 - train logits = "
            "log_softmax(S) + alpha*log_softmax(L.detach()); eval UNCHANGED",
            flush=True,
        )

    # The model is selected by VALID MRR; TEST runs EXACTLY ONCE at the end on best-valid.
    start = time.time()
    best_valid, best_state = 0.0, None
    _first_batch = True  # print the subset info ONCE, at the first batch
    for epoch in range(1, args.num_epoch + 1):
        model.train()
        perm = torch.randperm(train.shape[0])
        tot = 0.0
        for s in range(0, train.shape[0], args.batch_size):
            idx = perm[s : s + args.batch_size]
            h = train[idx, 0].to(dev)
            r = train[idx, 1].to(dev)
            t = train[idx, 2].to(dev)
            # --- weights / augmentation (both no-ops when --egen_mode none) ---
            eg_w = None
            T_ep = T
            if EG in ("reweight", "shuffle"):
                _kc = torch.tensor(
                    [
                        frozenset((int(a), int(b))) not in eg_edges
                        for a, b in zip(h.tolist(), t.tolist())
                    ],
                    device=dev,
                    dtype=torch.bool,
                )
            if EG == "reweight":
                # w_i = 1 + beta * [no mention edge], renormalised so mean(w) = 1 per batch,
                # which keeps the loss scale (and therefore the lr) comparable to baseline.
                w = 1.0 + args.egen_param * _kc.float()
                eg_w = w / w.mean().clamp(min=1e-8)
            elif EG == "shuffle" and args.egen_param > 0:
                # With probability p, a query WITH a mention edge gets the description row of
                # another entity drawn from the same batch (in-batch permutation). Train only.
                # The `args.egen_param > 0` guard is NOT cosmetic: calling torch.rand/randperm at
                # p = 0 still CONSUMES RNG state, so the later torch.randperm(train) draws differ
                # and the run stops being bit-identical to the baseline.
                _co = ~_kc
                _hit = (torch.rand(h.shape[0], device=dev) < args.egen_param) & _co
                if _hit.any():
                    _src = h[torch.randperm(h.shape[0], device=dev)]
                    T_ep = T.clone()
                    T_ep[h[_hit]] = T[_src[_hit]]
                eg_n_swapped += int(_hit.sum())
                eg_n_batch += int(_co.sum())
            enc = model.encode(
                T_ep,
                V,
                H=H,
                Hmask=Hmask,
                use_text=ut,
                use_image=ui,
                name_ids=name_ids,
                name_mask=name_mask,
                T2=T2,
            )
            if VARIANT in ("H2", "H2T", "X1", "X2", "LSR"):
                model.tucker.dat_chi_so(h, t)
            # WRAPS the base loss branch, does not replace any branch. `reader_skip` makes `_cham`
            # skip the Reader term on the full-N pass and KEEP the S-mix; `CE_sub` adds it below.
            # So `CE_full` is still exactly the baseline loss (L1 1/freq^0.5). (A separate
            # tail-only sampled loss plugged into the `else` branch would be bypassed whenever
            # `--lsr_l1 1` selects the `loss_freq_weighted` branch.)
            _nneg = args.reader_nneg if (args.reader != "none" and args.reader_nneg > 0) else 0
            if _nneg and not recip:
                raise SystemExit("--reader_nneg needs --reciprocal 1 (tail-only path)")
            if _nneg:
                model.tucker.reader_skip = True
            if VARIANT == "L1":
                import scan_variants as SV

                loss = SV.loss_freq_weighted(
                    model,
                    enc["h_e"],
                    h,
                    r,
                    t,
                    args.smoothing,
                    aux["freq"],
                    SV.VARIANT_DEFAULTS["L1"]["mu"],
                )
            elif VARIANT == "LSR" and args.lsr_l1:
                import scan_variants as SV

                loss = SV.loss_freq_weighted(
                    model,
                    enc["h_e"],
                    h,
                    r,
                    t,
                    args.smoothing,
                    aux["freq"],
                    SV.VARIANT_DEFAULTS["L1"]["mu"],
                    tail_only=recip,
                    answers_lookup=ANSWERS_BY_HR,
                    ls=LS,
                    w_extra=eg_w,
                )
            elif VARIANT == "L3":
                import scan_variants as SV

                loss = SV.loss_bce_all_tails(
                    model, enc["h_e"], h, r, t, aux["nhan_mt"], args.smoothing
                )
            elif VARIANT == "L2":
                import scan_variants as SV

                loss = SV.loss_hard_negatives(
                    model,
                    enc["h_e"],
                    h,
                    r,
                    t,
                    args.smoothing,
                    SV.VARIANT_DEFAULTS["L2"]["so_hard"],
                    SV.VARIANT_DEFAULTS["L2"]["tau"],
                )
            else:
                loss = (
                    l_rank_tail(model, enc["h_e"], h, r, t, args.smoothing)
                    if recip
                    else l_rank(model, enc["h_e"], h, r, t, args.smoothing)
                )
            if use_align:
                loss = loss + args.lambda_align * l_align(
                    enc["e_s"], enc["t_e"], torch.cat([h, t]), has_text, ALIGN_TEMP
                )
            if _nneg:
                model.tucker.reader_skip = False
                _l_sub, _U, _fill_ratio = sampled_ce_shared_pool(
                    model.tucker,
                    enc["h_e"],
                    h,
                    r,
                    t,
                    args.smoothing,
                    _nneg,
                    args.reader_nkho,
                    args.reader_npool,
                )
                loss = loss + _l_sub
                if _first_batch:
                    print(
                        f"[nneg] in-batch shared negatives | M={1 + _nneg} "
                        f"(hard {args.reader_nkho}) | |P| = {_U} / cap {args.reader_npool} "
                        f"(N={enc['h_e'].shape[0]}) | random fill ratio for hard negatives "
                        f"{_fill_ratio:.1%} | `W_tok` runs on P only",
                        flush=True,
                    )
                    _first_batch = False
            if use_gate_loss:
                loss = loss + args.lambda_gate * l_gate(enc["g"], prior_tier, has_image)
            if args.lambda_repel > 0:  # co-answer repulsion
                loss = loss + args.lambda_repel * l_repel(
                    enc["h_e"], repel_groups, args.repel_groups, args.repel_max, args.repel_temp
                )
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        if sched is not None:
            sched.step()  # lr decay after each epoch
        if EG == "shuffle" and (epoch == 1 or epoch % 10 == 0):
            print(
                f"[screen shuffle] epoch {epoch}: swapped {eg_n_swapped}/{eg_n_batch} rows WITH "
                f"mention edge = {eg_n_swapped / max(1, eg_n_batch):.4f} (declared p = {args.egen_param})",
                flush=True,
            )
        if epoch % 10 == 0 or epoch == 1 or args.valid_epoch == 1:
            # With a fine epoch grid, print the train loss EVERY epoch, to tell "valid drops
            # because of overfitting" from "valid drops because optimisation goes the wrong way".
            print(f"epoch {epoch}\tloss {tot:.4f}\t{time.time()-start:.0f}s", flush=True)
        if epoch % args.valid_epoch == 0:
            m = evaluate(
                model,
                T,
                V,
                data,
                "valid",
                use_text=ut,
                use_image=ui,
                H=H,
                Hmask=Hmask,
                recip_offset=(R0 if recip else 0),
                name_ids=name_ids,
                name_mask=name_mask,
                T2=T2,
            )
            flag = ""
            if m["MRR"] > best_valid:
                best_valid = m["MRR"]
                best_state = copy.deepcopy(model.state_dict())  # only stored; test untouched
                flag = " *best"
            print(
                f"  [valid e{epoch}] MRR {m['MRR']:.4f} H@1 {m['Hits@1']:.4f} "
                f"H@10 {m['Hits@10']:.4f}{flag}",
                flush=True,
            )
            _g = getattr(getattr(model, "tucker", None), "g_tb", None)
            if _g is not None:
                print(
                    f"  [LSR g e{epoch}] S {float(_g[:, 0].mean()):.4f} | "
                    f"L {float(_g[:, 1].mean()):.4f} | M {float(_g[:, 2].mean()):.4f}",
                    flush=True,
                )

    # load best-valid, then TEST once + dump ranks
    if best_state is not None:
        model.load_state_dict(best_state)
    # report the beta of the H2 text head (read from best_state, changes nothing)
    _head = getattr(getattr(model, "tucker", None), "dau", None)
    if _head is not None and hasattr(_head, "gamma"):
        print(f"  [X2] gamma_learned {float(_head.gamma.detach()):+.6f} (init 0.1)", flush=True)
    if _head is not None and hasattr(_head, "beta"):
        import scan_variants as SV

        _b = float(_head.beta.detach())
        # the relation table has a different name per head: H2/H2T use `Wr`, X1 uses `Er`
        _emb = getattr(_head, "Wr", None) or getattr(_head, "Er", None)
        _wr = float(_emb.weight.detach().norm(dim=-1).mean()) if _emb is not None else float("nan")
        print(
            f"  [{VARIANT or 'H2'}] beta_learned {_b:+.6f} "
            f"(init {SV.VARIANT_DEFAULTS.get(VARIANT or 'H2', {}).get('beta_init', 0.1)}) "
            f"· mean |relation emb| {_wr:.4f}",
            flush=True,
        )
    # Round 1: generate pseudo-triples. Queries are TAKEN FROM TRAIN (valid/test untouched).
    if args.gen_pseudo:
        from collections import defaultdict as _dd

        import scan_variants as SV

        mg = SV.VARIANT_DEFAULTS["P1"]["margin"]
        deg_ = _dd(int)
        tails_of = _dd(list)  # (h, r) -> tails already in TRAIN
        for _h, _r, _t in data.train:
            deg_[_h] += 1
            deg_[_t] += 1
            tails_of[(_h, _r)].append(_t)
        # pre-registered: only (h, r) pairs whose **h is sparse** (deg <= 2)
        pairs = sorted({(a, b) for a, b, _c in data.train if deg_[a] <= 2})
        model.eval()
        pseudo_out, n_skipped = [], 0
        with torch.no_grad():
            enc = model.encode(
                T,
                V,
                H=H,
                Hmask=Hmask,
                use_text=ut,
                use_image=ui,
                name_ids=name_ids,
                name_mask=name_mask,
                T2=T2,
            )
            ea = enc["h_e"]
            for s_ in range(0, len(pairs), 256):
                ck = pairs[s_ : s_ + 256]
                hb = torch.tensor([a for a, _ in ck], device=dev)
                rb = torch.tensor([b for _, b in ck], device=dev)
                if hasattr(model.tucker, "dat_chi_so"):
                    model.tucker.dat_chi_so(hb, hb)
                sc = model.tucker.score_tail(ea[hb], rb, ea)
                for i_, (a, b) in enumerate(ck):
                    v = sc[i_].clone()
                    known_ = tails_of.get((a, b))
                    if known_:
                        v[torch.tensor(known_, device=dev)] = float("-inf")  # drop known tails
                    top2 = torch.topk(v, 2)
                    if float(top2.values[0] - top2.values[1]) < mg:
                        n_skipped += 1
                        continue
                    pseudo_out.append((a, b, int(top2.indices[0])))  # max 1 per pair
        with open(args.gen_pseudo, "w", encoding="utf-8") as f:
            f.write("h\tr\tt\n")
            for a, b, c in pseudo_out:
                f.write(f"{a}\t{b}\t{c}\n")
        _vt = {(x, y, z) for x, y, z in list(data.valid) + list(data.test)}
        _n_overlap = sum(1 for e in pseudo_out if e in _vt)
        print(
            f"[pseudo] generated {len(pseudo_out)} pseudo-triples / {len(pairs)} (h,r) pairs with deg(h)<=2 "
            f"(skipped {n_skipped} for margin < {mg}) | OVERLAP with valid/test: {_n_overlap} "
            f"({100 * _n_overlap / max(len(pseudo_out), 1):.1f} %) - reported, NOT filtered",
            flush=True,
        )

    # capture S/L on TEST (enable BEFORE eval, write RIGHT AFTER, then switch off so valid is not mixed in)
    if args.dump_SL_test:
        _enable_sl_capture(model, args)
    if args.t_path_masked_test:
        _T_masked = torch.load(args.t_path_masked_test, map_location="cpu").to(dev)
        assert (
            _T_masked.shape == T.shape
        ), f"T_masked {tuple(_T_masked.shape)} != T {tuple(T.shape)}"
        # ASSERT, no longer a warning: the lexical block reads its buffers from `L_feat.pt` and
        # does NOT receive `T`; if `g_L > 0` and only `T` is masked, L still sees the answer name
        # (feature #3), so the drop is a LOWER BOUND and the difference against the baseline is
        # measured wrongly. Letting it through silently is worse than stopping.
        if float(getattr(args, "lsr_gL", 0) or 0) > 0 and not args.l_feat_masked_test:
            raise SystemExit(
                "g_L > 0 and --t_path_masked_test given but --l_feat_masked_test is MISSING.\n"
                "   Masking T does NOT mask L. Build the masked L_feat with build_masked_chain.py."
            )

    def _score_test_masked():
        """Second TEST pass: swap T **and** the L buffers, restore them in `finally`."""
        _L = getattr(model, "lsr_L", None) or getattr(model.tucker, "L", None)
        _bak = None
        if args.l_feat_masked_test and _L is not None:
            _d = torch.load(args.l_feat_masked_test, map_location="cpu", weights_only=False)
            _ix = _d["idx"].long().clone()
            _ft = _d["feat"].float()
            _ix[_ft[:, :, 0] <= 0] = _ft.shape[0]  # sentinel, as in LexicalBlock.__init__
            _bak = (_L.idx, _L.feat, _L.co)
            _L.idx = _ix.to(dev)
            _L.feat = _ft.to(dev)
            _L.co = (_ft[:, :, 0] > 0).float().to(dev)
            print(f"[masked] swapped L to {args.l_feat_masked_test}", flush=True)

        def _sweep_masked(path: str) -> None:
            """Sweep the `--sweep_gL` grid for the MASKED pass and write the table file.

            Runs WHILE T/L are swapped (called from the `try` of `_score_test_masked`), so it
            measures the masked pass. Columns, keys and row order match `--sweep_test_out` so the
            drop can be subtracted per key.
            """
            import csv as _csv_c

            gs_c = [float(x) for x in args.sweep_gL.split(",")]
            _g_old_c = model.tucker.g_L
            _tmp_c = path + ".tmp"
            col_c, keys_c = {}, None
            for _g in gs_c:
                model.tucker.g_L = _g
                evaluate(
                    model,
                    _T_masked,
                    V,
                    data,
                    "test",
                    use_text=ut,
                    use_image=ui,
                    H=H,
                    Hmask=Hmask,
                    recip_offset=(R0 if recip else 0),
                    name_ids=name_ids,
                    name_mask=name_mask,
                    T2=T2,
                    max_ent=(args.max_ent_gold or None),
                    dump_full=_tmp_c,
                )
                with open(_tmp_c, encoding="utf-8") as f:
                    rows_c = list(_csv_c.DictReader(f, delimiter="\t"))
                if keys_c is None:
                    keys_c = [(x["s_qid"], x["p_pid"], x["o_qid"], x["huong"]) for x in rows_c]
                col_c[_g] = [x["rank_gold"] for x in rows_c]
            model.tucker.g_L = _g_old_c
            os.remove(_tmp_c)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    "# LUOT CHE (text + L da che ten dap an TEST). CHI doc dung cot g* — "
                    "cung cot voi sweep_gL_test.tsv, 7 cot kia KHONG vao bang nao.\n"
                )
                f.write("s_qid\tp_pid\to_qid\thuong\t" + "\t".join(f"g{g}" for g in gs_c) + "\n")
                for i, k in enumerate(keys_c):
                    f.write("\t".join(k) + "\t" + "\t".join(col_c[g][i] for g in gs_c) + "\n")
            print(
                f"[masked] sweep_test_masked -> {path} ({len(keys_c)} keys x {len(gs_c)} g values)",
                flush=True,
            )

        try:
            # `recip_offset` + `name_ids/name_mask` + `max_ent` are passed here on purpose. If they
            # were omitted, the masked pass would score heads through the `recip_offset = 0`
            # branch while the original pass uses reciprocal, giving two different instruments
            # and a drop subtracted across instruments. Every other `evaluate` call in this file
            # passes `recip_offset`; this is the place that must not omit it.
            masked_result = evaluate(
                model,
                _T_masked,
                V,
                data,
                "test",
                dump_path=args.dump_test_masked or None,
                use_text=ut,
                use_image=ui,
                H=H,
                Hmask=Hmask,
                recip_offset=(R0 if recip else 0),
                name_ids=name_ids,
                name_mask=name_mask,
                T2=T2,
                max_ent=(args.max_ent_gold or None),
            )
            if args.sweep_test_masked_out and args.sweep_gL and hasattr(model.tucker, "g_L"):
                _sweep_masked(args.sweep_test_masked_out)
            elif args.sweep_test_masked_out:
                print(
                    "warning: --sweep_test_masked_out IGNORED: --sweep_gL is missing or the variant "
                    "has no g_L. The table will report `masked pass MISSING`, not silently.",
                    flush=True,
                )
            return masked_result
        finally:
            if _bak is not None:
                _L.idx, _L.feat, _L.co = _bak

    test = evaluate(
        model,
        T,
        V,
        data,
        "test",
        dump_path=args.dump_ranks,
        use_text=ut,
        use_image=ui,
        H=H,
        Hmask=Hmask,
        recip_offset=(R0 if recip else 0),
        dump_topk=args.dump_topk,
        name_ids=name_ids,
        name_mask=name_mask,
        T2=T2,
        max_ent=(args.max_ent_gold or None),
        dump_full=args.dump_full,
    )
    if args.dump_SL_test and getattr(getattr(model, "tucker", None), "luu_SL", None):
        _write_sl_dump(model, args, args.dump_SL_test, "test")

    # The second masked TEST pass is placed after the sweep block, see the end of this function.
    # VALIDATION eval is run when ANY valid dump is requested (previously only --dump_full_valid
    # triggered it, so --dump_scores_full alone was skipped).
    # Save the **top-K ids of S** for ALL triples of fold k, both directions. Ids only (int32), so
    # light: 11 399 triples x 2 x 100 x 4 B ~ 9 MB. Used as the CANDIDATE SET of the L block (union
    # variant); it must be the **OUT-OF-FOLD** top-K: a fully trained model memorises train, so
    # gold is always in the in-sample top-K, a distribution shift against valid.
    # The top-K comes from **S**, which depends only on (source, relation), never on gold.
    def _dump_topS(_triples, _path, _split_name):
        """Save the top-K ids of **S** (g_L = 0) for a triple set, BOTH directions. Ids only, so light."""
        model.eval()
        _ids, _kk = [], []
        with torch.no_grad():
            enc = model.encode(
                T,
                V,
                H=H,
                Hmask=Hmask,
                use_text=ut,
                use_image=ui,
                name_ids=name_ids,
                name_mask=name_mask,
                T2=T2,
            )
            ea = enc["h_e"]
            _g0 = model.tucker.g_L
            model.tucker.g_L = 0.0  # the S term only (g_L = 0), explicit
            for _s in range(0, len(_triples), 256):
                ck = _triples[_s : _s + 256]
                hb = torch.tensor([int(x[0]) for x in ck], device=dev)
                rb = torch.tensor([int(x[1]) for x in ck], device=dev)
                tb = torch.tensor([int(x[2]) for x in ck], device=dev)
                for _dir, _query, _r in (("tail", hb, rb), ("head", tb, rb + R0)):
                    model.tucker.dat_chi_so(_query, _query)
                    _sc = model.tucker.score_tail(ea[_query], _r, ea)
                    _ids.append(_sc.topk(args.topS_k, dim=1).indices.to(torch.int32).cpu())
                    _kk += [(x[0], x[1], x[2], _dir) for x in ck]
            model.tucker.g_L = _g0
        torch.save(
            {
                "split": _split_name,
                "fold_k": args.fold_k,
                "k": args.topS_k,
                "ghi_chu": "id top-k cua S (g_L=0), ca hai chieu",
                "keys": _kk,
                "idx": torch.cat(_ids),
            },
            _path,
        )
        print(
            f"[topS] top-{args.topS_k} of S ({_split_name}) -> {_path} "
            f"({len(_kk)} keys = {len(_triples)} triples x 2 directions)",
            flush=True,
        )

    if args.oof_topS and _fold_triples is not None:
        _dump_topS(_fold_triples_full, args.oof_topS, "oof_topS")  # the WHOLE fold k
    if args.topS_vt_out:
        _dump_topS(list(data.valid) + list(data.test), args.topS_vt_out, "valid_test_topS")

    # Score the triples of fold k (NOT in this run's train), BOTH directions, writing S and L
    # separately. The source of each direction is the entity on the query side.
    def _dump_fold_scores(_triples, _path, _split_name):
        """Score a triple set, BOTH directions, writing S/L separately (L compressed sparse, bit-exact).

        `_split_name` = "oof" (held-out fold) or "insample" (fold that is in train). Both calls use
        the **same** code path, so nothing differs but the triple set (required for the control).
        The source of each direction is the entity on the query side.
        """
        model.eval()
        _o = {"S": [], "L": [], "S_max": [], "L_max": []}
        _keys = []
        with torch.no_grad():
            enc = model.encode(
                T,
                V,
                H=H,
                Hmask=Hmask,
                use_text=ut,
                use_image=ui,
                name_ids=name_ids,
                name_mask=name_mask,
                T2=T2,
            )
            ea = enc["h_e"]
            for _s in range(0, len(_triples), 256):
                ck = _triples[_s : _s + 256]
                hb = torch.tensor([int(x[0]) for x in ck], device=dev)
                rb = torch.tensor([int(x[1]) for x in ck], device=dev)
                tb = torch.tensor([int(x[2]) for x in ck], device=dev)
                for _dir, _query, _r in (("tail", hb, rb), ("head", tb, rb + R0)):
                    model.tucker.dat_chi_so(_query, _query)
                    model.tucker.luu_SL = {"S": [], "L": [], "S_max": [], "L_max": []}
                    model.tucker.thu_SL = (
                        "head"  # label so TuckerLexicalFusion captures S/L separately
                    )
                    model.tucker.score_tail(ea[_query], _r, ea)
                    model.tucker.thu_SL = None
                    _b = model.tucker.luu_SL
                    model.tucker.luu_SL = None
                    for _k in ("S", "L", "S_max", "L_max"):
                        _o[_k].append(torch.cat(_b[_k]))
                    _keys += [(x[0], x[1], x[2], _dir) for x in ck]
        import scan_variants as SV  # in case it was not imported in an earlier branch

        _c = {_k: torch.cat(_v) for _k, _v in _o.items()}
        _sparse_L = SV.compress_sparse_lexical(_c.pop("L"))  # sparse L: exact
        torch.save(
            {
                "split": _split_name,
                "fold_k": args.fold_k,
                "huong": "ca_hai",
                "ghi_chu": "S,L da tru max moi query; diem goc = X + X_max. "
                "L luu THUA (L_idx/L_val/L_hang) — dung giai_nen_L_thua().",
                "keys": _keys,
                "oof_sample": args.oof_sample,
                **_c,
                **_sparse_L,
            },
            _path,
        )
        print(
            f"[fold] {_split_name} -> {_path} ({len(_keys)} keys = "
            f"{len(_triples)} triples x 2 directions)",
            flush=True,
        )

    if args.oof_out and _fold_triples is not None:
        _dump_fold_scores(_fold_triples, args.oof_out, "oof")
    if args.insample_out and _fold_triples is not None:
        _dump_fold_scores(_in_triples, args.insample_out, "insample")

    # Sweep g_L at eval time, BOTH directions, ranks only (light). Replaces dumping the S/L
    # matrices (head only, 153 MB/run).
    if args.sweep_gL and args.sweep_out and hasattr(model.tucker, "g_L"):
        import csv as _csv

        gs = [float(x) for x in args.sweep_gL.split(",")]
        _g_orig = model.tucker.g_L
        _tmp = args.sweep_out + ".tmp"
        col, keys = {}, None
        for _g in gs:
            model.tucker.g_L = _g
            evaluate(
                model,
                T,
                V,
                data,
                "valid",
                use_text=ut,
                use_image=ui,
                H=H,
                Hmask=Hmask,
                recip_offset=(R0 if recip else 0),
                name_ids=name_ids,
                name_mask=name_mask,
                T2=T2,
                max_ent=(args.max_ent_gold or None),
                dump_full=_tmp,
            )
            with open(_tmp, encoding="utf-8") as f:
                rows = list(_csv.DictReader(f, delimiter="\t"))
            if keys is None:
                keys = [(x["s_qid"], x["p_pid"], x["o_qid"], x["huong"]) for x in rows]
            col[_g] = [x["rank_gold"] for x in rows]
            print(f"  [sweep g_L={_g}] n={len(rows)}", flush=True)
        # TEST on the same grid, to avoid another GPU session just to score at g*.
        col_t, keys_t = {}, None
        if args.sweep_test_out:
            for _g in gs:
                model.tucker.g_L = _g
                evaluate(
                    model,
                    T,
                    V,
                    data,
                    "test",
                    use_text=ut,
                    use_image=ui,
                    H=H,
                    Hmask=Hmask,
                    recip_offset=(R0 if recip else 0),
                    name_ids=name_ids,
                    name_mask=name_mask,
                    T2=T2,
                    max_ent=(args.max_ent_gold or None),
                    dump_full=_tmp,
                )
                with open(_tmp, encoding="utf-8") as f:
                    rows = list(_csv.DictReader(f, delimiter="\t"))
                if keys_t is None:
                    keys_t = [(x["s_qid"], x["p_pid"], x["o_qid"], x["huong"]) for x in rows]
                col_t[_g] = [x["rank_gold"] for x in rows]
            with open(args.sweep_test_out, "w", encoding="utf-8") as f:
                f.write("# CHI doc dung cot g* (chon tren VALID). 7 cot con lai KHONG vao bang.\n")
                f.write("s_qid\tp_pid\to_qid\thuong\t" + "\t".join(f"g{g}" for g in gs) + "\n")
                for i, k in enumerate(keys_t):
                    f.write("\t".join(k) + "\t" + "\t".join(col_t[g][i] for g in gs) + "\n")
            print(f"[sweep] sweep_test -> {args.sweep_test_out} ({len(keys_t)} keys)", flush=True)
        model.tucker.g_L = _g_orig
        os.remove(_tmp)
        with open(args.sweep_out, "w", encoding="utf-8") as f:
            f.write("s_qid\tp_pid\to_qid\thuong\t" + "\t".join(f"g{g}" for g in gs) + "\n")
            for i, k in enumerate(keys):
                f.write("\t".join(k) + "\t" + "\t".join(col[g][i] for g in gs) + "\n")
        print(
            f"[sweep] sweep_gL -> {args.sweep_out} ({len(keys)} keys x {len(gs)} g values)",
            flush=True,
        )

    # Second TEST pass: same checkpoint, text (and L) with the answer name masked.
    # Placed AFTER the sweep block: previously it ran right after the original test pass, so a
    # crash in the masked pass erased the run's `sweep_gL*.tsv`. The reorder is safe for every
    # earlier run because this branch **only runs with `--t_path_masked_test`**, and `g_L` has
    # already been restored to `_g_orig` by the sweep block.
    if args.t_path_masked_test:
        _masked = _score_test_masked()
        print(
            f"  TEST MASKED MRR {_masked['MRR']:.4f} | drop {test['MRR'] - _masked['MRR']:+.5f}",
            flush=True,
        )

    # One extra READ-ONLY TEST eval at g = --lsr_gL (the trained value), writing three ranks over
    # the same score vector. Placed AFTER the sweep so the sweep is not overwritten, and uses a
    # FIXED g so the analysis is comparable with the `g<lsr_gL>` column of the sweep (same instrument).
    if args.dump_check_test:
        _g_old = getattr(getattr(model, "tucker", None), "g_L", None)
        if _g_old is not None:
            model.tucker.g_L = args.lsr_gL
        evaluate(
            model,
            T,
            V,
            data,
            "test",
            use_text=ut,
            use_image=ui,
            H=H,
            Hmask=Hmask,
            recip_offset=(R0 if recip else 0),
            name_ids=name_ids,
            name_mask=name_mask,
            T2=T2,
            max_ent=(args.max_ent_gold or None),
            dump_kiem=args.dump_check_test,
        )
        if _g_old is not None:
            model.tucker.g_L = _g_old

    # Capture the S and L matrices separately on valid/head, to sweep g_L post hoc (0 GPU).
    if args.dump_SL:
        _enable_sl_capture(model, args)
    if args.dump_full_valid or args.dump_scores_full or args.dump_scores_rel:
        # Dump scores at g* (if --scores_g) WITHOUT changing the model (training is done). Restored afterwards.
        _g_sc = getattr(getattr(model, "tucker", None), "g_L", None)
        if args.scores_g is not None and _g_sc is not None:
            model.tucker.g_L = args.scores_g
            print(
                f"[scores_g] dumping VALID scores at g_L={args.scores_g} (training unchanged)",
                flush=True,
            )
        mv = evaluate(
            model,
            T,
            V,
            data,
            "valid",
            use_text=ut,
            use_image=ui,
            H=H,
            Hmask=Hmask,
            recip_offset=(R0 if recip else 0),
            name_ids=name_ids,
            name_mask=name_mask,
            T2=T2,
            max_ent=(args.max_ent_gold or None),
            dump_full=args.dump_full_valid,
            dump_scores_rel=args.dump_scores_rel,
            scores_rel_pid=args.scores_rel_pid,
            dump_scores_full=args.dump_scores_full,
            scores_full_huong=args.scores_full_direction,
        )
        if args.scores_g is not None and _g_sc is not None:
            model.tucker.g_L = _g_sc
        if args.dump_SL and getattr(model.tucker, "luu_SL", None):
            _write_sl_dump(model, args, args.dump_SL, "valid")
        print(
            f"[valid-final] MRR {mv['MRR']:.4f} H@1 {mv['Hits@1']:.4f} H@10 {mv['Hits@10']:.4f} "
            f"(dump={args.dump_full_valid})",
            flush=True,
        )
    # TEST score dump: PARALLEL to the valid block above, same parameters, only `split` changes.
    # Placed AFTER valid so the eval order is unchanged for every old command (without this flag
    # the branch does not run, so earlier runs keep their trajectory).
    if args.dump_scores_full_test:
        _g_sc = getattr(getattr(model, "tucker", None), "g_L", None)
        if args.scores_g is not None and _g_sc is not None:
            model.tucker.g_L = args.scores_g
            print(
                f"[scores_g] dumping TEST scores at g_L={args.scores_g} (training unchanged)",
                flush=True,
            )
        mt = evaluate(
            model,
            T,
            V,
            data,
            "test",
            use_text=ut,
            use_image=ui,
            H=H,
            Hmask=Hmask,
            recip_offset=(R0 if recip else 0),
            name_ids=name_ids,
            name_mask=name_mask,
            T2=T2,
            max_ent=(args.max_ent_gold or None),
            dump_scores_full=args.dump_scores_full_test,
            scores_full_huong=args.scores_full_direction,
        )
        if args.scores_g is not None and _g_sc is not None:
            model.tucker.g_L = _g_sc
        print(
            f"[test-dump] MRR {mt['MRR']:.4f} H@1 {mt['Hits@1']:.4f} H@10 {mt['Hits@10']:.4f} "
            f"(scores -> {args.dump_scores_full_test})",
            flush=True,
        )
    # --- visual attribute slice (only with --v2_dir) ---
    if args.v2_dir:
        from eval_attribute_slice import eval_v2, print_summary

        kq_v2 = eval_v2(
            model,
            T,
            V,
            args.v2_dir,
            use_text=ut,
            use_image=ui,
            H=H,
            Hmask=Hmask,
            T2=T2,
            dump=args.v2_dump,
        )
        print_summary(kq_v2, args.exp)
        if args.v2_out:
            import json as _json

            os.makedirs(os.path.dirname(args.v2_out) or ".", exist_ok=True)
            _json.dump(
                {
                    "exp": args.exp,
                    "seed": args.seed,
                    "use_image": int(ui),
                    "gold_test": test,
                    "v2": kq_v2,
                },
                open(args.v2_out, "w"),
                ensure_ascii=False,
                indent=1,
            )

    if LS is not None:
        print(
            f"[structured_ls] fallback cases to `deu` (empty smoothing set): {LS['n_fallback']}",
            flush=True,
        )
    print(f"[MENTOR] DONE {args.exp} (best valid MRR={best_valid:.4f}).", flush=True)
    print(
        f"  TEST MRR {test['MRR']:.4f} H@1 {test['Hits@1']:.4f} H@3 {test['Hits@3']:.4f} "
        f"H@10 {test['Hits@10']:.4f} MR {test['MR']:.1f}  (dump={args.dump_ranks})",
        flush=True,
    )

    # --- gate diagnostics: learned g per entity + mean per tier ---
    with torch.no_grad():
        g = model.gate(V).detach().cpu()
    tiers = data.tier
    print("  [g] mean per tier (reference prior):", flush=True)
    for tk, pr in TIER_PRIOR.items():
        m = (tiers == tk) & data.has_image
        if int(m.sum()) > 0:
            print(
                f"      tier{tk}: g_mean={float(g[m].mean()):.4f} (prior={pr}) n={int(m.sum())}",
                flush=True,
            )
    if args.dump_gate:
        with open(args.dump_gate, "w") as f:
            f.write("id\ttier\tg\n")
            for i in range(data.num_ent):
                if bool(data.has_image[i]):
                    f.write(f"{i}\t{int(tiers[i])}\t{float(g[i]):.4f}\n")
        print(f"  [g] dump -> {args.dump_gate}", flush=True)


if __name__ == "__main__":
    main()
