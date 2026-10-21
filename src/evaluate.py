#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Filtered ranking evaluation for the model, plus rank dumps (ranks.tsv).

Standard two-direction filtered ranking on the test split; MRR / Hits@1 / Hits@3 / Hits@10. The
ranks.tsv dump has the form (s_qid p_pid o_qid head_rank tail_rank) so that stratified analyses
and the attribute slice can reuse it.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional, Tuple

import torch

SEP = "\x1f"  # top-k separator in dump_full (entity names may contain commas)


def _id_maps(data_dir: str) -> Tuple[Dict[int, str], Dict[int, str]]:
    """Numeric id -> qid / pid (reads entity2id.tsv, relation2id.tsv).

    Reads the first two columns only: entity2id.tsv of the v2 dataset has three columns
    (name, id, kind), and unpacking the whole line would fail.
    """
    id2qid, id2pid = {}, {}
    with open(os.path.join(data_dir, "entity2id.tsv")) as f:
        next(f)
        for line in f:
            if not line.strip():
                continue
            qid, i = line.rstrip("\n").split("\t")[:2]
            id2qid[int(i)] = qid
    with open(os.path.join(data_dir, "relation2id.tsv")) as f:
        next(f)
        for line in f:
            if not line.strip():
                continue
            pid, i = line.rstrip("\n").split("\t")[:2]
            id2pid[int(i)] = pid
    return id2qid, id2pid


def _filtered_rank(scores: torch.Tensor, target: int, filt: set) -> int:
    """Rank of the target after removing the other true answers (filtered). scores [N]."""
    gold = scores[target].item()
    others = torch.tensor([e for e in filt if e != target], dtype=torch.long)
    if others.numel() > 0:
        scores = scores.clone()
        scores[others] = float("-inf")
    # rank = number of scores > gold, plus half of the ties (rounded up)
    greater = (scores > gold).sum().item()
    equal = (scores == gold).sum().item()
    return greater + (equal + 1) // 2


@torch.no_grad()
def evaluate(
    model,
    T: torch.Tensor,
    V: torch.Tensor,
    data,
    split: str = "test",
    dump_path: Optional[str] = None,
    batch: int = 256,
    use_text: bool = True,
    use_image: bool = True,
    H: Optional[torch.Tensor] = None,
    Hmask: Optional[torch.Tensor] = None,
    recip_offset: int = 0,
    dump_topk: Optional[str] = None,
    topk: int = 5,
    dump_full: Optional[str] = None,
    topk_full: int = 10,
    dump_scores_rel: Optional[str] = None,
    scores_rel_pid: str = "",
    dump_scores_full: Optional[str] = None,
    scores_full_huong: str = "head",
    name_ids: Optional[torch.Tensor] = None,
    name_mask: Optional[torch.Tensor] = None,
    T2: Optional[torch.Tensor] = None,
    max_ent: Optional[int] = None,
    dump_kiem: Optional[str] = None,
) -> Dict[str, float]:
    """Two-direction filtered evaluation. Returns metrics; writes ranks.tsv if dump_path is given.

    model.eval() makes BatchNorm use running statistics (no batch-dependent eval noise).
    H/Hmask: token states for the attn/tokfuse text modes (None for mean).
    recip_offset>0: head prediction = score_tail(t, r+offset) instead of score_head(r, t).
    dump_scores_full: write the FULL fp16 score matrix [n_query, N] of the split (usually
             "valid"), with the other true answers filtered out, so that later analyses (re-ranking,
             evidence mixing, oracle ceilings) need no GPU rerun. `scores_full_huong` is one of
             {head, tail, ca_hai}.
    dump_full: write rank_gold / score_gold / top10_qid / top10_score, one row per query and direction.
    dump_kiem (READ-ONLY, never changes the reported ranks): write one row per query and direction
             with three ranks computed on the SAME score vector: `rank_ours` (`_filtered_rank`, the
             function in use), `rank_mygo` (`mygo_reference_eval.calculate_rank`, a verbatim copy of
             the MyGO reference) and `rank_toi_uu` (optimistic tie-break, only used as a reverse
             check). It also records `n_loc` and `mask_sha256` of the filter set (gold excluded) for
             comparison with an independent rebuild. `rank_ours` must equal `rank_gold` of
             `dump_full` at the same `g`.
    max_ent: rank only over entity ids < max_ent (masks the newly added v2 value nodes from the
             candidate set of the gold test, so gold-test keeps the original 15 707 candidates).
             None = no masking.
    """
    model.eval()
    dev = next(model.parameters()).device
    Hd = H.to(dev) if H is not None else None
    Hmd = Hmask.to(dev) if Hmask is not None else None
    nid = name_ids.to(dev) if name_ids is not None else None
    nmk = name_mask.to(dev) if name_mask is not None else None
    T2d = T2.to(dev) if T2 is not None else None
    enc = model.encode(
        T.to(dev),
        V.to(dev),
        H=Hd,
        Hmask=Hmd,
        use_text=use_text,
        use_image=use_image,
        name_ids=nid,
        name_mask=nmk,
        T2=T2d,
    )
    ent_all = enc["h_e"]  # [N,d]

    id2pid_cache = {}
    if dump_scores_rel or dump_scores_full:
        _, id2pid_cache = _id_maps(data.dir)
    triples = {"valid": data.valid, "test": data.test}[split]
    ranks: List[int] = []
    records = []  # (h,r,t,hr,tr) for the dump
    topk_records = []  # (h,r,t,hr, top-k head ids)
    full_records = []  # (h,r,t,huong,rank,score_gold,[top10],[score10])
    sc_keys, sc_rows, sc_max = [], [], []  # full scores, head direction, one relation
    sf_keys, sf_h, sf_t = [], [], []  # full score matrices
    sf_hmax, sf_tmax = [], []  # per-query max (fp32) for restoring
    audit_records = []  # (h,r,t,huong,3 ranks,n_loc,sha)
    invariance_checked = False  # gold-invariance check: once per eval
    for s in range(0, len(triples), batch):
        chunk = triples[s : s + batch]
        h = torch.tensor([x[0] for x in chunk], device=dev)
        r = torch.tensor([x[1] for x in chunk], device=dev)
        t = torch.tensor([x[2] for x in chunk], device=dev)
        # Wrapped scorers need the SOURCE entity indices (they cannot be inferred from the vector).
        # Guarded: other variants have no such method.
        # The source indices must be reset before EACH scoring call. Calling dat_chi_so(h, t) only
        # once and scoring both directions would leave the head direction (through the reciprocal
        # relation) using self._h = h, which IS the gold of the head direction, so the text term
        # would rank candidates by their text similarity to the answer itself.
        if hasattr(model.tucker, "dat_chi_so"):
            model.tucker.dat_chi_so(h, t)
        # Collect BOTH directions when requested (`hai_chieu`). By default the tail direction is not
        # collected, which keeps every older dump identical (head only).
        _both_dirs = getattr(model.tucker, "hai_chieu", False) and getattr(
            model.tucker, "luu_SL", None
        )
        if _both_dirs:
            model.tucker.thu_SL = "tail"
            model.tucker.khoa_SL += [
                (int(a), int(b), int(c), "tail")
                for a, b, c in zip(h.tolist(), r.tolist(), t.tolist())
            ]
        s_tail = model.tucker.score_tail(ent_all[h], r, ent_all)  # [B,N] - source = h
        if _both_dirs:
            model.tucker.thu_SL = None
        if recip_offset > 0:
            if hasattr(model.tucker, "dat_chi_so"):
                model.tucker.dat_chi_so(t, h)  # source of head prediction = t
            if hasattr(model.tucker, "thu_SL"):
                model.tucker.thu_SL = "head"  # tag used to collect S/L
                if getattr(model.tucker, "luu_SL", None) is not None and hasattr(
                    model.tucker, "khoa_SL"
                ):
                    model.tucker.khoa_SL += [
                        (int(a), int(b), int(c), "head")
                        for a, b, c in zip(h.tolist(), r.tolist(), t.tolist())
                    ]
            s_head = model.tucker.score_tail(
                ent_all[t], r + recip_offset, ent_all
            )  # tail prediction of r^-1
            if hasattr(model.tucker, "thu_SL"):
                model.tucker.thu_SL = None
            if hasattr(model.tucker, "dat_chi_so"):
                model.tucker.dat_chi_so(h, t)  # restore, avoid leaking into the next round
        else:
            s_head = model.tucker.score_head(r, ent_all[t], ent_all)  # [B,N]
        # ---- Gold-invariance check (automatic, first batch of every eval).
        # Permute the gold and score again: the score of EVERY candidate must be unchanged. The check
        # runs on the raw matrix, before filtering: filtering may depend on the answer, scoring may not.
        if not invariance_checked and len(chunk) > 1:
            invariance_checked = True
            with torch.no_grad():
                t_perm = torch.roll(t, 1)  # permute the gold of the tail direction
                h_perm = torch.roll(h, 1)  # permute the gold of the head direction
                if hasattr(model.tucker, "dat_chi_so"):
                    model.tucker.dat_chi_so(h, t_perm)
                s_tail2 = model.tucker.score_tail(ent_all[h], r, ent_all)
                if recip_offset > 0:
                    if hasattr(model.tucker, "dat_chi_so"):
                        model.tucker.dat_chi_so(t, h_perm)
                    s_head2 = model.tucker.score_tail(ent_all[t], r + recip_offset, ent_all)
                else:
                    # The indices must be reset in this branch too; otherwise the head direction
                    # would be scored in the state (h, t_perm) left by the tail check. Every wrapper's
                    # `score_head` reads `self._t`, the QUERY side of the head direction, so changing
                    # `_t` changes the scores by design and the check would always fail. The gold of the
                    # head direction is `h`, stored in `_h`: permute `h` and KEEP `_t = t`.
                    if hasattr(model.tucker, "dat_chi_so"):
                        model.tucker.dat_chi_so(h_perm, t)
                    s_head2 = model.tucker.score_head(r, ent_all[t], ent_all)
                if hasattr(model.tucker, "dat_chi_so"):
                    model.tucker.dat_chi_so(h, t)
                d_t = (s_tail2 - s_tail).abs().max().item()
                d_h = (s_head2 - s_head).abs().max().item()
            if max(d_t, d_h) > 0:
                raise SystemExit(
                    f"GOLD INVARIANCE BROKEN - permuting the gold changed candidate scores: "
                    f"|delta| tail {d_t:.6g} - head {d_h:.6g}. The scoring function uses "
                    f"information that only the answer carries."
                )
            print(
                f"  [gold-invariance] PASS (|delta| tail {d_t:.1g} - head {d_h:.1g}, "
                f"n={len(chunk)}, split={split})",
                flush=True,
            )

        if max_ent is not None and max_ent < s_tail.shape[1]:  # mask the v2 value nodes
            s_tail[:, max_ent:] = float("-inf")
            s_head[:, max_ent:] = float("-inf")
        for i, (hh, rr, tt) in enumerate(chunk):
            sh = s_head[i].cpu()
            st = s_tail[i].cpu()
            tr = _filtered_rank(st, tt, data.filter_tail[(hh, rr)])
            hr = _filtered_rank(sh, hh, data.filter_head[(rr, tt)])
            ranks.append(tr)
            ranks.append(hr)
            records.append((hh, rr, tt, hr, tr))
            if dump_scores_full:
                # Store the SAME vector used for ranking (filtered), fp16, without changing the rank.
                sf_keys.append((hh, rr, tt))
                # fp16 has fixed RELATIVE precision: at |score| ~ 7 the quantisation step is ~0.007,
                # enough to swap the order near the top. Subtracting the per-query max BEFORE
                # downcasting puts the top near 0, where fp16 is finest (~1e-5). The fp32 max is
                # stored alongside to restore the scores.
                if scores_full_huong in ("head", "ca_hai"):
                    vh = sh.clone()
                    o_ = [e for e in data.filter_head[(rr, tt)] if e != hh]
                    if o_:
                        vh[torch.tensor(o_, dtype=torch.long)] = float("-inf")
                    mx = vh[torch.isfinite(vh)].max()
                    sf_hmax.append(float(mx))
                    sf_h.append((vh - mx).to(torch.float16))
                if scores_full_huong in ("tail", "ca_hai"):
                    vt_ = st.clone()
                    o_ = [e for e in data.filter_tail[(hh, rr)] if e != tt]
                    if o_:
                        vt_[torch.tensor(o_, dtype=torch.long)] = float("-inf")
                    mx = vt_[torch.isfinite(vt_)].max()
                    sf_tmax.append(float(mx))
                    sf_t.append((vt_ - mx).to(torch.float16))
            if dump_scores_rel and id2pid_cache.get(rr) == scores_rel_pid:
                # Filtered score vector of ALL candidates, head direction, one relation.
                others = [e for e in data.filter_head[(rr, tt)] if e != hh]
                v = sh.clone()
                if others:
                    v[torch.tensor(others, dtype=torch.long)] = float("-inf")
                mx = v[torch.isfinite(v)].max()  # subtract the max (see above)
                sc_keys.append((hh, rr, tt))
                sc_max.append(float(mx))
                sc_rows.append((v - mx).to(torch.float16))
            if dump_full:  # same vector, same filter; only reads, never changes the rank
                for huong, vec, gold, filt, rk in (
                    ("head", sh, hh, data.filter_head[(rr, tt)], hr),
                    ("tail", st, tt, data.filter_tail[(hh, rr)], tr),
                ):
                    others = [e for e in filt if e != gold]
                    v = vec.clone()
                    if others:
                        v[torch.tensor(others, dtype=torch.long)] = float("-inf")
                    k = min(topk_full, v.numel())
                    top = torch.topk(v, k)
                    full_records.append(
                        (
                            hh,
                            rr,
                            tt,
                            huong,
                            rk,
                            float(vec[gold].item()),
                            top.indices.tolist(),
                            [float(x) for x in top.values.tolist()],
                        )
                    )
            if dump_kiem:  # read-only audit: same vector, same filter set
                import hashlib as _hl

                import mygo_reference_eval as _REF

                for huong, vec, gold, filt, rk in (
                    ("head", sh, hh, data.filter_head[(rr, tt)], hr),
                    ("tail", st, tt, data.filter_tail[(hh, rr)], tr),
                ):
                    _scores_np = vec.detach().float().cpu().numpy()
                    _filt_sorted = sorted(filt)  # as in MyGO: the filter dict INCLUDES the gold
                    _rank_mygo = int(
                        _REF.calculate_rank(_scores_np.copy(), int(gold), _filt_sorted)
                    )
                    _v = _scores_np.copy()
                    _g0 = _v[int(gold)]
                    _v[_filt_sorted] = _g0 - 1  # optimistic tie-break (reverse check)
                    _rank_optimistic = int((_v > _g0).sum() + 1)
                    _filt_wo_gold = sorted(x for x in filt if x != gold)
                    _sha = _hl.sha256(",".join(map(str, _filt_wo_gold)).encode()).hexdigest()
                    audit_records.append(
                        (
                            hh,
                            rr,
                            tt,
                            huong,
                            rk,
                            _rank_mygo,
                            _rank_optimistic,
                            len(_filt_wo_gold),
                            _sha,
                        )
                    )
            if dump_topk:  # top-k head predictions (other true heads filtered, gold kept)
                others = data.filter_head[(rr, tt)] - {hh}
                if others:
                    sh = sh.clone()
                    sh[list(others)] = float("-inf")
                    topk_records.append((hh, rr, tt, hr, torch.topk(sh, topk).indices.tolist()))
                else:
                    topk_records.append((hh, rr, tt, hr, torch.topk(sh, topk).indices.tolist()))

    rt = torch.tensor(ranks, dtype=torch.float)
    metrics = {
        "MRR": float((1.0 / rt).mean()),
        "Hits@1": float((rt <= 1).float().mean()),
        "Hits@3": float((rt <= 3).float().mean()),
        "Hits@10": float((rt <= 10).float().mean()),
        "MR": float(rt.mean()),
        "n": len(triples),
    }
    if dump_path:
        _dump_ranks(dump_path, records, data.dir)
    if dump_topk:
        _dump_topk(dump_topk, topk_records, data.dir)
    if dump_full:
        _dump_full(dump_full, full_records, data.dir)
    if dump_kiem:
        _dump_rank_audit(dump_kiem, audit_records, data.dir, split)
    if dump_scores_full:
        id2qid, id2pid = _id_maps(data.dir)
        os.makedirs(os.path.dirname(dump_scores_full) or ".", exist_ok=True)
        blob = {
            "split": split,
            "huong": scores_full_huong,
            "N": int(ent_all.shape[0]),
            "keys": [(id2qid[h], id2pid[r], id2qid[t]) for h, r, t in sf_keys],
        }
        # NOTE: scores_* already have the per-query max subtracted. Original score = scores_* + max_*[i].
        blob["ghi_chu"] = "scores_* da tru max moi query; diem goc = scores_* + max_*"
        if sf_h:
            blob["scores_head"] = torch.stack(sf_h)
            blob["max_head"] = torch.tensor(sf_hmax, dtype=torch.float32)
        if sf_t:
            blob["scores_tail"] = torch.stack(sf_t)
            blob["max_tail"] = torch.tensor(sf_tmax, dtype=torch.float32)
        torch.save(blob, dump_scores_full)
        mb = sum(v.numel() * 2 for k, v in blob.items() if torch.is_tensor(v)) / 1048576
        print(
            f"[dump_scores_full] {split}/{scores_full_huong}: "
            f"{len(sf_keys)} queries x {int(ent_all.shape[0])} candidates, fp16 ~{mb:.0f} MB "
            f"-> {dump_scores_full}",
            flush=True,
        )
    if dump_scores_rel:
        id2qid, id2pid = _id_maps(data.dir)
        os.makedirs(os.path.dirname(dump_scores_rel) or ".", exist_ok=True)
        torch.save(
            {
                "pid": scores_rel_pid,
                "huong": "head",
                "ghi_chu": "scores da TRU max moi query; diem goc = scores + max",
                "keys": [(id2qid[h], id2pid[r], id2qid[t]) for h, r, t in sc_keys],
                "scores": torch.stack(sc_rows) if sc_rows else torch.empty(0),
                "max": torch.tensor(sc_max, dtype=torch.float32),
            },
            dump_scores_rel,
        )
        print(
            f"[dump_scores_rel] {scores_rel_pid}: {len(sc_rows)} queries "
            f"x {sc_rows[0].numel() if sc_rows else 0} candidates -> {dump_scores_rel}",
            flush=True,
        )
    return metrics


def _dump_rank_audit(path: str, records, data_dir: str, split: str) -> None:
    """Write the rank audit: three ranks on the SAME vector plus a fingerprint of the filter set.

    Columns: s_qid p_pid o_qid huong rank_ours rank_mygo rank_toi_uu n_loc mask_sha256.
    `rank_ours` is exactly the rank used for the metrics, so an analysis can assert that it equals
    `rank_gold` of `dump_full` at the same `g`. Producing this file changes no rank.
    """
    import mygo_reference_eval as _REF

    id2qid, id2pid = _id_maps(data_dir)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            f"# split={split} · rank_ours=_filtered_rank · rank_mygo=mygo_ref."
            f"calculate_rank (verbatim {_REF.MYGO_COMMIT[:12]}) · rank_toi_uu=tie-break "
            f"lac quan (phep kiem nguoc N2) · mask_sha256 = sha256 cua tap filter BO gold\n"
        )
        f.write(
            "s_qid\tp_pid\to_qid\thuong\trank_ours\trank_mygo\trank_toi_uu\t" "n_loc\tmask_sha256\n"
        )
        for hh, rr, tt, huong, r_o, r_m, r_t, n_l, sha in records:
            f.write(
                f"{id2qid[hh]}\t{id2pid[rr]}\t{id2qid[tt]}\t{huong}\t{r_o}\t{r_m}\t"
                f"{r_t}\t{n_l}\t{sha}\n"
            )
    print(f"[rank_audit] dump_kiem[{split}] -> {path} ({len(records)} rows)", flush=True)


def _dump_full(path: str, records, data_dir: str) -> None:
    """One row per query and direction: rank_gold, score_gold, top10_qid, top10_score.

    The top-10 is taken on the FILTERED vector (other true answers removed, gold kept), the same
    filter as `_filtered_rank`, so top-1 is consistent with the rank. score_gold is the RAW
    (unfiltered) gold score.
    """
    id2qid, id2pid = _id_maps(data_dir)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("s_qid\tp_pid\to_qid\thuong\trank_gold\tscore_gold\ttop10_qid\ttop10_score\n")
        for h, r, t, huong, rk, sg, tops, vals in records:
            # Separator = U+001F (unit separator): entity names can contain commas
            # (e.g. "Girl, Interrupted"), so "," must not be used.
            f.write(
                f"{id2qid[h]}\t{id2pid[r]}\t{id2qid[t]}\t{huong}\t{rk}\t{sg:.6f}\t"
                f"{SEP.join(id2qid[i] for i in tops)}\t"
                f"{SEP.join(f'{v:.6f}' for v in vals)}\n"
            )


def _dump_topk(path: str, records, data_dir: str) -> None:
    """Write the top-k predicted heads (qid) for each test triple."""
    id2qid, id2pid = _id_maps(data_dir)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("s_qid\tp_pid\to_qid\thead_rank\ttop_heads\n")
        for h, r, t, hr, tops in records:
            f.write(
                f"{id2qid[h]}\t{id2pid[r]}\t{id2qid[t]}\t{hr}\t{','.join(id2qid[i] for i in tops)}\n"
            )


def _dump_ranks(path: str, records: List[Tuple[int, int, int, int, int]], data_dir: str) -> None:
    id2qid, id2pid = _id_maps(data_dir)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("s_qid\tp_pid\to_qid\thead_rank\ttail_rank\n")
        for h, r, t, hr, tr in records:
            f.write(f"{id2qid[h]}\t{id2pid[r]}\t{id2qid[t]}\t{hr}\t{tr}\n")


def vis_slice_metrics(ranks_path: str, vis_rel_path: str) -> Optional[Dict[str, float]]:
    """Visual-slice metrics: MRR/Hits over the triples whose relation is labelled VIS (if the file exists)."""
    if not os.path.exists(vis_rel_path):
        return None
    vis = set()
    with open(vis_rel_path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            # last-label column: the final file uses 'nhan_cuoi'; 'chot'/'nhan' are accepted as well
            lab = (r.get("nhan_cuoi") or r.get("chot") or r.get("nhan") or "").upper()
            if lab == "VIS":
                vis.add(r["pid"])
    if not vis:
        return None
    rr = []
    with open(ranks_path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["p_pid"] in vis:
                rr.append(int(r["head_rank"]))
                rr.append(int(r["tail_rank"]))
    if not rr:
        return None
    rt = torch.tensor(rr, dtype=torch.float)
    return {
        "vis_MRR": float((1.0 / rt).mean()),
        "vis_Hits@10": float((rt <= 10).float().mean()),
        "vis_n_triple": len(rr) // 2,
        "vis_n_rel": len(vis),
    }
