"""Generate `bang_canh_v2.{tex,md}` and the mention-edge x slice/regular numbers used by the
delta-L and per-level tables.

Headline model = A3'* lambda 8 (ranks_v2/A3pLC, overall 41.20); baseline* = lambda 8
(luoi_c nen_l8.0). Splits queries into with / without a mention edge x slice / regular,
MRR % (both directions = key), mean of 3 seeds.

Inputs: v2 mention-edge labels `canh_v2/v10_canh_nhac_ten_ViMMKG2_v2.tsv` (numeric id pairs,
load_mention_edges); slice/regular groups from `build_hits_table.query_groups_v2` (qid).
A key has a mention edge iff `frozenset({h_id, t_id})` is in the edge set.
Cross-check: A3'* lambda 8 overall MRR = 41.20 +- 0.01; n regular-with-edge = 1348 keys.
"""

from __future__ import annotations

import csv
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_hits_table import EXT2, query_groups_v2  # noqa: E402

ROOT = HERE.parents[1]
RANKS_DIR = ROOT / "data" / "experimental" / "v3" / "ranks_v2"
MENTION_EDGES_FILE = (
    ROOT / "data" / "experimental" / "v3" / "canh_v2" / "v10_canh_nhac_ten_ViMMKG2_v2.tsv"
)
SEEDS = ("42", "7", "2024")


def load_entity_ids() -> dict:
    m = {}
    with open(f"{EXT2}/entity2id.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            if line.strip():
                v = line.rstrip("\n").split("\t")
                m[v[0]] = int(v[1])  # qid -> numeric id
    return m


def load_mention_edges() -> set:
    with open(MENTION_EDGES_FILE, encoding="utf-8") as f:
        next(f)
        return {
            frozenset((int(a), int(b)))
            for a, b in (line.split("\t") for line in f.read().splitlines()[1:] if line)
        }


def read_wide(p: str) -> dict:
    out = {}
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            k = (r["s_qid"], r["p_pid"], r["o_qid"])
            out[k + ("head",)] = float(r["head_rank"])
            out[k + ("tail",)] = float(r["tail_rank"])
    return out


def read_adamf(p: str, e2i: dict) -> dict:
    """AdaMF long format: key = 's|r|o' (numeric ids), direction, rank. Returns QID keys, like the other tables."""
    i2q = {v: k for k, v in e2i.items()}
    rel = {}
    with open(f"{EXT2}/relation2id.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            if line.strip():
                v = line.rstrip("\n").split("\t")
                rel[int(v[1])] = v[0]
    out = {}
    with open(p, encoding="utf-8") as f:
        next(f)
        for line in f:
            if not line.strip():
                continue
            c = line.rstrip("\n").split("\t")
            si, ri, oi = (int(x) for x in c[0].split("|"))
            out[(i2q[si], rel[ri], i2q[oi], c[1])] = float(c[2])
    return out


def read_dump(p: str) -> dict:
    out = {}
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            out[(r["s_qid"], r["p_pid"], r["o_qid"], r["huong"])] = float(r["rank_gold"])
    return out


MODELS = [
    ("A3′* λ8 (headline)", lambda s: (RANKS_DIR / f"A3pLC_s{s}" / "ranks_test.tsv", "wide")),
    ("nền* λ8", lambda s: (RANKS_DIR / f"nen_l8_s{s}" / "dump_test.tsv", "dump")),
    (
        "MyGO+seg",
        lambda s: (
            ROOT / f"data/experimental/mygo_seg/MyGO_ViMMKG2_seg_s{s}/ranks_test.tsv",
            "wide",
        ),
    ),
    ("MoMoK", lambda s: (RANKS_DIR / f"momok_s{s}" / "ranks_test.tsv", "wide")),
    ("TSAM", lambda s: (ROOT / f"baselines/tsam/tsam_ViMMKG2_s{s}/ranks_test.tsv", "wide")),
    ("AdaMF-MAT", lambda s: (RANKS_DIR / f"adamf_s{s}" / "rank.tsv", "adamf")),
    ("DSoM re-impl.", lambda s: (RANKS_DIR / f"dsom_s{s}" / "ranks_test.tsv", "wide")),
    (
        "TuckER (cấu trúc)",
        lambda s: (ROOT / f"data/from_kaggle/v2_03/result/D4_tucker_s{s}/ranks_test.tsv", "wide"),
    ),
]

EDGE_GROUPS = ("tổng", "thường·CÓ", "thường·KHÔNG", "lát·CÓ", "lát·KHÔNG", "CÓ", "KHÔNG")


def main() -> int:
    e2i = load_entity_ids()
    mention_edges = load_mention_edges()
    slice_sets = query_groups_v2()

    def group_of(k):
        triple_key = k[:3]
        is_slice = triple_key in slice_sets["lát"]
        has_edge = frozenset((e2i[k[0]], e2i[k[2]])) in mention_edges
        base = "lát" if is_slice else "thường"
        return "tổng", f"{base}·{'CÓ' if has_edge else 'KHÔNG'}"

    res = {}
    n_check = {}
    for name, f in MODELS:
        per = {g: [] for g in EDGE_GROUPS}
        n_keys = {g: None for g in EDGE_GROUPS}
        ok = True
        for s in SEEDS:
            p, fmt = f(s)
            if not Path(p).exists():
                print(f"WARNING {name} s{s}: MISSING {p}; skipping model")
                ok = False
                break
            r = {"wide": read_wide, "adamf": lambda x: read_adamf(x, e2i), "dump": read_dump}[fmt](
                str(p)
            )
            acc = {g: [] for g in EDGE_GROUPS}
            for k, rank in r.items():
                g_total, g_detail = group_of(k)
                acc["tổng"].append(1.0 / rank)
                acc[g_detail].append(1.0 / rank)
                acc["CÓ" if g_detail.endswith("CÓ") else "KHÔNG"].append(
                    1.0 / rank
                )  # overall by edge
            for g in EDGE_GROUPS:
                per[g].append(100 * st.fmean(acc[g]))
                if n_keys[g] is None:
                    n_keys[g] = len(acc[g])
        if not ok:
            continue
        res[name] = {g: st.fmean(per[g]) for g in EDGE_GROUPS}
        n_check[name] = n_keys

    # cross-check
    a = res["A3′* λ8 (headline)"]
    assert abs(a["tổng"] - 41.20) < 0.05, f"A3'* overall {a['tổng']:.2f} != 41.20"
    assert (
        n_check["A3′* λ8 (headline)"]["thường·CÓ"] == 1348
    ), f"n thường·CÓ {n_check['A3′* λ8 (headline)']['thường·CÓ']} != 1348"
    print(f"cross-check OK: A3'* overall {a['tổng']:.2f}, n regular-with-edge 1348")
    for name in res:
        print(f"  {name:22} " + "  ".join(f"{g}={res[name][g]:.2f}" for g in EDGE_GROUPS))

    # ------- write bang_canh_v2.tex + .md
    label_tex = {
        "A3′* λ8 (headline)": r"\textbf{MENTOR} (A3$'$*, $\lambda$=8, headline)",
        "nền* λ8": r"nền* ($\lambda$=8, ablation)",
        "MyGO+seg": "MyGO+seg",
        "MoMoK": "MoMoK",
        "TSAM": "TSAM",
        "AdaMF-MAT": "AdaMF-MAT",
        "DSoM re-impl.": r"DSoM re-impl.\ (approx.)",
        "TuckER (cấu trúc)": "TuckER (structure-only)",
    }
    gap_tucker = res["TuckER (cấu trúc)"]["CÓ"] - res["TuckER (cấu trúc)"]["KHÔNG"]  # +7.89
    L = [
        r"% SINH bởi scripts/v3/sinh_cham_canh_v2.py (TIN_278 B8) — mức CÓ/KHÔNG + DiD vs TuckER, headline λ8.",
        r"% Nguồn: ranks_v2/* + luoi_c nen_l8.0 + canh_v2 (cefc05167b7123d3). CẦN: booktabs.",
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\caption{Mức MRR (\%, 2 chiều = khoá) theo nhóm \textbf{CÓ/KHÔNG cạnh nhắc-tên} trên "
        r"ViMMKG~2.0, 8 mô hình. \emph{gap} = CÓ $-$ KHÔNG (khoảng cách nhắc-tên của chính mô hình); "
        r"\emph{gap$-$gap$_{\text{TuckER}}$} = phần vượt \textbf{độ khó hub của nền cấu trúc} "
        r"(TuckER gap = "
        + f"{gap_tucker:+.2f}"
        + r"). \textbf{MENTOR\_base+L} có DiD lớn nhất (+4.76); "
        r"các mô hình text mạnh (TSAM, nền*) cũng vượt TuckER ($\sim$+3.9), còn "
        r"MyGO/MoMoK/AdaMF/DSoM $\approx$ hoặc $\le$ TuckER. Cột KHÔNG (mention-free, T\_free) trải "
        r"$\sim$9.4 điểm giữa các mô hình.}",
        r"\label{tab:canh_v2}",
        r"\begin{tabular}{l r r r r}",
        r"\toprule",
        r"Mô hình & CÓ-cạnh & KHÔNG-cạnh & gap & gap$-$gap$_{\text{TuckER}}$ \\",
        r"\midrule",
    ]
    M = [
        "<!-- SINH bởi scripts/v3/sinh_cham_canh_v2.py (TIN_278 B8) — mức CÓ/KHÔNG + DiD vs TuckER. -->",
        "",
        "# Bảng mức CÓ/KHÔNG cạnh + gap−gap_TuckER — ViMMKG 2.0 (MRR %, 2 chiều, 3 seed), headline A3′* λ8",
        "",
        f"> gap_TuckER = {gap_tucker:+.2f} (độ khó hub nền cấu trúc). gap−gap_TuckER > 0 ⇒ vượt nền cấu trúc.",
        "",
        "| mô hình | CÓ-cạnh | KHÔNG-cạnh | gap | gap−gap_TuckER |",
        "|---|--:|--:|--:|--:|",
    ]
    for name, _ in MODELS:
        if name not in res:
            continue
        v = res[name]
        gap = v["CÓ"] - v["KHÔNG"]
        did = gap - gap_tucker
        best = name == "A3′* λ8 (headline)"
        dids = f"\\textbf{{{did:+.2f}}}" if best else f"{did:+.2f}"
        L.append(f"{label_tex[name]} & {v['CÓ']:.2f} & {v['KHÔNG']:.2f} & {gap:+.2f} & {dids} \\\\")
        M.append(
            f"| {name} | {v['CÓ']:.2f} | {v['KHÔNG']:.2f} | {gap:+.2f} | "
            f"{'**'+format(did,'+.2f')+'**' if best else format(did,'+.2f')} |"
        )
    L += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2pt}",
        r"\begin{minipage}{\linewidth}\footnotesize\raggedright",
        r"Mức CÓ/KHÔNG gộp hai chiều (lát+thường). \textbf{MENTOR\_base+L} có "
        r"gap$-$gap$_{\text{TuckER}}$ lớn nhất (+4.76); phần lớn khoảng cách này đã có ở \emph{nền*} "
        r"(+3.92, text/ảnh không-L) và ở TSAM (+3.96), nên bước +L chỉ thêm $\sim$+0.84 (khớp "
        r"$\Delta_L$ v2 yếu, Bảng~\ref{tab:trung_tam}). Các baseline yếu MyGO/MoMoK/AdaMF/DSoM "
        r"$\approx$ hoặc $\le$ TuckER. Cột mention-free (KHÔNG) trải $\sim$9.4 điểm. "
        r"Cạnh v2 \texttt{cefc05167b7123d3}.",
        r"\end{minipage}",
        r"\end{table}",
    ]
    (ROOT / "paper_b/bang_canh_v2.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    (ROOT / "paper_b/bang_canh_v2.md").write_text("\n".join(M) + "\n", encoding="utf-8")
    # numbers for the TuckER delta-L table (with minus without) and the per-level table
    out = {name: res[name] for name in res}
    (ROOT / "data/experimental/v3/cham_canh_v2_lambda8.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print("✅ ghi paper_b/bang_canh_v2.{tex,md} + cham_canh_v2_lambda8.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
