"""Build Table 1 (dataset statistics, label tab:dataset) from the released data files.

Every ViProKG number is measured here from `data/` and asserted against the frozen value; the profile row of the three
standard benchmarks is recomputed on their released split files (pass `--bench_root`, a folder holding
DB15K/, MKG-W/, MKG-Y/ with train.txt, valid.txt, test.txt and entities.txt / relations.txt).
Entity / relation / triple counts of the three standard benchmarks are the figures of the original papers
(DB15K: Liu et al. 2019; MKG-W / MKG-Y: Xu et al. 2022) and are checked against the released files.

High-cardinality many-to-one query profile: for every test triple (h, r, t), C_head = set of heads h' with (h', r, t) in
train + valid + test; reported = share of test triples with |C_head| >= 10, and the median |C_head|.
Writes tables/bang_dataset.tex (backup first if it exists). Exit code 0 = every assert passed.
"""

from __future__ import annotations

import argparse
import re
import shutil
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "data" / "experimental" / "v3" / "ext" / "ViMMKG2"
MANIFEST = ROOT / "data" / "final" / "ViMMKG2_core" / "manifest.tsv"
EDGES = ROOT / "data" / "experimental" / "v3" / "canh_v2" / "v10_canh_nhac_ten_ViMMKG2_v2.tsv"
OUT = ROOT / "tables" / "bang_dataset.tex"
# original-paper figures (entities, relations, triples)
PAPER = {"DB15K": (12842, 279, 99028), "MKG-W": (15000, 169, 42746), "MKG-Y": (15000, 28, 26638)}
THRESHOLD = 10


def triples(path: Path) -> list[tuple[str, str, str]]:
    """Read a whitespace-separated (h, r, t) split file."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        a = line.split()
        if len(a) >= 3:
            out.append((a[0], a[1], a[2]))
    return out


def profile(folder: Path) -> tuple[float, float]:
    """Share of test triples with |C_head| >= THRESHOLD and median |C_head| (heads over train+valid+test)."""
    tr, va, te = (triples(folder / f"{s}.txt") for s in ("train", "valid", "test"))
    heads = defaultdict(set)
    for h, r, t in tr + va + te:
        heads[(r, t)].add(h)
    sizes = [len(heads[(r, t)]) for _, r, t in te]
    return sum(s >= THRESHOLD for s in sizes) / len(sizes), statistics.median(sizes)


def rows(path: Path) -> list[list[str]]:
    """TSV rows without header."""
    return [l.split("\t") for l in path.read_text(encoding="utf-8").splitlines()[1:] if l]


def viprokg() -> dict:
    """Measure the ViProKG numbers of Table 1."""
    ent = rows(V2 / "entity2id.tsv")
    n_entity = sum(r[2] == "entity" for r in ent)
    n_value = len(ent) - n_entity
    n_rel = len(rows(V2 / "relation2id.tsv"))
    split = {s: len(triples(V2 / f"{s}.txt")) for s in ("train", "valid", "test")}
    img = rows(V2 / "image_index.tsv")
    with_img = sum(r[2] == "1" for r in img)
    man = MANIFEST.read_text(encoding="utf-8").splitlines()
    head = man[0].split("\t")
    i_page, i_why = head.index("trang_goc"), head.index("ly_do_loai")
    kept, removed = [], {}
    for line in man[1:]:
        c = line.split("\t")
        c += [""] * (len(head) - len(c))
        if c[i_why]:
            removed[c[i_why]] = removed.get(c[i_why], 0) + 1
        else:
            kept.append(urlparse(c[i_page]).netloc.lower())
    commons = sum(h.endswith("commons.wikimedia.org") for h in kept)
    flickr = sum("flickr.com" in h for h in kept)
    n_lat = len(triples(V2 / "test_lat.txt"))
    n_edges = (
        len(EDGES.read_text(encoding="utf-8").splitlines()) - 1
    )  # header excluded (CLAUDE.md án lệ #6)
    share, med = profile(V2)
    return dict(
        entity=n_entity,
        value=n_value,
        rel=n_rel,
        split=split,
        total=sum(split.values()),
        with_img=with_img,
        kept=len(kept),
        commons=commons,
        flickr=flickr,
        other=len(kept) - commons - flickr,
        removed=removed,
        lat=n_lat,
        edges=n_edges,
        share=share,
        median=med,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--bench_root", required=True, help="folder with DB15K/, MKG-W/, MKG-Y/ split files"
    )
    a = ap.parse_args()
    v = viprokg()
    # frozen values (BAN_GIAO_NAO_V17 §2, MANIFEST §4); any drift stops the build
    assert v["entity"] == 8207 and v["value"] == 48 and v["rel"] == 81, v
    assert v["split"] == {"train": 29355, "valid": 3610, "test": 2637} and v["total"] == 35602, v
    assert v["with_img"] == 7790 == v["kept"], v
    assert (
        round(100 * v["commons"] / v["kept"], 2) == 94.80 and v["flickr"] == 401 and v["other"] == 4
    ), v
    assert sum(v["removed"].values()) == 239, v["removed"]
    assert round(100 * v["lat"] / v["split"]["test"], 2) == 37.43 and v["edges"] == 44770, v
    assert round(100 * v["share"], 1) == 89.5 and v["median"] == 96, v
    bench = {}
    for b, (ne, nr, nt) in PAPER.items():
        d = Path(a.bench_root) / b
        n_ent = len(
            [l for l in (d / "entities.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
        )
        n_rel = len(
            [l for l in (d / "relations.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
        )
        n_tri = sum(len(triples(d / f"{s}.txt")) for s in ("train", "valid", "test"))
        assert (n_ent, n_rel, n_tri) == (ne, nr, nt), (b, n_ent, n_rel, n_tri)
        bench[b] = profile(d)
    pct = lambda x: f"{100 * x:.1f}\\%"
    cov = 100 * v["with_img"] / v["entity"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Dataset statistics. ViProKG counts are measured on the released files; counts for DB15K~\cite{db15k} and "
        r"MKG-W / MKG-Y~\cite{mmrns} are from the original papers. Last row: share of test triples with $|C_{\text{head}}| \geq 10$ "
        r"and median $|C_{\text{head}}|$, computed by us with the same code on every dataset.}",
        r"\label{tab:dataset}",
        r"\begin{tabular}{l r r r r}",
        r"\toprule",
        r" & \textbf{ViProKG} & DB15K & MKG-W & MKG-Y \\",
        r"\midrule",
        f"Entities & {v['entity']:,}$^{{\\dagger}}$ & {PAPER['DB15K'][0]:,} & {PAPER['MKG-W'][0]:,} & {PAPER['MKG-Y'][0]:,} \\\\",
        f"Relations & {v['rel']} & {PAPER['DB15K'][1]} & {PAPER['MKG-W'][1]} & {PAPER['MKG-Y'][1]} \\\\",
        f"Triples & {v['total']:,} & {PAPER['DB15K'][2]:,} & {PAPER['MKG-W'][2]:,} & {PAPER['MKG-Y'][2]:,} \\\\",
        f"\\quad train / valid / test & {v['split']['train']:,}/{v['split']['valid']:,}/{v['split']['test']:,} & -- & -- & -- \\\\",
        r"Text language & Vietnamese & English & English & English \\",
        r"Image source & Commons, Flickr, other$^{\ddagger}$ & web & web & web \\",
        f"Entities with an image & {v['with_img']:,} ({cov:.1f}\\%) & -- & -- & -- \\\\",
        r"Per-image license log & yes & no & no & no \\",
        f"High-card.\ profile ($\\geq$10 / median) & {pct(v['share'])} / {v['median']:g} & "
        + " & ".join(
            f"{pct(s)} / {m:g}" for s, m in (bench[b] for b in ("DB15K", "MKG-W", "MKG-Y"))
        )
        + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2pt}",
        r"\begin{minipage}{\linewidth}\footnotesize\raggedright",
        f"$^{{\\dagger}}$Plus {v['value']} value nodes of the attribute slice ({v['entity'] + v['value']:,} nodes in the ranking space). $^{{\\ddagger}}$Europeana, the Metropolitan Museum and iNaturalist ({v['other']} images).",
        r"\end{minipage}",
        r"\end{table}",
        "",
    ]
    text = re.sub(
        r"(?<=\d),(?=\d{3})", "{,}", "\n".join(lines)
    )  # dấu phẩy nghìn kiểu LaTeX, chỉ giữa chữ số
    if OUT.exists():
        bak = OUT.with_name(OUT.name + ".bak-20261019e")
        if not bak.exists():
            shutil.copy2(OUT, bak)
    OUT.write_text(text, encoding="utf-8")
    print(
        "ViProKG:",
        {
            k: v[k]
            for k in (
                "entity",
                "value",
                "rel",
                "total",
                "with_img",
                "commons",
                "flickr",
                "other",
                "lat",
                "edges",
            )
        },
        "removed",
        v["removed"],
        f"commons {100 * v['commons'] / v['kept']:.2f}%",
        f"coverage {cov:.2f}%",
        f"profile {100 * v['share']:.2f}% / {v['median']}",
    )
    for b, (s, m) in bench.items():
        print(f"{b}: profile {100 * s:.2f}% / {m}")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
