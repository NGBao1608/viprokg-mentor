"""Build data/wikipedia_attribution.tsv: Vietnamese Wikipedia page of every entity (CC BY-SA attribution).

For each Wikidata QID in ViMMKG2/entity2id.tsv: the `viwiki` sitelink title (Wikidata API), the page URL, and the
id + timestamp of the page's latest revision at retrieval time (Wikipedia API). The revision is the latest one on the
retrieval date; the stored description text was fetched earlier, so it may correspond to an older revision of the
same page (the page history lists all authors). Page titles come from `--titles_tsv` (qid, title) when given
(the titles recorded when the text was crawled), otherwise from the Wikidata `viwiki` sitelink. Entities without a Vietnamese page (and attribute-value nodes
`VAL::...`) get empty fields.

Polite crawling: identified User-Agent, <= 1 request/second, exponential backoff on 429/5xx, resumable checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER_AGENT = "ViProKG-attribution/1.0 (HCMUS thesis, research use; <= 1 request/second)"
WIKIDATA = "https://www.wikidata.org/w/api.php"
VIWIKI = "https://vi.wikipedia.org/w/api.php"
BATCH = 50


def get_json(base: str, params: dict, retries: int = 6) -> dict:
    url = base + "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        time.sleep(1.0)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001 - network errors are retried with backoff
            wait = 2 ** (attempt + 1)
            print(f"  retry {attempt + 1}/{retries} after {wait}s: {e}", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"giving up on {url[:120]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--entities", default=str(ROOT / "data/experimental/v3/ext/ViMMKG2/entity2id.tsv")
    )
    ap.add_argument("--out", default=str(ROOT / "data/wikipedia_attribution.tsv"))
    ap.add_argument("--checkpoint", default=str(ROOT / "wikipedia_attribution.ckpt.jsonl"))
    ap.add_argument(
        "--titles_tsv", default="", help="optional TSV (qid, title): titles recorded at crawl time"
    )
    a = ap.parse_args()

    with open(a.entities, encoding="utf-8") as f:
        qids = [r["ten"] for r in csv.DictReader(f, delimiter="\t")]
    real = [q for q in qids if q[:1] == "Q" and q[1:].isdigit()]
    known: dict[str, str] = {}
    if a.titles_tsv:
        with open(a.titles_tsv, encoding="utf-8") as f:
            known = {r["qid"]: r["title"] for r in csv.DictReader(f, delimiter="\t") if r["title"]}
    ckpt = Path(a.checkpoint)
    done: dict[str, dict] = {}
    if ckpt.exists():
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            done[d["qid"]] = d
    todo = [q for q in real if q not in done]
    print(f"{len(real)} QIDs, {len(done)} already done, {len(todo)} to fetch", flush=True)

    for i in range(0, len(todo), BATCH):
        batch = todo[i : i + BATCH]
        titles = {q: known[q] for q in batch if q in known}
        source = {q: "crawl" for q in titles}
        unknown = [q for q in batch if q not in known]
        if unknown:
            wd = get_json(
                WIKIDATA,
                {
                    "action": "wbgetentities",
                    "ids": "|".join(unknown),
                    "props": "sitelinks",
                    "sitefilter": "viwiki",
                    "format": "json",
                },
            )
            for q in unknown:
                ent = wd.get("entities", {}).get(q, {})
                t = ent.get("sitelinks", {}).get("viwiki", {}).get("title")
                if t:
                    titles[q] = t
                    source[q] = "wikidata"
        revs: dict[str, tuple[int, str]] = {}
        if titles:
            wp = get_json(
                VIWIKI,
                {
                    "action": "query",
                    "titles": "|".join(titles.values()),
                    "prop": "revisions",
                    "rvprop": "ids|timestamp",
                    "format": "json",
                    "formatversion": "2",
                },
            )
            norm = {n["from"]: n["to"] for n in wp.get("query", {}).get("normalized", [])}
            by_title = {
                p["title"]: p for p in wp.get("query", {}).get("pages", []) if "revisions" in p
            }
            for q, t in titles.items():
                p = by_title.get(norm.get(t, t))
                if p:
                    rv = p["revisions"][0]
                    revs[q] = (rv["revid"], rv["timestamp"])
        with open(ckpt, "a", encoding="utf-8") as f:
            for q in batch:
                t = titles.get(q, "")
                rev = revs.get(q)
                d = {
                    "qid": q,
                    "page_title": t,
                    "url": (
                        "https://vi.wikipedia.org/wiki/"
                        + urllib.parse.quote(t.replace(" ", "_"), safe="()_,:~-'")
                        if t
                        else ""
                    ),
                    "revision_id": rev[0] if rev else "",
                    "revision_timestamp": rev[1] if rev else "",
                    "title_source": source.get(q, ""),
                }
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
                done[q] = d
        print(f"  {min(i + BATCH, len(todo))}/{len(todo)}", flush=True)

    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(
            ["qid", "page_title", "url", "revision_id", "revision_timestamp", "title_source"]
        )
        for q in qids:
            d = done.get(q)
            w.writerow(
                [
                    q,
                    d["page_title"],
                    d["url"],
                    d["revision_id"],
                    d["revision_timestamp"],
                    d["title_source"],
                ]
                if d
                else [q, "", "", "", "", ""]
            )
    n_page = sum(1 for d in done.values() if d["page_title"])
    print(
        f"wrote {a.out}: {len(qids)} rows, {n_page} with a viwiki page; retrieved {datetime.now(timezone.utc):%Y-%m-%d}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
