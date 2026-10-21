"""Driver for TSAM (repos/TSAM-main) that runs the ORIGINAL model unchanged.

Reuses TSAM's VTKG + TSAM + calculate_rank as they are, under the baseline contract:
  - best-valid: the checkpoint is selected on validation MRR (the original Train.py:212 tracks the
    best on TEST, i.e. peeks at test).
  - test ONCE: load the best checkpoint, score test, dump per-key filtered ranks for both directions.
  - writes BOTH best-valid (the submitted number) and the best-test oracle (for comparison with the
    published number) together with the epoch.
  - config.json (lenh / seed / hash_dau_vao) + moi_truong.json.
The protocol lives HERE (the driver); TSAM model/score code is not modified (0 diff in the repo for W).
The training loop and model construction are verbatim from Train.py:98-152.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

# default hyper-parameters, verbatim from the Train.py argparse
DEFAULTS = dict(
    lr=5e-4,
    dim=200,
    num_epoch=100,
    valid_epoch=100,
    exp="TSAM",
    num_layer_enc_ent=1,
    num_layer_enc_rel=1,
    num_layer_dec=2,
    num_head=2,
    hidden_dim=200,
    dropout=0.01,
    emb_dropout=0.9,
    vis_dropout=0.4,
    txt_dropout=0.1,
    smoothing=0.0,
    batch_size=2048,
    decay=0.0,
    max_img_num=3,
    step_size=50,
    max_vis_token=8,
    max_txt_token=8,
    score_function="tucker",
    mu=0.0,
    patience=0,
)  # patience>0 (number of validations without improvement) triggers early stop (best-valid stays correct)
PINNED_FILES = [
    "Train.py",
    "TSAM.py",
    "dataset.py",
    "utils.py",
    "merge_tokens.py",
    "Tucker.py",
    "Rotate.py",
    "TransELayer.py",
    "CLossModel.py",
]


def sha16(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:16]


def read_id_to_name(path: str) -> dict:
    d = {}
    with open(path) as f:
        next(f)
        for line in f:
            c = line.rstrip("\n").split("\t")
            d[int(c[1])] = c[0]
    return d


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument(
        "--ent_names", default=None, help="tsv id -> QID (v2); if absent TSAM names are used"
    )
    p.add_argument("--rel_names", default=None)
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k}", type=type(v), default=v)
    a = p.parse_args()
    lenh = "python " + " ".join(sys.argv)

    repo = os.path.abspath(a.repo)
    out = os.path.abspath(a.out_dir)
    os.makedirs(out, exist_ok=True)
    os.chdir(repo)
    sys.path.insert(0, repo)

    # deterministic (TSAM tie-break in calculate_rank is the midpoint, no RNG)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if os.environ.get(
        "TSAM_CPU_SMOKE"
    ):  # local CPU smoke only; the GPU run does not set this env var
        torch.Tensor.cuda = lambda self, *a, **k: self
        torch.nn.Module.cuda = lambda self, *a, **k: self
        torch.cuda.is_available = lambda: False
        torch.cuda.empty_cache = lambda *a, **k: None
        torch.cuda.set_device = lambda *a, **k: None
        torch.cuda.manual_seed_all = lambda *a, **k: None

    from dataset import VTKG
    from merge_tokens import get_entity_textual_tokens, get_entity_visual_tokens
    from TSAM import TSAM
    from utils import calculate_rank

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())

    # ---- build KG + tokens + model (verbatim from Train.py:98-130) ----
    KG = VTKG(a.data, logger, max_vis_len=a.max_img_num)
    KG_Loader = torch.utils.data.DataLoader(KG, batch_size=a.batch_size, shuffle=True)
    vti, vkm = get_entity_visual_tokens(dataset=a.data, max_num=a.max_vis_token)
    vti = vti.cuda()
    tti, tkm = get_entity_textual_tokens(dataset=a.data, max_num=a.max_txt_token)
    tti = tti.cuda()
    model = TSAM(
        num_ent=KG.num_ent,
        num_rel=KG.num_rel,
        ent_vis_mask=vkm,
        ent_txt_mask=tkm,
        dim_str=a.dim,
        num_head=a.num_head,
        dim_hid=a.hidden_dim,
        num_layer_enc_ent=a.num_layer_enc_ent,
        num_layer_enc_rel=a.num_layer_enc_rel,
        num_layer_dec=a.num_layer_dec,
        dropout=a.dropout,
        emb_dropout=a.emb_dropout,
        vis_dropout=a.vis_dropout,
        txt_dropout=a.txt_dropout,
        visual_token_index=vti,
        text_token_index=tti,
        score_function=a.score_function,
    ).cuda()
    loss_fn = nn.CrossEntropyLoss(label_smoothing=a.smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, a.step_size, T_mult=2
    )

    def score_split(split_triples):
        """Per-key filtered ranks for both directions (verbatim from Train.py:170-199)."""
        ent_embs, rel_embs, _ = model()
        rows = []
        for h, r, t in split_triples:
            hs = (
                model.score(
                    ent_embs,
                    rel_embs,
                    torch.tensor(
                        [[KG.num_ent + KG.num_rel, r + KG.num_ent, t + KG.num_rel]]
                    ).cuda(),
                )[0]
                .detach()
                .cpu()
                .numpy()
            )
            hr = calculate_rank(hs, h, KG.filter_dict[(-1, r, t)])
            ts = (
                model.score(
                    ent_embs,
                    rel_embs,
                    torch.tensor(
                        [[h + KG.num_rel, r + KG.num_ent, KG.num_ent + KG.num_rel]]
                    ).cuda(),
                )[0]
                .detach()
                .cpu()
                .numpy()
            )
            tr = calculate_rank(ts, t, KG.filter_dict[(h, r, -1)])
            rows.append((h, r, t, int(hr), int(tr)))
        return rows

    def mrr_of(rows):
        rk = np.array([x[3] for x in rows] + [x[4] for x in rows])
        return float(np.mean(1.0 / rk))

    best_valid = -1.0
    best_epoch = -1
    best_test_oracle = -1.0
    bt_epoch = -1
    best_path = os.path.join(out, "best.pth")
    best_test_path = os.path.join(out, "best_test.pth")
    t0 = time.time()
    epoch_seconds = []  # time per epoch (the last epoch gives the steady-state time)
    no_improve = 0
    for epoch in range(1, a.num_epoch + 1):
        te = time.time()
        model.train()
        for batch, label in KG_Loader:
            ent_embs, rel_embs, closs = model()
            scores = model.score(ent_embs, rel_embs, batch.cuda())
            loss = loss_fn(scores, label.cuda()) + closs * 0.01
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
        scheduler.step()
        if epoch % a.valid_epoch == 0:
            model.eval()
            with torch.no_grad():
                vm = mrr_of(score_split(KG.valid))
                tm = mrr_of(score_split(KG.test))
            if tm > best_test_oracle:
                best_test_oracle = tm
                bt_epoch = epoch
                torch.save(model.state_dict(), best_test_path)
            if vm > best_valid:
                best_valid = vm
                best_epoch = epoch
                no_improve = 0
                torch.save(model.state_dict(), best_path)
            else:
                no_improve += 1
            print(
                f"[e{epoch}] valid_MRR={vm:.5f} (best {best_valid:.5f}@e{best_epoch}) "
                f"test_MRR={tm:.5f} (best-test {best_test_oracle:.5f}@e{bt_epoch}) ni={no_improve}",
                flush=True,
            )
            model.train()
            if a.patience > 0 and no_improve >= a.patience:
                print(
                    f"[early-stop] valid did not improve {a.patience} times: stopping at e{epoch} "
                    f"(best-valid @e{best_epoch})",
                    flush=True,
                )
                epoch_seconds.append(round(time.time() - te, 3))
                break
        epoch_seconds.append(round(time.time() - te, 3))
    train_s = time.time() - t0

    # ---- TEST ONCE on best-valid + per-key dump ----
    if best_epoch < 0:
        torch.save(model.state_dict(), best_path)
        best_epoch = a.num_epoch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    id2ent = (
        read_id_to_name(a.ent_names) if a.ent_names else {i: n for i, n in enumerate(KG.id2ent)}
    )
    id2rel = (
        read_id_to_name(a.rel_names) if a.rel_names else {i: n for i, n in enumerate(KG.id2rel)}
    )

    def dump_test(ckpt: str, path: str) -> float:
        """Load a checkpoint, score test ONCE, write per-key ranks for both directions to TSV; return MRR."""
        model.load_state_dict(torch.load(ckpt, map_location=dev))
        model.eval()
        with torch.no_grad():
            rows = score_split(KG.test)
        with open(path, "w") as f:
            f.write("s_qid\tp_pid\to_qid\thead_rank\ttail_rank\n")
            for h, r, t, hr, tr in rows:
                assert hr >= 1 and tr >= 1
                f.write(f"{id2ent[h]}\t{id2rel[r]}\t{id2ent[t]}\t{hr}\t{tr}\n")
        mrr = mrr_of(rows)
        print(f"[dump] {path} ({len(rows)} triple, test MRR={mrr:.5f})", flush=True)
        return mrr

    # dump test twice: (1) best-valid = the submitted number; (2) best-test oracle = for comparison with the published number
    test_mrr = dump_test(best_path, os.path.join(out, "ranks_test.tsv"))
    test_mrr_oracle_dump = test_mrr
    if os.path.exists(best_test_path):
        test_mrr_oracle_dump = dump_test(
            best_test_path, os.path.join(out, "ranks_test_besttest.tsv")
        )

    data_dir = f"data/{a.data}/"
    inputs = {
        k: data_dir + k
        for k in [
            "entities.txt",
            "relations.txt",
            "entity2id.txt",
            "relation2id.txt",
            "train.txt",
            "valid.txt",
            "test.txt",
        ]
    }
    inputs["tokens-visual.json"] = f"tokens/{a.data}-visual.json"
    inputs["tokens-textual.json"] = f"tokens/{a.data}-textual.json"
    inputs["visual.pth"] = "tokens/visual.pth"
    inputs["textual.pth"] = "tokens/textual.pth"
    hash_dau_vao = {k: sha16(v) for k, v in inputs.items() if os.path.exists(v)}
    if a.ent_names and os.path.exists(a.ent_names):
        hash_dau_vao["ent_names"] = sha16(a.ent_names)
    code_sha = {f: sha16(f) for f in PINNED_FILES if os.path.exists(f)}

    json.dump(
        {
            "lenh": lenh,
            "seed": a.seed,
            "hash_dau_vao": hash_dau_vao,
            "model": "TSAM",
            "dataset": a.data,
            "best_epoch": best_epoch,
            "valid_MRR": best_valid,
            "test_MRR_submitted": test_mrr,
            "best_test_MRR": best_test_oracle,
            "best_test_epoch": bt_epoch,
            "test_MRR_besttest_dump": test_mrr_oracle_dump,
            "code_sha": code_sha,
            "protocol": "best-valid select, test-once; filter train∪valid∪test; "
            "đổi split chọn test→valid so Train.py gốc",
        },
        open(os.path.join(out, "config.json"), "w"),
        ensure_ascii=False,
        indent=2,
    )
    json.dump(
        {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "train_giay": round(train_s, 1),
            "epoch_giay": epoch_seconds,
            "sha_train_py": code_sha.get("Train.py"),
        },
        open(os.path.join(out, "moi_truong.json"), "w"),
        ensure_ascii=False,
        indent=2,
    )
    print("[done] config.json + moi_truong.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
