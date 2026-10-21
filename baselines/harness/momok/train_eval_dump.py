"""Driver for MoMoK (github.com/zjukg/MoMoK) that runs the ORIGINAL model unchanged.

It reuses MoMoK's RelMoE model, corpus and ranking as they are, but applies the baseline contract:
  - best-valid: the checkpoint is selected on validation MRR (the original train.py:152-164 tracks
    the best on the 'test' split, i.e. peeks at test; this driver fixes that).
  - test ONCE: after selection the best checkpoint is loaded and scored on test exactly once.
  - per-key filtered ranks in both directions are dumped to ranks_test.tsv (stable QID/PID keys).
  - config.json (lenh / seed / hash_dau_vao) + moi_truong.json (torch / cuda / code sha).

The MoMoK model/score code is not modified. The ONLY change inside the repo is a
`corpus._dump_ranks` branch in data_loader.py that exposes the per-triple ranks that are already
computed (the ranking computation is unchanged). Everything else lives in this driver, outside the repo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

# ----- original MoMoK config (verbatim copy of train.py:16-56, values unchanged) -----
ORIGINAL_CONFIG: dict[str, Any] = {
    "lr": 0.0005,
    "dropout_gat": 0.3,
    "dropout": 0.3,
    "cuda": 0,
    "epochs_gat": 3000,
    "epochs": 2000,
    "weight_decay_gat": 1e-5,
    "weight_decay": 0,
    "seed": 10010,
    "model": "RMoE",
    "num_layers": 3,
    "dim": 256,
    "r_dim": 256,
    "k_w": 10,
    "k_h": 20,
    "n_heads": 2,
    "dataset": "DB15K",
    "pre_trained": 0,
    "encoder": 0,
    "image_features": 1,
    "text_features": 1,
    "patience": 5,
    "eval_freq": 100,
    "lr_reduce_freq": 500,
    "gamma": 1.0,
    "bias": 1,
    "neg_num": 2,
    "neg_num_gat": 2,
    "alpha": 0.2,
    "alpha_gat": 0.2,
    "out_channels": 32,
    "kernel_size": 3,
    "batch_size": 1024,
    "save": 1,
    "n_exp": 3,
    "mu": 0.0001,
    "img_dim": 256,
    "txt_dim": 256,
}

# Repo files whose content pins the code version.
PINNED_CODE_FILES = [
    "train.py",
    "models/RelMoE.py",
    "models/model.py",
    "models/modules.py",
    "layers/layer.py",
    "utils/data_loader.py",
    "utils/data_util.py",
]


def sha16(path: str) -> str:
    """First 16 hex chars of a file's sha256 (stable key for hash_dau_vao)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def set_deterministic(seed: int) -> None:
    """Fix every randomness source so that rerunning with the SAME seed is bit-identical.

    Uses explicit numpy/torch seeds (deterministic across processes, unlike str.__hash__).
    MoMoK breaks ties with np.random.randint, so the numpy seed decides ranks.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def build_args(cli: argparse.Namespace) -> argparse.Namespace:
    """Build the full Namespace: ORIGINAL_CONFIG overridden by the CLI flags that were passed."""
    a = argparse.Namespace(**ORIGINAL_CONFIG)
    for k, v in vars(cli).items():
        if v is not None and k in ORIGINAL_CONFIG:
            setattr(a, k, v)
    # driver-specific flags (not part of the original config)
    a.out_dir = cli.out_dir
    a.momok_dir = cli.momok_dir
    a.mention_edges = cli.mention_edges
    a.name_prefix = cli.name_prefix
    a.ent_names = cli.ent_names
    a.rel_names = cli.rel_names
    return a


def read_id_to_name(path: str) -> dict[int, str]:
    """Read a tsv with header 'name<TAB>id[...]' -> {id: name}. Used for the v2 dump (id -> QID/PID);
    names can contain spaces, so they are NOT put into the MoMoK dataset."""
    d = {}
    with open(path) as f:
        next(f)
        for line in f:
            c = line.rstrip("\n").split("\t")
            d[int(c[1])] = c[0]
    return d


def main() -> int:
    p = argparse.ArgumentParser(description="MoMoK TAY2 driver (best-valid + dump per-key)")
    p.add_argument(
        "--momok_dir", required=True, help="MoMoK repo directory (contains datasets/, models/)"
    )
    p.add_argument("--dataset", required=True, help="dataset name under datasets/ (e.g. MKG-W)")
    p.add_argument("--out_dir", required=True, help="where to write ranks_test.tsv + config.json")
    p.add_argument(
        "--name_prefix", "--tien_to", default="momok", help="prefix of the row directory name"
    )
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--eval_freq", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--cuda", type=int, default=None, help=">=0 uses that GPU id; <0 uses CPU")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--mu", type=float, default=None)
    p.add_argument("--dim", type=int, default=None)
    p.add_argument("--r_dim", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument(
        "--mention_edges",
        "--canh",
        default=None,
        help="v2 canh_nhac_ten file (only used to record its hash)",
    )
    p.add_argument(
        "--ent_names",
        default=None,
        help="tsv id -> QID for the dump (v2); if absent MoMoK names are used",
    )
    p.add_argument(
        "--rel_names",
        default=None,
        help="tsv id -> PID for the dump (v2); if absent MoMoK names are used",
    )
    cli = p.parse_args()

    lenh = "python " + " ".join(map(_q, sys.argv))
    args = build_args(cli)

    momok_dir = os.path.abspath(args.momok_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # RelMoE loads datasets/<ds>/... by RELATIVE path, so chdir into the repo.
    sys.path.insert(0, momok_dir)
    os.chdir(momok_dir)

    from models.modules import MIEstimator  # noqa: E402
    from models.RelMoE import RelMoE  # noqa: E402  (sau chdir/sys.path)
    from utils.data_loader import ConvECorpus  # noqa: E402
    from utils.data_util import load_data  # noqa: E402

    set_deterministic(args.seed)

    args.device = ("cuda:" + str(args.cuda)) if int(args.cuda) >= 0 else "cpu"
    on_gpu = int(args.cuda) >= 0 and torch.cuda.is_available()
    if int(args.cuda) >= 0 and not torch.cuda.is_available():
        print("[WARNING] --cuda >=0 but no GPU available: running on CPU", flush=True)
        args.device = "cpu"

    # ----- load data + features (verbatim from train.py:73-83) -----
    ent2id, rel2id, img_feat, txt_feat, tr, va, te = load_data(args.dataset)
    print(
        f"[data] train={len(tr[0])} valid={len(va[0])} test={len(te[0])} "
        f"ent={len(ent2id)} rel={len(rel2id)}",
        flush=True,
    )
    corpus = ConvECorpus(args, tr, va, te, ent2id, rel2id)
    if args.image_features:
        args.img = F.normalize(torch.Tensor(img_feat), p=2, dim=1)
    if args.text_features:
        args.desp = F.normalize(torch.Tensor(txt_feat), p=2, dim=1)
    args.entity2id = ent2id
    args.relation2id = rel2id
    print(
        f"[feat] img={tuple(torch.as_tensor(img_feat).shape)} "
        f"txt={tuple(torch.as_tensor(txt_feat).shape)}",
        flush=True,
    )

    # ----- build model / estimator / optimizer (verbatim from train_decoder:93-105) -----
    model = RelMoE(args)
    args.img_dim = model.img_dim
    args.txt_dim = model.txt_dim
    estimator = MIEstimator(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.gamma)
    optimizer_mi = torch.optim.Adam(
        estimator.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    if args.device != "cpu":
        model = model.to(args.device)
        estimator = estimator.to(args.device)

    # ----- training loop (verbatim from train_decoder:114-145) + SELECTION ON VALID -----
    best_val_mrr = -1.0
    best_epoch = -1
    best_test_mrr = (
        -1.0
    )  # oracle-on-test, only for comparison with the reference, never used for selection
    best_test_epoch = -1
    best_path = os.path.join(out_dir, "best.pth")
    best_test_path = os.path.join(
        out_dir, "best_test.pth"
    )  # oracle checkpoint (test is dumped twice)
    patience_count = 0
    corpus.batch_size = args.batch_size
    corpus.neg_num = args.neg_num
    t_total = time.time()
    epoch_seconds = []  # time per epoch (the last epoch gives the steady-state time)
    for epoch in range(args.epochs):
        te = time.time()
        model.train()
        epoch_loss, epoch_mi_loss = [], []
        corpus.shuffle()
        for batch_num in range(corpus.max_batch_num):
            estimator.eval()
            optimizer.zero_grad()
            train_indices, train_values = corpus.get_batch(batch_num)
            train_indices = torch.LongTensor(train_indices)
            if args.device != "cpu":
                train_indices = train_indices.to(args.device)
                train_values = train_values.to(args.device)
            output, embeddings = model.forward(train_indices)
            loss = model.loss_func(output, train_values) + args.mu * estimator(embeddings)
            loss.backward()
            optimizer.step()
            estimator.train()
            optimizer_mi.zero_grad()
            with torch.no_grad():
                embeddings = model.get_batch_embeddings(train_indices)
            estimator_loss = estimator.train_estimator(embeddings)
            estimator_loss.backward()
            optimizer_mi.step()
            epoch_loss.append(loss.data.item())
            epoch_mi_loss.append(estimator_loss.item())
        lr_scheduler.step()

        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            corpus._dump_ranks = False
            with torch.no_grad():
                val_metrics, _ = corpus.get_validation_pred(model, "val")
                test_metrics_now, _ = corpus.get_validation_pred(model, "test")
            vm = val_metrics["MRR"]
            tm = test_metrics_now["MRR"]
            if tm > best_test_mrr:  # oracle-on-test (only for comparison with the reference)
                best_test_mrr = tm
                best_test_epoch = epoch + 1
                torch.save(model.state_dict(), best_test_path)
            print(
                f"[epoch {epoch + 1:04d}] loss={sum(epoch_loss):.4f} "
                f"valid_MRR={vm:.5f} (best {best_val_mrr:.5f}@e{best_epoch}) "
                f"test_MRR={tm:.5f} (best-test {best_test_mrr:.5f}@e{best_test_epoch})",
                flush=True,
            )
            if vm > best_val_mrr:
                best_val_mrr = vm
                best_epoch = epoch + 1
                torch.save(model.state_dict(), best_path)
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= args.patience:
                    print(
                        f"[early-stop] patience {args.patience} exhausted at e{epoch + 1}",
                        flush=True,
                    )
                    break
        epoch_seconds.append(round(time.time() - te, 3))
    train_s = time.time() - t_total
    print(
        f"[training done] {train_s:.1f}s, best valid MRR={best_val_mrr:.5f} @e{best_epoch}",
        flush=True,
    )

    # ----- TEST ONCE on the best checkpoint + per-key dump -----
    if best_epoch < 0:
        # never evaluated (epochs < eval_freq): use the current model
        torch.save(model.state_dict(), best_path)
        best_epoch = args.epochs
    # Output keys: v2 maps id -> QID/PID through the ext files (names can contain spaces);
    # W/Y use the MoMoK names (full-URL QID) already present in the corpus.
    id2ent = read_id_to_name(args.ent_names) if args.ent_names else corpus.id2entity
    id2rel = read_id_to_name(args.rel_names) if args.rel_names else corpus.id2relation
    test_triples = corpus.test_triples

    def dump_test(ckpt: str, path: str) -> float:
        """Load a checkpoint, score test, write per-key ranks for both directions to TSV; return metrics."""
        model.load_state_dict(torch.load(ckpt, map_location=args.device))
        model.eval()
        corpus._dump_ranks = True
        with torch.no_grad():
            tm_, _, (rh, rt) = corpus.get_validation_pred(model, "test")
        assert (
            len(rh) == len(rt) == len(test_triples)
        ), f"triple count mismatch: rh={len(rh)} rt={len(rt)} test={len(test_triples)}"
        with open(path, "w") as f:
            f.write("s_qid\tp_pid\to_qid\thead_rank\ttail_rank\n")
            for i, (h, r, t) in enumerate(test_triples):
                hr, tr_ = int(rh[i]), int(rt[i])
                assert hr >= 1 and tr_ >= 1, f"rank < 1 at row {i}"
                f.write(f"{id2ent[h]}\t{id2rel[r]}\t{id2ent[t]}\t{hr}\t{tr_}\n")
        print(f'[dump] {path} ({len(test_triples)} triple, MRR test={tm_["MRR"]:.5f})', flush=True)
        return tm_

    # dump test twice: (1) best-valid = the submitted number; (2) best-test oracle = for comparison with the reference
    test_metrics = dump_test(best_path, os.path.join(out_dir, "ranks_test.tsv"))
    test_mrr_besttest_dump = test_metrics["MRR"]
    if os.path.exists(best_test_path):
        test_mrr_besttest_dump = dump_test(
            best_test_path, os.path.join(out_dir, "ranks_test_besttest.tsv")
        )["MRR"]

    # ----- config.json -----
    path = f"datasets/{args.dataset}/"
    inputs = {
        "train.txt": path + "train.txt",
        "valid.txt": path + "valid.txt",
        "test.txt": path + "test.txt",
        "entity2id.txt": path + "entity2id.txt",
        "relation2id.txt": path + "relation2id.txt",
        "img_features.pth": path + "img_features.pth",
        "text_features.pth": path + "text_features.pth",
    }
    hash_dau_vao = {}
    for name, fp in inputs.items():
        if os.path.exists(fp):
            hash_dau_vao[name] = sha16(fp)
    if args.mention_edges and os.path.exists(args.mention_edges):
        hash_dau_vao["canh_nhac_ten"] = sha16(args.mention_edges)
    code_sha = {fp: sha16(fp) for fp in PINNED_CODE_FILES if os.path.exists(fp)}

    config = {
        "lenh": lenh,
        "seed": args.seed,
        "hash_dau_vao": hash_dau_vao,
        "model": "MoMoK (RMoE)",
        "dataset": args.dataset,
        "best_epoch": best_epoch,  # epoch selected on VALID (submitted number)
        "valid_MRR": best_val_mrr,
        "test_MRR_submitted": test_metrics[
            "MRR"
        ],  # test at the best-valid checkpoint = submitted number (ranks)
        "best_test_MRR": best_test_mrr,  # oracle-on-test (comparison with the original reference)
        "best_test_epoch": best_test_epoch,
        "test_MRR_besttest_dump": test_mrr_besttest_dump,  # MRR from the oracle dump (test dumped twice)
        "test_metrics": test_metrics,
        "code_sha": code_sha,
        "protocol": "best-valid select, test-once (HĐ §3.1); "
        "filter train∪valid∪test; đổi split chọn từ test→valid so với train.py gốc",
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    environment = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if on_gpu else None,
        "train_giay": round(train_s, 1),
        "epoch_giay": epoch_seconds,
        "sha_train_py": code_sha.get("train.py"),
        "model_cfg_sha": hashlib.sha256(json.dumps(code_sha, sort_keys=True).encode()).hexdigest()[
            :16
        ],
    }
    with open(os.path.join(out_dir, "moi_truong.json"), "w") as f:
        json.dump(environment, f, ensure_ascii=False, indent=2)
    print("[done] config.json + moi_truong.json written", flush=True)
    return 0


def _q(s: str) -> str:
    """Quote one token safely for the printed command line."""
    return s if all(c.isalnum() or c in "./_-=:" for c in s) else repr(s)


if __name__ == "__main__":
    raise SystemExit(main())
