#!/usr/bin/env python3
"""Run grid for the fusion-gate model (Kaggle GPU).

Idempotent: a run whose ranks file exists is skipped. Environment: EPOCHS (300), VALID_EPOCH (50),
ONLY (substring filter on run names), DRY (1 = list the runs only). Each run dumps ranks and, for
gated configurations, the gate values. Run from the model directory (data/ViMMKG; feature paths via env).

Grid (28 runs):
  - 4 gate modes {off, blind, prior_frozen, prior_init} x 3 seeds (d_r=64, default view)
  - no_image x 3 seeds + structure_only x 3 seeds (subtrahend of the image delta + sanity)
  - full-view gate: prior_init on V-full x 3 seeds
  - tuning: lambda_align=0.5 and lr=1e-3 (prior_init and no_image), seed 42, chosen on validation
  - capacity: d_r=256 (prior_init and no_image), seed 42
  - corrupt-provenance: prior_init with shuffled tiers x 3 seeds (last)
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

SEEDS = [42, 2024, 123]
FEAT = os.environ.get("FEAT_DIR", "feats")  # directory holding T.pt / V-default.pt / V-full.pt
TP = f"{FEAT}/T.pt"
VD = f"{FEAT}/V-default.pt"
VF = f"{FEAT}/V-full.pt"


def build_runs():
    R = []

    def add(
        name, gate="gate_prior_init", ut=1, ui=1, v=VD, dr=64, corrupt=0, la=0.1, lr=5e-4, seed=42
    ):
        R.append(
            dict(
                name=name,
                gate=gate,
                ut=ut,
                ui=ui,
                v=v,
                dr=dr,
                corrupt=corrupt,
                la=la,
                lr=lr,
                seed=seed,
            )
        )

    for s in SEEDS:
        for gm in ["gate_off", "gate_blind", "gate_prior_frozen", "gate_prior_init"]:
            add(f"{gm}_s{s}", gate=gm, seed=s)  # four gate modes
        add(f"no_image_s{s}", ui=0, seed=s)  # subtrahend of the image delta
        add(f"structure_only_s{s}", ut=0, ui=0, seed=s)  # sanity
    for s in SEEDS:
        add(f"fullview_gate_s{s}", v=VF, seed=s)  # full-view gate
    add("tune_la05_gate_s42", la=0.5)  # stronger text alignment
    add("tune_la05_noimg_s42", ui=0, la=0.5)
    add("tune_lr1e3_gate_s42", lr=1e-3)  # small lr sweep
    add("tune_lr1e3_noimg_s42", ui=0, lr=1e-3)
    add("cap256_gate_s42", dr=256)  # capacity d_r=256
    add("cap256_noimg_s42", ui=0, dr=256)
    for s in SEEDS:
        add(f"corrupt_gate_s{s}", corrupt=1, seed=s)  # corrupt-provenance (last)
    return R


def run_one(cfg, epochs, ve, idx, total):
    rp = f"result/ranks_{cfg['name']}.tsv"
    gp = f"result/g_{cfg['name']}.tsv"
    if os.path.exists(rp):
        print(f"[{idx}/{total}] SKIP {cfg['name']}", flush=True)
        return "skip"
    print(
        f"[{idx}/{total}] START {cfg['name']} (gate={cfg['gate']} ut={cfg['ut']} ui={cfg['ui']} "
        f"dr={cfg['dr']} corrupt={cfg['corrupt']} la={cfg['la']} seed={cfg['seed']})",
        flush=True,
    )
    cmd = [
        sys.executable,
        "-u",
        "train.py",
        "--data_dir",
        "data/ViMMKG",
        "--t_path",
        TP,
        "--v_path",
        cfg["v"],
        "--gate_mode",
        cfg["gate"],
        "--seed",
        str(cfg["seed"]),
        "--use_text",
        str(cfg["ut"]),
        "--use_image",
        str(cfg["ui"]),
        "--dim_rel",
        str(cfg["dr"]),
        "--corrupt_tier",
        str(cfg["corrupt"]),
        "--lambda_align",
        str(cfg["la"]),
        "--lr",
        str(cfg["lr"]),
        "--exp",
        cfg["name"],
        "--num_epoch",
        str(epochs),
        "--valid_epoch",
        str(ve),
        "--dump_ranks",
        rp,
    ]
    if cfg["ui"] and cfg["gate"] in ("gate_blind", "gate_prior_frozen", "gate_prior_init"):
        cmd += ["--dump_gate", gp]
    t0 = time.time()
    r = subprocess.run(cmd)
    if r.returncode != 0:
        if r.returncode < 0:
            raise KeyboardInterrupt
        print(f"[{idx}/{total}] ERROR {cfg['name']} (exit {r.returncode})", flush=True)
        return "fail"
    print(f"[{idx}/{total}] DONE {cfg['name']} ({time.time()-t0:.0f}s)", flush=True)
    return "ok"


def main():
    epochs = int(os.environ.get("EPOCHS", "300"))
    ve = int(os.environ.get("VALID_EPOCH", "50"))
    only = os.environ.get("ONLY", "")
    dry = os.environ.get("DRY", "") == "1"
    os.makedirs("result", exist_ok=True)
    runs = build_runs()
    if only:
        runs = [r for r in runs if only in r["name"]]
    print(
        f"[grid] {len(runs)} runs | epochs={epochs}{' | ONLY='+only if only else ''}"
        f"{' | DRY' if dry else ''}",
        flush=True,
    )
    for i, c in enumerate(runs, 1):
        print(f"  {i:2d}. {c['name']}", flush=True)
    if dry:
        return
    fails = []
    try:
        for i, c in enumerate(runs, 1):
            if run_one(c, epochs, ve, i, len(runs)) == "fail":
                fails.append(c["name"])
    except KeyboardInterrupt:
        print("\n[grid] INTERRUPTED - rerun to resume (idempotent).", flush=True)
        return
    print(f"\n[grid] DONE. Failures: {fails or 'none'}", flush=True)


if __name__ == "__main__":
    main()
