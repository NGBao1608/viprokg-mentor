#!/usr/bin/env python3
"""Constants and default configuration for ViProNet.

d=256 (model width), N=15707 entities, R=73 relations.
"""

from __future__ import annotations

import os

# --- dimensions ---
DIM_STR = 256  # structural embedding / model width
DIM_TXT = 768  # PhoBERT-base hidden size
# Default visual feature width is 512 (CLIP ViT-B/32). The BEiT-v2 variant uses the mean codebook
# embedding of the visual tokens (dim 32). Override with the VIS_DIM environment variable (read at
# import time; every training subprocess re-reads it) so no code has to change.
DIM_VIS = int(os.environ.get("VIS_DIM", "512"))
DIM_REL = 64  # low-rank TuckER relation dimension: core [256x64x256] ~ 4.2M parameters

# --- per-tier image prior ---
TIER_PRIOR = {1: 0.97, 2: 0.66, 3: 0.73}  # tier -> human-QA accuracy (fallback when no prior file)


def load_tier_prior(path: str) -> dict:
    """Read the per-tier prior table (`prior_tang.tsv`) instead of the three hard-coded values.

    Keys are (tier, scored_label). The file uses the labels `0` / `1_nguoi` / `2_nguoi_phan_xu`
    whereas `image_index.tsv` only has `0` / `1`; within one tier exactly one non-`0` label
    exists, so `scored != "0"` maps to that key. The function asserts this uniqueness.
    Rows with an empty `prior` are skipped, not set to 0.
    """
    import csv as _csv

    out, unique_label = {}, {}
    with open(path, encoding="utf-8") as f:
        for r in _csv.DictReader(f, delimiter="\t"):
            tang, scored, prior = r["tang"].strip(), r["da_cham"].strip(), r["prior"].strip()
            if not prior:
                continue
            out[(tang, scored)] = float(prior)
            if scored != "0":
                assert (
                    tang not in unique_label
                ), f"tier {tang} has more than one scored label: {unique_label[tang]}, {scored}"
                unique_label[tang] = scored
    out["_unique_label"] = unique_label
    return out


# one-hot order of image_kind
IMAGE_KINDS = ["photo", "map", "flag", "logo", "emblem"]

# --- the four gate modes ---
GATE_MODES = ["gate_off", "gate_blind", "gate_prior_frozen", "gate_prior_init"]
#  gate_off          : g=1 for every entity with an image (g=0 if no image)
#  gate_blind        : g = sigmoid(Linear(V[i])), learned from the image feature only, no metadata
#  gate_prior_frozen : g = prior_tier (frozen)
#  gate_prior_init   : g = sigmoid(logit(prior_tier) + delta(metadata)), delta init 0 so g(0)=prior;
#                      keeps learning, anchored by L_gate

# --- loss weights ---
LAMBDA_ALIGN = 0.1  # lambda_1 in {0.1, 0.5}
LAMBDA_GATE = 0.01  # lambda_2 in {0.01, 0.1}
ALIGN_TEMP = 0.5  # InfoNCE temperature of L_align

# --- training defaults ---
SEED = 42
LR = 5e-4
NUM_EPOCH = 300
BATCH_SIZE = 512
VALID_EPOCH = 50

# --- data layout (read-only) ---
DATA_DIR = "data/final/ViMMKG"  # entities/relations/{train,valid,test}.txt, *.tsv, image_index.tsv
