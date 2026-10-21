"""Approximate re-implementation of DSoM (arXiv 2507.20738, MM'25).

Minimal reproduction following the method description. Reported label: "DSoM re-impl. (approx.)".
It does not read our model code; it is built independently from the method description. Every
hyper-parameter the paper does not state is our own choice, recorded here and in config.json.

Components:
  (1) ComplEx + reciprocal relations + softmax CE over |E| (no negative sampling).
  (2) Three ComplEx teachers S/V/D: S = learned random embeddings; V = fixed visual features @ W_v;
      D = fixed text features @ W_d. Trained simultaneously (Eq.3 = sum of the three CE losses),
      each teacher with its own random relation embeddings.
  (3) Structure-only ComplEx student; test uses only E_stu / R_stu.
  (4) Policy MLP: [p_S | p_V | p_D] (3|E|) -> 1024 ReLU -> 7 (non-empty subsets of {S,V,D});
      ensemble = mean of the selected teachers (Eq.7); reward +1/-10 (Eq.8, +1 if top-1 == gold);
      REINFORCE baseline = reward of the mean of all three (Eq.9-10).
  (5) NDKD (Eq.11-12): tau=4, alpha=beta=1, gamma=2; N(h,r) = all training tails of (h,r)
      (including inverse); b=[p_N, 1-p_N], p_N = mean_{n in N} p_n; P~ = p_e/(1-p_N) on E minus N.
  (6) Eq.4 = CE + L_RC + gamma*NDKD. L_RC = Eq.10 = the policy loss (REINFORCE): theta_policy is
      updated at EVERY step through opt_pi; `(loss_stu + loss_pi).backward()` connects the graph
      delta*log(pi) -> theta_policy. `pi_gmax` (max |grad theta_policy|) is measured and asserted
      > 0 to confirm theta receives gradient.

Teachers are FROZEN while the student trains (Alg.1). The best-valid checkpoint is selected and
per-key filtered ranks are dumped for both directions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# our own hyper-parameters, fixed in advance and not changed after seeing results
HP = dict(
    d_complex=128,
    lr=1e-3,
    batch=1024,
    epoch_teacher=300,
    epoch_student=300,
    patience=20,
    tau=4.0,
    alpha=1.0,
    beta=1.0,
    gamma=2.0,
    policy_hidden=1024,
)
ACTIONS = [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]  # non-empty subsets of {S,V,D}


def sha16(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:16]


# ---------------- data (MoMoK/OpenKE numeric-id format) ----------------
def read_map(path):
    d = {}
    with open(path) as f:
        for line in f:
            c = line.rstrip("\n").replace("\t", " ").split(" ")
            if len(c) >= 2 and c[1].lstrip("-").isdigit():
                d[c[0]] = int(c[1])
    return d


def read_triples(path, e2i, r2i):
    """train/valid/test lines 'h r t [.]' by NAME (URL for W/Y; numeric-id string for v2), mapped through e2i/r2i."""
    out = []
    with open(path) as f:
        for line in f:
            c = line.replace("\t", " ").split()
            if len(c) >= 3 and c[0] in e2i and c[1] in r2i and c[2] in e2i:
                out.append((e2i[c[0]], r2i[c[1]], e2i[c[2]]))
    return out


def load_data(data_dir):
    e2i = read_map(os.path.join(data_dir, "entity2id.txt"))
    r2i = read_map(os.path.join(data_dir, "relation2id.txt"))
    n_e, n_r = len(e2i), len(r2i)
    tr = read_triples(os.path.join(data_dir, "train.txt"), e2i, r2i)
    va = read_triples(os.path.join(data_dir, "valid.txt"), e2i, r2i)
    te = read_triples(os.path.join(data_dir, "test.txt"), e2i, r2i)
    img = torch.load(os.path.join(data_dir, "img_features.pth"), map_location="cpu").float()
    txt = torch.load(os.path.join(data_dir, "text_features.pth"), map_location="cpu").float()
    return e2i, r2i, n_e, n_r, tr, va, te, img, txt


# ---------------- ComplEx scorer (reciprocal, CE over |E|) ----------------
class ComplEx(nn.Module):
    """Entity emb (re, im); relation emb (re, im) with 2R rows (reciprocal). score(h, r) over every t."""

    def __init__(self, n_ent, n_rel2, d, ent_emb=None, learn_ent=True):
        super().__init__()
        self.d = d
        if ent_emb is None:
            self.E = nn.Parameter(torch.empty(n_ent, 2 * d))
            nn.init.xavier_uniform_(self.E)
        else:
            self.register_parameter("E", None)
            self._E = ent_emb  # callable returning [N, 2d]
        self.learn_ent = learn_ent
        self.R = nn.Parameter(torch.empty(n_rel2, 2 * d))
        nn.init.xavier_uniform_(self.R)

    def ent(self):
        return self.E if self.E is not None else self._E()

    def scores(self, h, r):
        """h, r: [B] -> scores [B, N] (ComplEx: Re(<e_h, w_r, conj(e_t)>))."""
        E = self.ent()
        d = self.d
        er, ei = E[:, :d], E[:, d:]
        hr, hi = er[h], ei[h]
        wr, wi = self.R[r][:, :d], self.R[r][:, d:]
        # a = h * w (complex): (hr+i*hi)(wr+i*wi)
        ar = hr * wr - hi * wi
        ai = hr * wi + hi * wr
        # <a, conj(e_t)> real = ar·er + ai·ei
        return ar @ er.t() + ai @ ei.t()


class Proj(nn.Module):
    """E_teacher = fixed_features @ W (W is learned). Returns [N, 2d]."""

    def __init__(self, feat, d2):
        super().__init__()
        self.register_buffer("feat", feat)
        self.W = nn.Linear(feat.shape[1], d2, bias=False)

    def __call__(self):
        return self.W(self.feat)


# ---------------- filter dict (both directions, train U valid U test) ----------------
def build_filter(n_r, *splits):
    ft, fh = defaultdict(set), defaultdict(set)
    for sp in splits:
        for h, r, t in sp:
            ft[(h, r)].add(t)
            fh[(r, t)].add(h)
    return ft, fh


def recip(triples, n_r):
    out = []
    for h, r, t in triples:
        out.append((h, r, t))
        out.append((t, r + n_r, h))  # reciprocal
    return out


# ---------------- per-key filtered ranks, both directions ----------------
def score_split(model, triples, n_r, ft, fh, dev):
    """Return rows [(h, r, t, head_rank, tail_rank)], filtered."""
    model.eval()
    rows = []
    with torch.no_grad():
        for h, r, t in triples:
            # tail: score(h,r) over E
            sc = model.scores(torch.tensor([h], device=dev), torch.tensor([r], device=dev))[0]
            gold = sc[t].item()
            filt = ft[(h, r)] - {t}
            sc_f = sc.clone()
            if filt:
                sc_f[list(filt)] = -1e9
            tr = int((sc_f > gold).sum().item()) + 1
            # head: use the reciprocal (t, r+n_r) over E
            sc2 = model.scores(torch.tensor([t], device=dev), torch.tensor([r + n_r], device=dev))[
                0
            ]
            gold2 = sc2[h].item()
            filt2 = fh[(r, t)] - {h}
            sc2_f = sc2.clone()
            if filt2:
                sc2_f[list(filt2)] = -1e9
            hr = int((sc2_f > gold2).sum().item()) + 1
            rows.append((h, r, t, hr, tr))
    return rows


def mrr_of(rows):
    rk = np.array([x[3] for x in rows] + [x[4] for x in rows], dtype=float)
    return float(np.mean(1.0 / rk))


# ---------------- train one ComplEx with softmax CE over |E| (reciprocal) ----------------
def train_complex(
    model, tri_recip, n_e, dev, lr, epochs, patience, valid, n_r, ft, fh, batch, name="S"
):
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    tri = torch.tensor(tri_recip, device=dev)
    best, best_ep, bad, best_state = -1.0, -1, 0, None
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(tri.shape[0], device=dev)
        for i in range(0, tri.shape[0], batch):
            b = tri[perm[i : i + batch]]
            h, r, t = b[:, 0], b[:, 1], b[:, 2]
            sc = model.scores(h, r)
            loss = F.cross_entropy(sc, t)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % 5 == 0 or ep == epochs:
            vm = mrr_of(score_split(model, valid, n_r, ft, fh, dev))
            if vm > best:
                best, best_ep, bad = vm, ep, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            print(f"  [{name}] e{ep} valid_MRR={vm:.5f} (best {best:.5f}@e{best_ep})", flush=True)
            if bad >= patience:
                print(f"  [{name}] early-stop @e{ep}", flush=True)
                break
    if best_state:
        model.load_state_dict(best_state)
    return best, best_ep


# ---------------- policy MLP + NDKD + student (plan §1.4-§1.6) ----------------
class Policy(nn.Module):
    def __init__(self, n_e, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * n_e, hidden), nn.ReLU(), nn.Linear(hidden, len(ACTIONS))
        )

    def forward(self, feat3):  # feat3: [B, 3|E|] → logits [B,7]
        return self.net(feat3)


def ndkd_loss(stu_sc, tea_p, Nmask, tau, alpha, beta):
    """NDKD (our approximation of Eq.11-12): separates the group N(h,r) from E minus N.
    stu_sc [B,E] student logits; tea_p [B,E] teacher-ensemble probabilities; Nmask [B,E] bool (tail in N).
    b=[p_N, 1-p_N] (p_N = mean teacher prob on N); P~ = p/(1-p_N) on E minus N. KL at temperature tau.
    """
    eps = 1e-9
    stu_p = F.softmax(stu_sc / tau, dim=1)
    # p_N: mean probability over the elements of N (by count), pooled into one block
    cntN = Nmask.sum(1).clamp(min=1)
    pN_t = (tea_p * Nmask).sum(1) / cntN  # mean teacher prob on N
    pN_s = (stu_p * Nmask).sum(1) / cntN
    # (a) binary block b=[pN, 1-pN]
    bt = torch.stack([pN_t, 1 - pN_t], 1).clamp(eps, 1)
    bs = torch.stack([pN_s, 1 - pN_s], 1).clamp(eps, 1)
    l_bin = (bt * (bt.log() - bs.log())).sum(1).mean()
    # (b) P~ on E minus N: p/(1-pN) (renormalised non-N block)
    nonN = (~Nmask).float()
    massT = (tea_p * nonN).sum(1, keepdim=True).clamp(min=eps)
    massS = (stu_p * nonN).sum(1, keepdim=True).clamp(min=eps)
    Pt = (tea_p * nonN) / massT
    Ps = (stu_p * nonN) / massS
    l_non = (Pt * ((Pt + eps).log() - (Ps + eps).log())).sum(1).mean()
    return (alpha * l_bin + beta * l_non) * (tau * tau)


def train_student(stu, teachers, policy, tri_recip, n_e, n_r, dev, hp, valid, ft, fh, tail_of):
    opt = torch.optim.Adam(stu.parameters(), lr=hp["lr"])
    opt_pi = torch.optim.Adam(policy.parameters(), lr=hp["lr"])
    for m in teachers:
        m.eval()
    tri = torch.tensor(tri_recip, device=dev)
    best, best_ep, bad, best_state = -1.0, -1, 0, None
    act_count = torch.zeros(len(ACTIONS))
    pi_gmax = 0.0  # max |grad| of theta_policy (L_RC=Eq.10): checks that theta receives gradient from delta*log(pi)
    for ep in range(1, hp["epoch_student"] + 1):
        stu.train()
        policy.train()
        perm = torch.randperm(tri.shape[0], device=dev)
        for i in range(0, tri.shape[0], hp["batch"]):
            b = tri[perm[i : i + hp["batch"]]]
            h, r, t = b[:, 0], b[:, 1], b[:, 2]
            with torch.no_grad():
                tp = [F.softmax(m.scores(h, r), dim=1) for m in teachers]  # p_S,p_V,p_D [B,E]
            feat3 = torch.cat(tp, dim=1)  # [B,3E]
            logits = policy(feat3)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()  # [B] actions
            # ensemble by action (Eq.7) + mean-of-3 baseline
            ens = torch.zeros_like(tp[0])
            for bi in range(b.shape[0]):
                sel = ACTIONS[a[bi].item()]
                ens[bi] = torch.stack([tp[j][bi] for j in sel]).mean(0)
            base = torch.stack(tp).mean(0)  # mean of all 3
            rew = torch.where(ens.argmax(1) == t, 1.0, -10.0)  # Eq.8
            rew_b = torch.where(base.argmax(1) == t, 1.0, -10.0)
            adv = (rew - rew_b).detach()
            loss_pi = -(adv * dist.log_prob(a)).mean()  # REINFORCE (Eq.9-10)
            for aa in a.detach().cpu().tolist():
                act_count[aa] += 1
            # student: CE + gamma*NDKD (target = frozen ensemble chosen by the policy)
            sc = stu.scores(h, r)
            ce = F.cross_entropy(sc, t)
            Nmask = torch.zeros(b.shape[0], n_e, dtype=torch.bool, device=dev)
            for bi in range(b.shape[0]):
                nb = tail_of.get((int(h[bi]), int(r[bi])))
                if nb:
                    Nmask[bi, list(nb)] = True
            kd = ndkd_loss(sc, ens.detach(), Nmask, hp["tau"], hp["alpha"], hp["beta"])
            loss_stu = ce + hp["gamma"] * kd
            opt.zero_grad()
            opt_pi.zero_grad()
            (loss_stu + loss_pi).backward()
            # L_RC = Eq.10 (policy): theta_policy MUST receive gradient from delta*log(pi); measure |grad|
            g = sum(float(p.grad.abs().sum()) for p in policy.parameters() if p.grad is not None)
            pi_gmax = max(pi_gmax, g)
            opt.step()
            opt_pi.step()
        if ep % 5 == 0 or ep == hp["epoch_student"]:
            vm = mrr_of(score_split(stu, valid, n_r, ft, fh, dev))
            if vm > best:
                best, best_ep, bad = vm, ep, 0
                best_state = {k: v.detach().clone() for k, v in stu.state_dict().items()}
            else:
                bad += 1
            print(f"  [stu] e{ep} valid_MRR={vm:.5f} (best {best:.5f}@e{best_ep})", flush=True)
            if bad >= hp["patience"]:
                print(f"  [stu] early-stop @e{ep}", flush=True)
                break
    if best_state:
        stu.load_state_dict(best_state)
    print(
        f"  [policy] max |grad theta_policy| (L_RC=Eq.10) = {pi_gmax:.4e} "
        f'({"nonzero: theta receives gradient" if pi_gmax > 0 else "ZERO: theta receives NO gradient"})',
        flush=True,
    )
    assert (
        pi_gmax > 0
    ), "theta_policy receives NO gradient from delta*log(pi) (L_RC Eq.10 is not connected to the graph)"
    return best, best_ep, (act_count / act_count.sum().clamp(min=1)).tolist(), pi_gmax


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--cuda", type=int, default=0)
    p.add_argument("--ent_names", default=None)
    p.add_argument("--rel_names", default=None)
    p.add_argument(
        "--teacher_only", action="store_true", help="G1: teacher S only, dump, no student/policy"
    )
    for k, v in HP.items():
        p.add_argument(f"--{k}", type=type(v), default=v)
    a = p.parse_args()
    hp = {k: getattr(a, k) for k in HP}
    lenh = "python " + " ".join(sys.argv)
    os.makedirs(a.out_dir, exist_ok=True)
    dev = f"cuda:{a.cuda}" if (a.cuda >= 0 and torch.cuda.is_available()) else "cpu"
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    e2i, r2i, n_e, n_r, tr, va, te, img, txt = load_data(a.data_dir)
    d = hp["d_complex"]
    train_recip = recip(tr, n_r)
    ft, fh = build_filter(n_r, tr, va, te)
    tail_of = defaultdict(set)
    for h, r, t in train_recip:
        tail_of[(h, r)].add(t)
    img = img.to(dev)
    txt = txt.to(dev)
    t0 = time.time()

    # teacher S (G1 only needs S)
    S = ComplEx(n_e, 2 * n_r, d).to(dev)
    vS, epS = train_complex(
        S,
        train_recip,
        n_e,
        dev,
        hp["lr"],
        hp["epoch_teacher"],
        hp["patience"],
        va,
        n_r,
        ft,
        fh,
        hp["batch"],
        "S",
    )
    acts = None
    pi_gmax = None
    if a.teacher_only:
        model = S
        best_ep = epS
    else:
        V = ComplEx(n_e, 2 * n_r, d, ent_emb=Proj(img, 2 * d).to(dev)).to(dev)
        D = ComplEx(n_e, 2 * n_r, d, ent_emb=Proj(txt, 2 * d).to(dev)).to(dev)
        train_complex(
            V,
            train_recip,
            n_e,
            dev,
            hp["lr"],
            hp["epoch_teacher"],
            hp["patience"],
            va,
            n_r,
            ft,
            fh,
            hp["batch"],
            "V",
        )
        train_complex(
            D,
            train_recip,
            n_e,
            dev,
            hp["lr"],
            hp["epoch_teacher"],
            hp["patience"],
            va,
            n_r,
            ft,
            fh,
            hp["batch"],
            "D",
        )
        stu = ComplEx(n_e, 2 * n_r, d).to(dev)
        policy = Policy(n_e, hp["policy_hidden"]).to(dev)
        _, best_ep, acts, pi_gmax = train_student(
            stu, [S, V, D], policy, train_recip, n_e, n_r, dev, hp, va, ft, fh, tail_of
        )
        model = stu
    train_s = time.time() - t0

    rows = score_split(model, te, n_r, ft, fh, dev)
    test_mrr = mrr_of(rows)
    id2e = {v: k for k, v in e2i.items()}
    id2r = {v: k for k, v in r2i.items()}
    if a.ent_names:
        id2e = {
            int(l.rstrip("\n").split("\t")[1]): l.split("\t")[0]
            for l in open(a.ent_names).read().splitlines()[1:]
            if l
        }
    if a.rel_names:
        id2r = {
            int(l.rstrip("\n").split("\t")[1]): l.split("\t")[0]
            for l in open(a.rel_names).read().splitlines()[1:]
            if l
        }
    rp = os.path.join(a.out_dir, "ranks_test.tsv")
    with open(rp, "w") as f:
        f.write("s_qid\tp_pid\to_qid\thead_rank\ttail_rank\n")
        for h, r, t, hr, tr_ in rows:
            assert hr >= 1 and tr_ >= 1
            f.write(f"{id2e[h]}\t{id2r[r]}\t{id2e[t]}\t{hr}\t{tr_}\n")
    print(f"[dump] {rp} ({len(rows)} triple, test MRR={test_mrr:.5f})", flush=True)

    json.dump(
        {
            "lenh": lenh,
            "seed": a.seed,
            "model": "DSoM re-impl. (approx.)",
            "nhan": "DSoM re-impl. (approx.)",
            "hp_cua_ta": hp,
            "teacher_only": a.teacher_only,
            "best_epoch": best_ep,
            "valid_MRR_teacherS": vS,
            "test_MRR_submitted": test_mrr,
            "policy_action_dist": acts,
            "actions": [str(x) for x in ACTIONS],
            "policy_grad_max": pi_gmax,  # L_RC=Eq.10: >0 means theta_policy receives gradient from delta*log(pi)
            "hash_dau_vao": {
                k: sha16(os.path.join(a.data_dir, k))
                for k in (
                    "entity2id.txt",
                    "train.txt",
                    "valid.txt",
                    "test.txt",
                    "img_features.pth",
                    "text_features.pth",
                )
                if os.path.exists(os.path.join(a.data_dir, k))
            },
            "train_giay": round(train_s, 1),
        },
        open(os.path.join(a.out_dir, "config.json"), "w"),
        ensure_ascii=False,
        indent=2,
    )
    json.dump(
        {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "train_giay": round(train_s, 1),
        },
        open(os.path.join(a.out_dir, "moi_truong.json"), "w"),
        ensure_ascii=False,
        indent=2,
    )
    print("[done] config + moi_truong", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
