#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ViProNet: multimodal KG completion with a provenance gate.

Entity-level vectors (no tokenization). The gate g reads provenance metadata (tier/kind/n_share)
and is anchored to the human-QA prior. Four gate modes via --gate_mode. Low-rank TuckER scorer.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from config import DIM_REL, DIM_STR, DIM_TXT, DIM_VIS, IMAGE_KINDS


class ProvenanceGate(nn.Module):
    """Per-entity image reliability gate g in [0,1], four modes.

    Metadata features (buffers, not learned): prior_tier, kind_onehot, log_nshare, has_image.
    - gate_off          : g = has_image (1/0), no parameters.
    - gate_blind        : g = sigmoid(Linear(V)) * has_image, learned from the image feature only.
    - gate_prior_frozen : g = prior_tier (frozen).
    - gate_prior_init   : g = sigmoid(logit(prior) + delta(meta)) * has_image, delta init 0 so
                          g(0)=prior; keeps learning.
    """

    def __init__(self, mode: str, meta: Dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.mode = mode
        self.register_buffer("has_image", meta["has_image"].float())
        self.register_buffer("prior_tier", meta["prior_tier"].float())
        # logit(prior) for gate_prior_init; entities without image -> 0 (masked by has_image)
        p = meta["prior_tier"].float().clamp(1e-4, 1 - 1e-4)
        logit = torch.log(p / (1 - p))
        logit[meta["has_image"] == 0] = 0.0
        self.register_buffer("prior_logit", logit)
        # metadata features for delta: [prior_tier(1), kind_onehot(5), log_nshare(1)] = 7 dims
        feat = torch.cat(
            [
                meta["prior_tier"].float().unsqueeze(1),
                meta["kind_onehot"].float(),
                meta["log_nshare"].float().unsqueeze(1),
            ],
            dim=1,
        )
        self.register_buffer("meta_feat", feat)  # [N, 7]

        if mode == "gate_blind":
            self.blind = nn.Linear(DIM_VIS, 1)  # learned from V[i]
        elif mode == "gate_prior_init":
            self.delta = nn.Linear(1 + len(IMAGE_KINDS) + 1, 1)  # delta(meta), init 0
            nn.init.zeros_(self.delta.weight)
            nn.init.zeros_(self.delta.bias)
        # gate_off / gate_prior_frozen: no parameters

    def forward(self, V: torch.Tensor) -> torch.Tensor:
        """Return g [N] in [0,1]; entities without an image always get g=0."""
        if self.mode == "gate_off":
            return self.has_image
        if self.mode == "gate_prior_frozen":
            return self.prior_tier
        if self.mode == "gate_blind":
            g = torch.sigmoid(self.blind(V)).squeeze(1)
            return g * self.has_image
        if self.mode == "gate_prior_init":
            g = torch.sigmoid(self.prior_logit + self.delta(self.meta_feat).squeeze(1))
            return g * self.has_image
        raise ValueError(f"unknown gate_mode: {self.mode}")


class AttnPool(nn.Module):
    """Pool text tokens with one learned query.

    H [N,L,Din] + mask [N,L] (True = real token) -> vector [N,Din]. Entities without tokens -> 0.
    """

    def __init__(self, dim_in: int) -> None:
        super().__init__()
        self.q = nn.Parameter(torch.randn(dim_in) * 0.02)

    def forward(self, H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = (H * self.q).sum(-1)  # [N,L]
        scores = scores.masked_fill(~mask, float("-inf"))
        w = torch.softmax(scores, dim=1)
        w = torch.nan_to_num(w, 0.0)  # entity without tokens: all -inf -> 0
        return (w.unsqueeze(-1) * H).sum(1)  # [N,Din]


class NameEncoder(nn.Module):
    """Syllable-level encoder of the Vietnamese entity name (tone marks kept), a plug-in channel.

    The name is the plain entity label (not taken from the summary/bio), split on whitespace into
    syllables (PAD=0, UNK=1). A small one-layer GRU captures syllable order. Entities without a
    name -> zero vector. About 0.35M parameters.
    """

    def __init__(self, vocab: int, emb: int = 64, hid: int = 64, dim_name: int = 128) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.gru = nn.GRU(emb, hid, batch_first=True)  # small one-layer GRU (syllable order)
        self.proj = nn.Linear(hid, dim_name)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """ids/mask [N,L] -> [N,dim_name]. Entities without a name (mask all False) -> 0."""
        e = self.emb(ids)  # [N,L,emb]
        lengths = mask.sum(1).clamp(min=1).cpu()  # clamp: avoid length 0 (entities without a name)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lengths, batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)  # h [1,N,hid]
        out = self.proj(h.squeeze(0))  # [N,dim_name]
        return out * (mask.sum(1, keepdim=True) > 0)  # entity without a name -> 0


class TuckER(nn.Module):
    """Low-rank TuckER scoring function: core [d x d_r x d], relation dimension d_r.

    Supports 1-vs-all scoring in both directions (masked head / masked tail) without inverse
    relations.
    """

    def __init__(
        self,
        num_rel: int,
        dim: int = DIM_STR,
        dim_rel: int = DIM_REL,
        in_drop: float = 0.2,
        hid_drop: float = 0.3,
        num_ent: int = 0,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        self.R = nn.Embedding(num_rel, dim_rel)
        self.W = nn.Parameter(torch.empty(dim, dim_rel, dim))  # [i, d, j]
        nn.init.xavier_normal_(self.W)
        self.bn0 = nn.BatchNorm1d(dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.in_drop = nn.Dropout(in_drop)
        self.hid_drop = nn.Dropout(hid_drop)
        nn.init.xavier_normal_(self.R.weight)
        # optional entity bias b_e (added to every candidate score, ConvE-style)
        self.b_e = nn.Parameter(torch.zeros(num_ent)) if (use_bias and num_ent) else None

    def _mat(self, r_idx: torch.Tensor) -> torch.Tensor:
        r = self.R(r_idx)  # [B, d_r]
        return torch.einsum("idj,bd->bij", self.W, r)  # [B, d, d]

    def score_tail(
        self, h: torch.Tensor, r_idx: torch.Tensor, ent_all: torch.Tensor
    ) -> torch.Tensor:
        """score(h, r, .) over all entities. h [B,d]; ent_all [N,d]; returns [B,N]."""
        x = self.bn0(h)
        x = self.in_drop(x)
        M = self._mat(r_idx)  # [B,d,d]
        x = torch.einsum("bi,bij->bj", x, M)  # [B,d]
        x = self.bn1(x)
        x = self.hid_drop(x)
        s = x @ ent_all.t()  # [B,N]
        return s + self.b_e if self.b_e is not None else s  # add entity bias

    def score_head(
        self, r_idx: torch.Tensor, t: torch.Tensor, ent_all: torch.Tensor
    ) -> torch.Tensor:
        """score(., r, t) over all entities. t [B,d]; returns [B,N]."""
        M = self._mat(r_idx)  # [B,d,d]
        x = torch.einsum("bij,bj->bi", M, self.bn0(t))  # [B,d]  (contract the tail)
        x = self.bn1(x)
        x = self.hid_drop(x)
        s = x @ ent_all.t()  # [B,N]
        return s + self.b_e if self.b_e is not None else s


class ViProNet(nn.Module):
    """Full model. encode() returns h_e [N,d] plus auxiliary tensors for the losses.

    text_mode:
      - mean    : t_e = W_t(mean-pool T); fusion = concat [e_s; t_e; v_hat] (default).
      - attn    : t_e = W_t(AttnPool(H)); fusion = concat (only the pooling changes, ablation).
      - tokfuse : sequence [CLS, struct, gated image, k text tokens] through a small transformer
                  -> [CLS] (MyGO-style). t_e (AttnPool) is still computed for L_align.
    attn/tokfuse need H [N,k,DIM_TXT] + Hmask [N,k]; mean only needs T [N,DIM_TXT].
    """

    def __init__(
        self,
        num_ent: int,
        num_rel: int,
        meta: Dict[str, torch.Tensor],
        gate_mode: str,
        dim: int = DIM_STR,
        dim_rel: int = DIM_REL,
        text_mode: str = "mean",
        fuse_layers: int = 1,
        fuse_heads: int = 4,
        entity_bias: bool = False,
        use_name: bool = False,
        name_vocab: int = 0,
        dim_name: int = 128,
        dual_text: bool = False,
    ) -> None:
        super().__init__()
        self.num_ent = num_ent
        self.dim = dim
        self.gate_mode = gate_mode
        self.text_mode = text_mode
        self.use_name = use_name
        self.dual_text = dual_text
        # structural embedding learned from scratch
        self.E_s = nn.Parameter(torch.empty(num_ent, dim))
        nn.init.xavier_normal_(self.E_s)
        # project text/visual to d
        self.W_t = nn.Linear(DIM_TXT, dim)
        self.W_v = nn.Linear(DIM_VIS, dim)
        # name channel (4th fusion input); use_name=False keeps the original 3*dim
        if use_name:
            self.name_enc = NameEncoder(name_vocab, dim_name=dim_name)
            self.W_f = nn.Linear(3 * dim + dim_name, dim)  # extended fusion: [e_s;t_e;v_e;name_e]
        elif dual_text:
            # two independent text channels: [e_s; W_t*T_base; W_t2*T_ft; v_e]
            self.W_t2 = nn.Linear(DIM_TXT, dim)
            self.W_f = nn.Linear(4 * dim, dim)
        else:
            self.W_f = nn.Linear(3 * dim, dim)  # base fusion (concat of 3 sources)
        self.fuse_ln = nn.LayerNorm(dim)
        # token pooling / fusion
        if text_mode in ("attn", "tokfuse"):
            self.attn_pool = AttnPool(DIM_TXT)
        if text_mode == "tokfuse":
            self.W_ttok = nn.Linear(DIM_TXT, dim)  # project each text token -> d
            self.cls = nn.Parameter(torch.randn(dim) * 0.02)
            self.type_emb = nn.Embedding(4, dim)  # 0=CLS 1=struct 2=image 3=text
            layer = nn.TransformerEncoderLayer(
                dim, fuse_heads, dim_feedforward=2 * dim, dropout=0.1, batch_first=True
            )
            self.fuse_tf = nn.TransformerEncoder(layer, num_layers=fuse_layers)
        # gate
        self.gate = ProvenanceGate(gate_mode, meta)
        # scorer (dim_rel default 64; capacity ablation {64,256}); optional entity bias
        self.tucker = TuckER(num_rel, dim, dim_rel, num_ent=num_ent, use_bias=entity_bias)

    def _token_fuse(
        self,
        e_s: torch.Tensor,
        v_e: torch.Tensor,
        H: torch.Tensor,
        Hmask: torch.Tensor,
        use_image: bool,
    ) -> torch.Tensor:
        """Transformer over [CLS, struct, image, k text tokens] -> [CLS] vector."""
        N, k = e_s.shape[0], H.shape[1]
        ttok = self.W_ttok(H)  # [N,k,d]
        seq = torch.cat(
            [self.cls.expand(N, 1, -1), e_s.unsqueeze(1), v_e.unsqueeze(1), ttok], dim=1
        )  # [N, 3+k, d]
        tid = torch.cat(
            [torch.tensor([0, 1, 2], device=e_s.device), torch.full((k,), 3, device=e_s.device)]
        )
        seq = seq + self.type_emb(tid).unsqueeze(0)
        pad = torch.zeros(N, 3 + k, dtype=torch.bool, device=e_s.device)  # False = keep
        pad[:, 3:] = ~Hmask  # text tokens follow the mask
        if not use_image:
            pad[:, 2] = True  # no_image: mask the image token
        out = self.fuse_tf(seq, src_key_padding_mask=pad)  # [N, 3+k, d]
        return self.fuse_ln(out[:, 0])  # [CLS] vector

    def encode(
        self,
        T: torch.Tensor,
        V: torch.Tensor,
        H: torch.Tensor = None,
        Hmask: torch.Tensor = None,
        use_text: bool = True,
        use_image: bool = True,
        name_ids: torch.Tensor = None,
        name_mask: torch.Tensor = None,
        T2: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """Fused entity embedding h_e [N,d] plus e_s, t_e, g for the losses.

        use_text/use_image switch modalities off: no_image (use_image=False) / structure_only
        (both False). name_ids/name_mask [N,L] feed the name channel (only used with use_name;
        mean/attn concat path, tokfuse is unaffected).
        """
        e_s = self.E_s
        if use_image:
            g = self.gate(V)  # [N]
            v_e = g.unsqueeze(1) * self.W_v(V)  # [N,d]  gate multiplies before fusion
        else:
            g = torch.zeros(self.num_ent, device=e_s.device)
            v_e = torch.zeros_like(e_s)
        # pooled text vector t_e (used for concat + L_align)
        if use_text:
            if self.text_mode == "mean":
                t_pool = T
            else:
                t_pool = self.attn_pool(H, Hmask)  # [N,DIM_TXT]
            t_e = self.W_t(t_pool)  # [N,d]
        else:
            t_e = torch.zeros_like(e_s)
        # name vector (4th branch); entity without a name -> 0
        name_e = (
            self.name_enc(name_ids, name_mask) if (self.use_name and name_ids is not None) else None
        )
        # fuse into h_e
        if self.text_mode == "tokfuse" and use_text:
            h_e = self._token_fuse(e_s, v_e, H, Hmask, use_image)
        elif name_e is not None:
            h_e = self.fuse_ln(self.W_f(torch.cat([e_s, t_e, v_e, name_e], dim=1)))
        elif self.dual_text:
            # second text channel (T_ft); entity without text or use_text=False -> 0
            t2_e = self.W_t2(T2) if (T2 is not None and use_text) else torch.zeros_like(e_s)
            h_e = self.fuse_ln(self.W_f(torch.cat([e_s, t_e, t2_e, v_e], dim=1)))
        else:
            h_e = self.fuse_ln(self.W_f(torch.cat([e_s, t_e, v_e], dim=1)))
        return {"h_e": h_e, "e_s": e_s, "t_e": t_e, "g": g}
