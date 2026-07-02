"""Stage 2 — cross-modal JEPA (minimal, 2-modality).

Two per-modality encoders (vision CLS 768, state 28) -> d-dim embeddings. Train by predicting
each modality's EMA-target embedding FROM THE OTHER modality (cross-modal, predict-don't-equate),
plus per-modal SIGReg to stop the targets collapsing to a constant (which would make prediction
trivial). Fused latent for downstream = the two modality embeddings combined.

This is the minimal decisive test of "does fusing video+state learn cross-modal structure."
Upgrade path: Perceiver over vision PATCH tokens with token-level masking (matters once vision
is many tokens, not one CLS).
"""
import copy

import torch
import torch.nn as nn

from stable_pretraining.methods.lejepa import SlicedEppsPulley


def _mlp(i, o, h=256):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class MMJepa(nn.Module):
    def __init__(self, d=256, vis_dim=768, state_dim=28, n_slices=512, lamb=0.02, ema=0.99):
        super().__init__()
        self.enc_v = _mlp(vis_dim, d)
        self.enc_s = _mlp(state_dim, d)
        self.pred_v2s = _mlp(d, d)   # from vision embedding -> predict state's target
        self.pred_s2v = _mlp(d, d)   # from state embedding  -> predict vision's target
        # EMA target encoders (stop-grad) — supply the prediction targets
        self.tgt_v = copy.deepcopy(self.enc_v)
        self.tgt_s = copy.deepcopy(self.enc_s)
        for p in list(self.tgt_v.parameters()) + list(self.tgt_s.parameters()):
            p.requires_grad = False
        self.sigreg = SlicedEppsPulley(num_slices=n_slices)
        self.lamb, self.ema = lamb, ema

    @torch.no_grad()
    def update_target(self):
        for online, target in [(self.enc_v, self.tgt_v), (self.enc_s, self.tgt_s)]:
            for po, pt in zip(online.parameters(), target.parameters()):
                pt.mul_(self.ema).add_(po.detach(), alpha=1 - self.ema)

    def forward(self, vision, state):
        ev, es = self.enc_v(vision), self.enc_s(state)          # online embeddings [B,d]
        with torch.no_grad():
            tv, ts = self.tgt_v(vision), self.tgt_s(state)      # targets (stop-grad)
        # cross-modal prediction (predict-don't-equate): each modality predicts the OTHER's target
        inv = (self.pred_v2s(ev) - ts).square().mean() + (self.pred_s2v(es) - tv).square().mean()
        sig = self.sigreg(ev) + self.sigreg(es)                 # per-modal anti-collapse
        loss = inv + self.lamb * sig
        return {"loss": loss, "inv": inv.detach(), "sig": sig.detach(),
                "z": torch.cat([ev, es], dim=-1).detach(),      # fused latent (downstream)
                "ev": ev.detach(), "es": es.detach()}

    @torch.no_grad()
    def embed(self, vision, state):
        return torch.cat([self.enc_v(vision), self.enc_s(state)], dim=-1)
