"""Stage-2-at-scale — Perceiver cross-modal JEPA over the chunk packet (all cfgs).

Extends mm_perceiver.py's validated 2-modality recipe to the 3-modality chunk packet
served by dataloader.py: vision patch tokens (196) + motor rows (8) + ee window
slots (13) -> 217 context tokens, one Perceiver bottleneck. Same losses: masked
cross-modal latent prediction (hide one modality, predict its EMA-target embedding
from the rest; predict-don't-equate) + per-modal SIGReg + joint SIGReg.

Single timestep: masking is over MODALITY, not time (temporal = Stage 5). Every ee
slot / motor row token is positioned by a learned slot embedding for now — the slot
embeddings are the placeholder the Stage-5 continuous-time embedding replaces (each
token then carries Fourier features of its real timestamp instead).

Packet validity masks are True=VALID; CrossAttention attn_mask is True=BLOCKED —
inverted in _attn_mask. ee can be entirely absent (all of cfg5, ~1/3 of cfg3
scenes): those samples are excluded from the ee prediction/SIGReg terms rather than
averaged over zero valid elements (NaN trap). No masked fuse pass can ever see an
all-blocked context: vision and motor are always present, and each hide-one pass
keeps at least one of them visible.
"""
import copy

import torch
import torch.nn as nn

from stable_pretraining.methods.lejepa import SlicedEppsPulley

from world_tokenizer.mm_perceiver import PerceiverFuse, _mlp

N_PATCH, N_MOTOR, MOTOR_CH, EE_T, EE_DIM = 196, 8, 3, 13, 15
T_CTX = N_PATCH + N_MOTOR + EE_T  # 217


def masked_mean(x, mask):
    # x: [B, T, d], mask: [B, T] bool (True=valid) -> [B, d]; rows with zero valid
    # entries return zeros (callers exclude them from any loss).
    m = mask.unsqueeze(-1).float()
    return (x * m).sum(1) / m.sum(1).clamp(min=1.0)


class MMPerceiverChunks(nn.Module):
    def __init__(self, d=256, vis_dim=768, n_queries=8, lamb=0.02, ema=0.99,
                 n_slices=512):
        super().__init__()
        self.proj_v = nn.Linear(vis_dim, d)             # vision patch: 768 -> d
        self.proj_m = nn.Linear(2 * MOTOR_CH, d)        # motor row: 3 masked vals + 3 mask bits -> d
        self.proj_e = nn.Linear(EE_DIM, d)              # ee slot: 15 -> d
        self.mod = nn.Parameter(torch.randn(3, d) * 0.02)          # modality emb (v, m, e)
        self.pos_m = nn.Parameter(torch.randn(N_MOTOR, d) * 0.02)  # which motor row
        self.pos_e = nn.Parameter(torch.randn(EE_T, d) * 0.02)     # ee slot (time placeholder)
        self.fuse = PerceiverFuse(d, n_queries)
        self.pred = nn.ModuleDict({k: _mlp(d, d) for k in ("v", "m", "e")})
        self.tgt_v = copy.deepcopy(self.proj_v)
        self.tgt_m = copy.deepcopy(self.proj_m)
        self.tgt_e = copy.deepcopy(self.proj_e)
        for t in (self.tgt_v, self.tgt_m, self.tgt_e):
            for p in t.parameters():
                p.requires_grad = False
        self.sigreg = SlicedEppsPulley(num_slices=n_slices)
        self.lamb, self.ema, self.n_queries = lamb, ema, n_queries

    @torch.no_grad()
    def update_target(self):
        for o, t in [(self.proj_v, self.tgt_v), (self.proj_m, self.tgt_m),
                     (self.proj_e, self.tgt_e)]:
            for po, pt in zip(o.parameters(), t.parameters()):
                pt.mul_(self.ema).add_(po.detach(), alpha=1 - self.ema)

    @staticmethod
    def unpack(batch, dev):
        """Squeeze the packet's singleton time axes and move to device."""
        return (batch["rgb"].squeeze(1).to(dev, non_blocking=True),        # [B,196,768]
                batch["motor"].squeeze(1).to(dev, non_blocking=True),      # [B,8,3]
                batch["motor_mask"].to(dev, non_blocking=True),            # [B,8,3] True=valid
                batch["ee"].to(dev, non_blocking=True),                    # [B,13,15]
                batch["ee_mask"].to(dev, non_blocking=True))               # [B,13]  True=valid

    @staticmethod
    def motor_feats(motor, m_mask):
        # zero invalid channels, append the mask bits -> [B, 8, 6]
        return torch.cat([motor * m_mask, m_mask.float()], dim=-1)

    def _context(self, rgb, mfeat, ee):
        vt = self.proj_v(rgb) + self.mod[0]                  # [B,196,d]
        mt = self.proj_m(mfeat) + self.mod[1] + self.pos_m   # [B,8,d]
        et = self.proj_e(ee) + self.mod[2] + self.pos_e      # [B,13,d]
        return torch.cat([vt, mt, et], dim=1)                # [B,217,d]

    def _attn_mask(self, m_row_valid, e_mask, hide=()):
        """[B, M, 217] bool, True=BLOCKED: invalid tokens always + modalities in `hide`."""
        B, dev = m_row_valid.shape[0], m_row_valid.device
        bv = torch.full((B, N_PATCH), "v" in hide, dtype=torch.bool, device=dev)
        bm = torch.ones_like(m_row_valid) if "m" in hide else ~m_row_valid
        be = torch.ones_like(e_mask) if "e" in hide else ~e_mask
        blocked = torch.cat([bv, bm, be], dim=1)             # [B,217]
        return blocked.unsqueeze(1).expand(B, self.n_queries, T_CTX)

    def forward(self, rgb, motor, m_mask, ee, e_mask):
        mfeat = self.motor_feats(motor, m_mask)
        ctx = self._context(rgb, mfeat, ee)
        m_row_valid = m_mask.any(-1)                         # [B,8]
        e_any = e_mask.any(-1)                               # [B]

        with torch.no_grad():
            tv = self.tgt_v(rgb).mean(1)                     # [B,d]
            tm = masked_mean(self.tgt_m(mfeat), m_row_valid)
            te = masked_mean(self.tgt_e(ee), e_mask)         # zeros where ~e_any (excluded)

        z_no_v = self.fuse(ctx, self._attn_mask(m_row_valid, e_mask, hide=("v",)))
        z_no_m = self.fuse(ctx, self._attn_mask(m_row_valid, e_mask, hide=("m",)))
        z_no_e = self.fuse(ctx, self._attn_mask(m_row_valid, e_mask, hide=("e",)))

        inv = (self.pred["v"](z_no_v) - tv).square().mean() \
            + (self.pred["m"](z_no_m) - tm).square().mean()
        if bool(e_any.any()):
            inv = inv + (self.pred["e"](z_no_e[e_any]) - te[e_any]).square().mean()

        ev = self.proj_v(rgb).mean(1)                        # online per-modal embs (grad)
        em = masked_mean(self.proj_m(mfeat), m_row_valid)
        z_full = self.fuse(ctx, self._attn_mask(m_row_valid, e_mask))
        sig = self.sigreg(ev) + self.sigreg(em) + self.sigreg(z_full)
        if int(e_any.sum()) >= 8:                            # enough samples for a stable term
            sig = sig + self.sigreg(masked_mean(self.proj_e(ee), e_mask)[e_any])
        loss = inv + self.lamb * sig
        return {"loss": loss, "inv": inv.detach(), "sig": sig.detach(),
                "z": z_full.detach()}

    @torch.no_grad()
    def embed_vision(self, rgb, motor, m_mask, ee, e_mask):
        """z_v: fused latent from VISION ONLY (motor + ee hidden) — the eval latent."""
        ctx = self._context(rgb, self.motor_feats(motor, m_mask), ee)
        return self.fuse(ctx, self._attn_mask(m_mask.any(-1), e_mask, hide=("m", "e")))
