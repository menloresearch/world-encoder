"""Stage-2-at-scale — Perceiver cross-modal JEPA with optional multi-camera vision.

Extends mm_perceiver2.py's 3-modality chunk recipe (vision + motor + ee) to accept an
OPTIONAL LIST of vision tensors, one per camera, for multi-camera setups (e.g. a front
cam + a wrist cam). Multi-camera is a genuine structural difference in the input rather
than an extra tensor axis, so cameras are passed as a list.

Guiding principle: multi-camera and single-camera get IDENTICAL treatment — the cameras
are treated as one big camera. Each camera is encoded the same way (196 ViT latents) and
the per-camera latents are simply APPENDED along the token axis in the fuse step: K
cameras -> K*196 vision tokens, concatenated with the 8 motor rows and 13 ee slots into
one context. The Perceiver's learned queries compress whatever number of vision tokens
are present. The vision target / online embedding / SIGReg all pool over ALL K*196 patch
tokens into a single [B, d] vision embedding — one vision target, one SIGReg term. With a
single camera (K=1) every path collapses exactly to mm_perceiver2's behavior.

`rgb` may be a bare tensor [B, 196, 768] (one camera, backward-compatible) or a list
[[B, 196, 768], ...] of length K (K cameras). Everything else (motor, ee, masks, losses)
is unchanged from mm_perceiver2.

Packet validity masks are True=VALID; CrossAttention attn_mask is True=BLOCKED —
inverted in _attn_mask. ee can be entirely absent: those samples are excluded from the ee
prediction/SIGReg terms rather than averaged over zero valid elements (NaN trap). No
masked fuse pass can ever see an all-blocked context: vision and motor are always
present, and each hide-one pass keeps at least one of them visible. Vision is one modality
regardless of camera count — hiding "v" blocks all cameras together.
"""
import copy

import torch
import torch.nn as nn

from stable_pretraining.methods.lejepa import SlicedEppsPulley

from world_tokenizer.mm_perceiver import PerceiverFuse, _mlp

N_PATCH, N_MOTOR, MOTOR_CH, EE_T, EE_DIM = 196, 8, 3, 13, 15


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
    def _rgb_list(rgb):
        # Normalize vision input to a list of per-camera tensors. Single choke point so
        # every method handles both a bare tensor (one camera) and a list (K cameras).
        return list(rgb) if isinstance(rgb, (list, tuple)) else [rgb]

    @staticmethod
    def motor_feats(motor, m_mask):
        # zero invalid channels, append the mask bits -> [B, 8, 6]
        return torch.cat([motor * m_mask, m_mask.float()], dim=-1)

    def _context(self, rgb, mfeat, ee):
        cams = self._rgb_list(rgb)
        vt = torch.cat([self.proj_v(c) for c in cams], dim=1) + self.mod[0]  # [B,K*196,d]
        mt = self.proj_m(mfeat) + self.mod[1] + self.pos_m                   # [B,8,d]
        et = self.proj_e(ee) + self.mod[2] + self.pos_e                      # [B,13,d]
        return torch.cat([vt, mt, et], dim=1)                               # [B,K*196+21,d]

    def _attn_mask(self, n_vis, m_row_valid, e_mask, hide=()):
        """[B, M, K*196+21] bool, True=BLOCKED: invalid tokens always + modalities in
        `hide`. n_vis = K*196 vision tokens; all cameras are blocked/shown together."""
        B, dev = m_row_valid.shape[0], m_row_valid.device
        t_ctx = n_vis + N_MOTOR + EE_T
        bv = torch.full((B, n_vis), "v" in hide, dtype=torch.bool, device=dev)
        bm = torch.ones_like(m_row_valid) if "m" in hide else ~m_row_valid
        be = torch.ones_like(e_mask) if "e" in hide else ~e_mask
        blocked = torch.cat([bv, bm, be], dim=1)             # [B, K*196+21]
        return blocked.unsqueeze(1).expand(B, self.n_queries, t_ctx)

    def forward(self, rgb, motor, m_mask, ee, e_mask):
        cams = self._rgb_list(rgb)
        n_vis = sum(c.shape[1] for c in cams)                # K*196
        mfeat = self.motor_feats(motor, m_mask)
        ctx = self._context(cams, mfeat, ee)
        m_row_valid = m_mask.any(-1)                         # [B,8]
        e_any = e_mask.any(-1)                               # [B]

        with torch.no_grad():
            # concat post-tgt_v tokens across cameras, then mean-pool -> [B,d]
            tv = torch.cat([self.tgt_v(c) for c in cams], dim=1).mean(1)
            tm = masked_mean(self.tgt_m(mfeat), m_row_valid)
            te = masked_mean(self.tgt_e(ee), e_mask)         # zeros where ~e_any (excluded)

        z_no_v = self.fuse(ctx, self._attn_mask(n_vis, m_row_valid, e_mask, hide=("v",)))
        z_no_m = self.fuse(ctx, self._attn_mask(n_vis, m_row_valid, e_mask, hide=("m",)))
        z_no_e = self.fuse(ctx, self._attn_mask(n_vis, m_row_valid, e_mask, hide=("e",)))

        inv = (self.pred["v"](z_no_v) - tv).square().mean() \
            + (self.pred["m"](z_no_m) - tm).square().mean()
        if bool(e_any.any()):
            inv = inv + (self.pred["e"](z_no_e[e_any]) - te[e_any]).square().mean()

        # online vision emb: concat post-proj_v across cameras, then mean-pool -> [B,d]
        ev = torch.cat([self.proj_v(c) for c in cams], dim=1).mean(1)
        em = masked_mean(self.proj_m(mfeat), m_row_valid)
        z_full = self.fuse(ctx, self._attn_mask(n_vis, m_row_valid, e_mask))
        sig = self.sigreg(ev) + self.sigreg(em) + self.sigreg(z_full)
        if int(e_any.sum()) >= 8:                            # enough samples for a stable term
            sig = sig + self.sigreg(masked_mean(self.proj_e(ee), e_mask)[e_any])
        loss = inv + self.lamb * sig
        return {"loss": loss, "inv": inv.detach(), "sig": sig.detach(),
                "z": z_full.detach()}

    @torch.no_grad()
    def embed_vision(self, rgb, motor, m_mask, ee, e_mask):
        """z_v: fused latent from VISION ONLY (motor + ee hidden) — the eval latent."""
        cams = self._rgb_list(rgb)
        n_vis = sum(c.shape[1] for c in cams)
        ctx = self._context(cams, self.motor_feats(motor, m_mask), ee)
        return self.fuse(ctx, self._attn_mask(n_vis, m_mask.any(-1), e_mask,
                                              hide=("m", "e")))
