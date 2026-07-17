"""v1 temporal (Phase-2) — flat multi-rate Perceiver over a window of C ticks.

Design: TEMPORAL_ARCH.md §5 (flat) + §16 (build target). Extends mm_perceiver3's single-tick
cross-modal JEPA to a window of C ticks:

  1. tokenize each tick's streams (v0.1 linear projections == Conv1d k=1),
  2. tag every token with modality + within-modality position + a CONTINUOUS-TIME embedding
     (mTAN-style Fourier features off the real per-tick ts),
  3. flatten all C*(196+8+13) tokens into ONE flat context and fuse with a single PerceiverBlock
     (N latents cross-attend, then latent self-attention) -> belief [B, N, d] (not pooled),
  4. predict held-out (modality x tick) cells from the belief via a query-based predictor whose
     query carries the (modality, time) coordinate,
  5. loss = masked cross-modal-across-time latent prediction (MSE, predict-don't-equate)
     + per-timestep SIGReg. Same family as v0.1.

Packet masks are True=VALID; CrossAttention attn_mask is True=BLOCKED (inverted here). ee can be
fully absent at a tick (cfg5); those cells are excluded from the ee prediction/SIGReg terms.
proj_v/proj_m/proj_e/mod/pos_m/pos_e keep v0.1 names so an mm_perceiver3 checkpoint warm-starts.

NOTE (v1 on the subsampled cache): the ee 13-slot block is treated as a per-tick snippet (linear
tokenizer), NOT the continuous C*13 1-D-CNN stream in §3.3 — that needs a dense re-precompute.
"""
import copy
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_pretraining.backbone.vit import CrossAttention
from stable_pretraining.methods.lejepa import SlicedEppsPulley

N_PATCH, N_MOTOR, MOTOR_CH, EE_T, EE_DIM = 196, 8, 3, 13, 15


def _mlp(i, o, h=None):
    h = h or o
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


def masked_mean(x, mask):
    # x:[...,T,d] mask:[...,T] bool True=valid -> [...,d]; zero rows where none valid
    m = mask.unsqueeze(-1).float()
    return (x * m).sum(-2) / m.sum(-2).clamp(min=1.0)


class TimeEmbed(nn.Module):
    """Continuous-time Fourier embedding (mTAN-style), log-spaced periods ~ms->s. Added to tokens.
    Frequency bank spans [min_period_ms, max_period_ms] — the make-or-break knob (§4)."""

    def __init__(self, d, n_freq=None, min_period_ms=10.0, max_period_ms=100_000.0):
        super().__init__()
        n_freq = n_freq or d // 2
        periods = torch.logspace(math.log10(min_period_ms), math.log10(max_period_ms), n_freq)
        self.register_buffer("omega", 2 * math.pi / periods)          # [n_freq]
        self.proj = nn.Linear(2 * n_freq, d)

    def forward(self, t_ms):                                          # t_ms:[B,C] float
        a = t_ms.unsqueeze(-1) * self.omega                          # [B,C,n_freq]
        return self.proj(torch.cat([a.sin(), a.cos()], dim=-1))      # [B,C,d]


class PerceiverBlock(nn.Module):
    """N learned latents cross-attend a context, then L latent self-attention blocks.
    Returns [B,N,d] (pool=None) or [B,d] (pool='mean'). latents= overrides the learned array."""

    def __init__(self, d, n_latents=64, n_cross=1, n_self=4, n_heads=8, ff_mult=4, pool=None):
        super().__init__()
        self.q = nn.Parameter(torch.randn(n_latents, d) * 0.02)
        self.ca = nn.ModuleList([CrossAttention(d, d, n_heads) for _ in range(n_cross)])
        self.ca_n = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_cross)])
        self.ca_ff = nn.ModuleList([_mlp(d, d, ff_mult * d) for _ in range(n_cross)])
        self.ca_fn = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_cross)])
        self.sa = nn.ModuleList([nn.MultiheadAttention(d, n_heads, batch_first=True)
                                 for _ in range(n_self)])
        self.sa_n = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_self)])
        self.sa_ff = nn.ModuleList([_mlp(d, d, ff_mult * d) for _ in range(n_self)])
        self.sa_fn = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_self)])
        self.out_norm = nn.LayerNorm(d)                     # bound latent scale (no drift)
        self.n_latents, self.pool = n_latents, pool

    def forward(self, context, attn_mask=None, latents=None):
        B = context.shape[0]
        x = latents if latents is not None else self.q.unsqueeze(0).expand(B, -1, -1)
        for ca, n1, ff, n2 in zip(self.ca, self.ca_n, self.ca_ff, self.ca_fn):
            x = x + ca(n1(x), context, attn_mask=attn_mask)
            x = x + ff(n2(x))
        for sa, n1, ff, n2 in zip(self.sa, self.sa_n, self.sa_ff, self.sa_fn):
            h = n1(x)
            x = x + sa(h, h, h, need_weights=False)[0]
            x = x + ff(n2(x))
        x = self.out_norm(x)
        return x.mean(1) if self.pool == "mean" else x


class MMPerceiverTemporal(nn.Module):
    MOD = {"v": 0, "m": 1, "e": 2}

    def __init__(self, d=256, vis_dim=768, n_latents=64, n_cross=1, n_self=4, n_heads=8,
                 lamb=0.02, ema=0.99, n_slices=512, mask_ratio=0.5, mask_mode="mixed",
                 norm_pred=True, pred_mode="query", joint_sigreg=False):
        super().__init__()
        self.proj_v = nn.Linear(vis_dim, d)
        self.proj_m = nn.Linear(2 * MOTOR_CH, d)
        self.proj_e = nn.Linear(EE_DIM, d)
        self.mod = nn.Parameter(torch.randn(3, d) * 0.02)
        self.pos_m = nn.Parameter(torch.randn(N_MOTOR, d) * 0.02)
        self.pos_e = nn.Parameter(torch.randn(EE_T, d) * 0.02)
        self.time = TimeEmbed(d)
        self.fuse = PerceiverBlock(d, n_latents, n_cross, n_self, n_heads, pool=None)
        # query-based predictor: query = per-modality seed + time(cell) -> predict cell target emb
        self.pred_q = nn.Parameter(torch.randn(3, d) * 0.02)
        self.dec = CrossAttention(d, d, n_heads)
        self.dec_n = nn.LayerNorm(d)
        self.pred_head = _mlp(d, d)
        # v0.1-style head: per-modality MLP on the POOLED fused latent (§18.12). Restores the
        # proven cross-modal recipe; predicts each modality's time-pooled EMA target.
        self.pred_mlp = nn.ModuleDict({k: _mlp(d, d) for k in ("v", "m", "e")})
        self.tgt_v = copy.deepcopy(self.proj_v)
        self.tgt_m = copy.deepcopy(self.proj_m)
        self.tgt_e = copy.deepcopy(self.proj_e)
        for t in (self.tgt_v, self.tgt_m, self.tgt_e):
            for p in t.parameters():
                p.requires_grad = False
        self.sigreg = SlicedEppsPulley(num_slices=n_slices)
        self.d, self.lamb, self.ema, self.mask_ratio = d, lamb, ema, mask_ratio
        self.mask_mode = mask_mode
        self.norm_pred = norm_pred      # L2-normalize pred+target (cosine) vs raw MSE (v0.1-style, §18.11)
        self.pred_mode = pred_mode      # "query" (temporal decoder) | "v01" (per-modal MLP on pooled z, §18.12)
        self.joint_sigreg = joint_sigreg  # SIGReg the fused latent too (v0.1 had it; stabilizes raw target)

    @torch.no_grad()
    def update_target(self):
        for o, t in [(self.proj_v, self.tgt_v), (self.proj_m, self.tgt_m), (self.proj_e, self.tgt_e)]:
            for po, pt in zip(o.parameters(), t.parameters()):
                pt.mul_(self.ema).add_(po.detach(), alpha=1 - self.ema)

    @staticmethod
    def motor_feats(motor, m_mask):
        return torch.cat([motor * m_mask, m_mask.float()], dim=-1)         # [...,8,6]

    def _tokenize(self, rgb, mfeat, ee, t_ms):
        """rgb[B,C,196,768] mfeat[B,C,8,6] ee[B,C,13,15] t_ms[B,C] -> per-stream tokens + time-tagged."""
        B, C = rgb.shape[:2]
        te = self.time(t_ms)                                              # [B,C,d]
        vt = self.proj_v(rgb) + self.mod[0] + te.unsqueeze(2)             # [B,C,196,d]
        mt = self.proj_m(mfeat) + self.mod[1] + self.pos_m + te.unsqueeze(2)   # [B,C,8,d]
        et = self.proj_e(ee) + self.mod[2] + self.pos_e + te.unsqueeze(2)      # [B,C,13,d]
        return vt, mt, et

    def _context(self, vt, mt, et):
        B, C = vt.shape[:2]
        return torch.cat([vt.reshape(B, C * N_PATCH, -1),
                          mt.reshape(B, C * N_MOTOR, -1),
                          et.reshape(B, C * EE_T, -1)], dim=1)            # [B, C*217, d]

    def forward(self, rgb, motor, m_mask, ee, e_mask, t_ms):
        """rgb[B,C,196,768] motor[B,C,8,3] m_mask[B,C,8,3] ee[B,C,13,15] e_mask[B,C,13] t_ms[B,C]."""
        B, C, dev = rgb.shape[0], rgb.shape[1], rgb.device
        mfeat = self.motor_feats(motor, m_mask)
        vt, mt, et = self._tokenize(rgb, mfeat, ee, t_ms)
        M = C * (N_PATCH + N_MOTOR + EE_T)

        # per-token validity along the flat context (True=valid). vision always valid.
        m_tok = m_mask.any(-1)                                            # [B,C,8]
        e_tok = e_mask                                                    # [B,C,13]
        valid = torch.cat([torch.ones(B, C * N_PATCH, dtype=torch.bool, device=dev),
                           m_tok.reshape(B, C * N_MOTOR),
                           e_tok.reshape(B, C * EE_T)], dim=1)            # [B,M]

        if self.pred_mode == "v01":
            return self._forward_v01(rgb, mfeat, ee, vt, mt, et, valid, m_tok, e_tok, B, C, dev)

        # --- choose masked cells (cell = (modality, tick)); mask_mode picks the scheme. ---
        # "modality": hide ALL C ticks of one modality -> forces CROSS-MODAL prediction (v0.1-style;
        #   this is what makes vision-only z_v carry force). "future": hide later ticks of all
        #   modalities -> temporal/future. "mixed": 50/50. "cell": old random-subset (too weak —
        #   lets the model predict a masked cell from same-modality neighbors; see §18.9).
        mode = self.mask_mode
        if mode == "mixed":
            mode = "modality" if random.random() < 0.5 else "future"
        if mode == "modality":
            m = random.randint(0, 2)
            cell_mod = torch.full((C,), m, dtype=torch.long, device=dev)
            cell_tick = torch.arange(C, device=dev)
        elif mode == "future":
            s = max(1, C // 2)
            tk = torch.arange(s, C, device=dev)
            cell_mod = torch.arange(3, device=dev).repeat_interleave(tk.numel())
            cell_tick = tk.repeat(3)
        else:                                                            # "cell" (old random subset)
            n_cells = 3 * C
            kk = min(max(1, int(round(self.mask_ratio * n_cells))), n_cells - 1)
            cell = torch.randperm(n_cells, device=dev)[:kk]
            cell_mod, cell_tick = cell // C, cell % C
        k = cell_mod.numel()

        # block masked cells + invalid tokens in the fuse context
        blocked = ~valid                                                  # [B,M]
        seg = [(0, N_PATCH), (C * N_PATCH, N_MOTOR), (C * N_PATCH + C * N_MOTOR, EE_T)]
        for m_id, tk in zip(cell_mod.tolist(), cell_tick.tolist()):
            base, width = seg[m_id]
            s = base + tk * width
            blocked[:, s:s + width] = True
        ctx = self._context(vt, mt, et)                                   # [B,M,d]
        amask = blocked.unsqueeze(1).expand(B, self.fuse.n_latents, M)    # [B,N,M] True=blocked
        z = self.fuse(ctx, amask)                                         # [B,N,d]

        # --- targets: pooled EMA-target embedding per masked cell ---
        with torch.no_grad():
            tv = self.tgt_v(rgb).mean(2)                                  # [B,C,d]
            tm = masked_mean(self.tgt_m(mfeat), m_tok)                    # [B,C,d]
            teg = masked_mean(self.tgt_e(ee), e_tok)                      # [B,C,d]
            tgt_all = torch.stack([tv, tm, teg], dim=1)                   # [B,3,C,d]
        target = tgt_all[:, cell_mod, cell_tick]                         # [B,k,d]

        # --- query-based predictor over masked cells ---
        q = self.pred_q[cell_mod] + self.time(t_ms)[:, cell_tick]        # [B,k,d]
        pred = self.pred_head(self.dec(self.dec_n(q), z))               # [B,k,d]
        if self.norm_pred:                                              # cosine (default) vs raw MSE
            pred = F.normalize(pred, dim=-1)
            target = F.normalize(target, dim=-1)                        # scale-free JEPA target

        # per-cell loss weight: exclude ee cells with no valid ee for a given sample
        w = torch.ones(B, k, device=dev)
        e_any = e_tok.any(-1).float()                                    # [B,C]
        is_e = (cell_mod == self.MOD["e"])
        if is_e.any():
            w[:, is_e] = e_any[:, cell_tick[is_e]]
        inv = ((pred - target).square().sum(-1) * w).sum() / w.sum().clamp(min=1.0)

        # --- per-timestep SIGReg on online marginals (each instant, never time-pooled) ---
        ev = self.proj_v(rgb).mean(2).reshape(B * C, -1)                 # [B*C,d]
        em = masked_mean(self.proj_m(mfeat), m_tok).reshape(B * C, -1)
        sig = self.sigreg(ev) + self.sigreg(em)
        ee_valid = e_tok.any(-1).reshape(B * C)
        if int(ee_valid.sum()) >= 8:
            ee_emb = masked_mean(self.proj_e(ee), e_tok).reshape(B * C, -1)[ee_valid]
            sig = sig + self.sigreg(ee_emb)

        loss = inv + self.lamb * sig
        return {"loss": loss, "inv": inv.detach(), "sig": sig.detach(), "z": z.detach()}

    def _hide(self, valid, C, mods):
        """[B,M] blocked mask: invalid tokens + every token of each modality in `mods`."""
        blk = ~valid.clone()
        vseg, mseg = C * N_PATCH, C * N_PATCH + C * N_MOTOR
        if "v" in mods:
            blk[:, :vseg] = True
        if "m" in mods:
            blk[:, vseg:mseg] = True
        if "e" in mods:
            blk[:, mseg:] = True
        return blk

    def _forward_v01(self, rgb, mfeat, ee, vt, mt, et, valid, m_tok, e_tok, B, C, dev):
        """v0.1-style cross-modal JEPA on the temporal fuse: three hide-one-modality passes,
        per-modality MLP on the POOLED fused latent -> each modality's TIME-POOLED raw EMA target.
        Latents still attend across all C ticks (temporal fusion kept); only the head reverts to the
        proven recipe. Joint SIGReg on the fused latent stabilizes the raw (unnormalized) target."""
        ctx = self._context(vt, mt, et)                                  # [B,M,d]
        N = self.fuse.n_latents

        def zp(mods):                                                    # pooled fused latent, mods hidden
            return self.fuse(ctx, self._hide(valid, C, mods).unsqueeze(1).expand(B, N, -1)).mean(1)

        with torch.no_grad():                                            # time-pooled raw targets [B,d]
            tv = self.tgt_v(rgb).mean(2).mean(1)
            tm = masked_mean(self.tgt_m(mfeat), m_tok).mean(1)
            te_tick = masked_mean(self.tgt_e(ee), e_tok)                 # [B,C,d]
            tick_ok = e_tok.any(-1).float()                              # [B,C]
            te = (te_tick * tick_ok.unsqueeze(-1)).sum(1) / tick_ok.sum(1).clamp(min=1).unsqueeze(-1)
        e_any = e_tok.any(-1).any(-1)                                    # [B] any valid ee in window

        inv = (self.pred_mlp["v"](zp("v")) - tv).square().mean() \
            + (self.pred_mlp["m"](zp("m")) - tm).square().mean()
        if bool(e_any.any()):
            z_e = zp("e")
            inv = inv + (self.pred_mlp["e"](z_e[e_any]) - te[e_any]).square().mean()

        ev = self.proj_v(rgb).mean(2).reshape(B * C, -1)                 # per-timestep marginals
        em = masked_mean(self.proj_m(mfeat), m_tok).reshape(B * C, -1)
        sig = self.sigreg(ev) + self.sigreg(em)
        ee_valid = e_tok.any(-1).reshape(B * C)
        if int(ee_valid.sum()) >= 8:
            ee_emb = masked_mean(self.proj_e(ee), e_tok).reshape(B * C, -1)[ee_valid]
            sig = sig + self.sigreg(ee_emb)
        if self.joint_sigreg:                                           # bound the fused latent (anti-drift)
            z_full = self.fuse(ctx, (~valid).unsqueeze(1).expand(B, N, -1)).mean(1)
            sig = sig + self.sigreg(z_full)

        loss = inv + self.lamb * sig
        return {"loss": loss, "inv": inv.detach(), "sig": sig.detach(), "z": zp("v").detach()}

    @torch.no_grad()
    def embed_vision_tokens(self, rgb, motor, m_mask, ee, e_mask, t_ms):
        """Un-pooled vision-only latent array [B, N, d] (motor + ee hidden). For diagnostics /
        alternative (non-mean-pool) readouts."""
        B, C, dev = rgb.shape[0], rgb.shape[1], rgb.device
        mfeat = self.motor_feats(motor, m_mask)
        vt, mt, et = self._tokenize(rgb, mfeat, ee, t_ms)
        M = C * (N_PATCH + N_MOTOR + EE_T)
        blocked = torch.zeros(B, M, dtype=torch.bool, device=dev)
        blocked[:, C * N_PATCH:] = True                                   # hide motor + ee
        return self.fuse(self._context(vt, mt, et),
                         blocked.unsqueeze(1).expand(B, self.fuse.n_latents, M))   # [B,N,d]

    @torch.no_grad()
    def embed_vision(self, rgb, motor, m_mask, ee, e_mask, t_ms):
        """Vision-only window belief, mean-pooled to [B,d] (the default eval readout)."""
        return self.embed_vision_tokens(rgb, motor, m_mask, ee, e_mask, t_ms).mean(1)
