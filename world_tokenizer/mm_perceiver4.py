"""Stage-2 — Perceiver cross-modal JEPA with explicit [B, T, S, P] axes (mm_perceiver4).

Rebuild of mm_perceiver3.py that stops confusing the temporal, spatial/sensor, and
physical axes of each modality. Every modality enters as a 4-D tensor

    [B, T, S, P]   Batch, Time (chunksize*sampling_rate), Spatial/sensor, Physical

and is reduced to one [B, d] world latent in three named stages, all per-modality until
the last:

  1. STEM  (physical -> d, TEMPORAL):  a Conv1d over T maps P -> d. Physics is a time
     series, so the P->d channel map is a temporal convolution, not a per-frame linear.
     Processed as [B*S, P, T] -> [B*S, d, T].
  2. SPATIAL compression (S -> n_s), config `spatial="perceiver"|"avgpool"`:
     Perceiver queries over the sensor axis, or a plain mean (n_s=1). Simple is a valid
     choice; both are exposed.
  3. TEMPORAL resampling (T -> R): Perceiver queries over time — the Perceiver used as a
     RESAMPLER so every modality, whatever its native rate, lands on a common R. This is
     the point of the redesign: alignment happens on the time axis, per modality, before
     any cross-modal mixing. T=1 modalities (rgb/motor in the current cache) pass through
     the same resampler as a learned broadcast — no special-casing, so the T=1 ablation
     compares architectures, not code paths.

Only after all modalities share [B, R, n_s, d] do we FUSE:

  4. FUSION (flattened): stack the 3 modalities -> [B, R, 3*n_s, d], flatten the
     modality/sensor/feature axes -> [B, R, w] with w = 3*n_s*d. A stack of WideBlocks
     attends over the R time tokens: the residual stream and FFN stay WIDE (w), while
     attention projects down to a smaller d_attn for Q/K/V and back up on output (W_v/W_o
     ARE the cross-modal remap; d_attn is configurable and need not equal d). Pool over R
     -> [B, w] then read out w -> d.

Hiding a modality for the cross-prediction objective is FEATURE-ZEROING here: with a
flattened fuse there is no attention column to block, so we zero the modality's n_s*d
slice of the wide token before the WideBlock stack. The mm_perceiver3 objective is otherwise unchanged
(hide-one cross-prediction + per-modal SIGReg + joint SIGReg, EMA targets).

Validity masks (True=VALID): motor_mask [B,8,3], ee_mask [B,T_ee]. ee_mask is a mask over
the TIME axis (a correction over mm_perceiver3, where the 13 ee ticks were treated as
tokens) and is applied inside ee's temporal resampler. Samples with no valid ee at all
are excluded from the ee loss terms rather than averaged over zero valid entries.

Order is fixed A (spatial then temporal); the flattened linear fuse makes order C
impossible and the spatial/temporal split is the only remaining compression ordering.
"""
import copy
import math

import torch
import torch.nn as nn

from stable_pretraining.methods.lejepa import SlicedEppsPulley

from world_tokenizer.mm_perceiver import PerceiverFuse, _mlp

# Physical-dim (P) width per modality, and the sensor width (S) each carries.
P_V, P_M, P_E = 768, 6, 15          # vision 768; motor 3 vals+3 mask bits; ee F/T+tcp+rot6
MODALITIES = ("v", "m", "e")


class Conv1dStem(nn.Module):
    """Temporal P->d map: Conv1d over the time axis. Input [B, T, S, P] -> [B, T, S, d].

    Physics is temporal, so channels (P) are mixed by a convolution along T. kernel=1
    degrades to a per-timestep linear, which is the right behavior for T=1 modalities.
    """

    def __init__(self, p_in, d, kernel=3):
        super().__init__()
        # 'same' padding keeps T; kernel is clamped so a T=1 modality stays valid.
        self.conv = nn.Conv1d(p_in, d, kernel_size=kernel, padding=kernel // 2)

    def forward(self, x):
        B, T, S, P = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B * S, P, T)   # [B*S, P, T]
        x = self.conv(x)                                 # [B*S, d, T]
        return x.reshape(B, S, -1, T).permute(0, 3, 1, 2)  # [B, T, S, d]


class SpatialCompress(nn.Module):
    """S -> n_s per (B, T). mode='perceiver' (learned queries) or 'avgpool' (n_s=1)."""

    def __init__(self, d, n_s, mode="perceiver"):
        super().__init__()
        self.mode, self.n_s = mode, (n_s if mode == "perceiver" else 1)
        if mode == "perceiver":
            self.fuse = PerceiverFuse(d, n_s)
        elif mode != "avgpool":
            raise ValueError(f"spatial mode {mode!r} not in ('perceiver', 'avgpool')")

    def forward(self, x):
        # x: [B, T, S, d] -> [B, T, n_s, d]
        B, T, S, d = x.shape
        if self.mode == "avgpool":
            return x.mean(2, keepdim=True)               # [B, T, 1, d]
        flat = x.reshape(B * T, S, d)                    # [B*T, S, d]
        # PerceiverFuse pools its queries to [., d]; we want the per-query tokens, so run
        # the query stack and skip the mean by re-expanding. Simplest: call fuse and treat
        # its pooled output as n_s=1 when n_s==1, else use the pre-pool path below.
        return _perceiver_tokens(self.fuse, flat).reshape(B, T, self.n_s, d)


def _perceiver_tokens(fuse, context, attn_mask=None):
    """Run a PerceiverFuse query stack but return the per-query tokens [., M, d] instead
    of the pooled mean. Mirrors PerceiverFuse.forward without the final .mean(1)."""
    x = fuse.q.unsqueeze(0).expand(context.shape[0], -1, -1)
    for ca, n1, ffn, n2 in zip(fuse.ca, fuse.n1, fuse.ffn, fuse.n2):
        x = x + ca(n1(x), context, attn_mask=attn_mask)
        x = x + ffn(n2(x))
    return x                                             # [., M, d]


class TemporalResample(nn.Module):
    """T -> R per (B, n_s). Perceiver queries over time; validity mask over T supported."""

    def __init__(self, d, R):
        super().__init__()
        self.R, self.d = R, d
        self.fuse = PerceiverFuse(d, R)

    def forward(self, x, t_valid=None):
        # x: [B, T, n_s, d] -> [B, R, n_s, d]. t_valid: [B, T] True=valid (optional).
        B, T, n_s, d = x.shape
        ctx = x.permute(0, 2, 1, 3).reshape(B * n_s, T, d)   # [B*n_s, T, d]
        mask = None
        if t_valid is not None:
            # blocked = ~valid, broadcast to every query row and every sensor slot.
            blk = (~t_valid).unsqueeze(1).expand(B, n_s, T).reshape(B * n_s, T)
            mask = blk.unsqueeze(1).expand(B * n_s, self.R, T)   # [B*n_s, R, T] True=blocked
        out = _perceiver_tokens(self.fuse, ctx, mask)            # [B*n_s, R, d]
        return out.reshape(B, n_s, self.R, d).permute(0, 2, 1, 3)  # [B, R, n_s, d]


class Modality(nn.Module):
    """One modality's stem -> spatial compress -> temporal resample (order A)."""

    def __init__(self, p_in, d, n_s, R, spatial, kernel):
        super().__init__()
        self.stem = Conv1dStem(p_in, d, kernel)
        self.spatial = SpatialCompress(d, n_s, spatial)
        self.temporal = TemporalResample(d, R)
        self.n_s = self.spatial.n_s

    def forward(self, x, t_valid=None):
        x = self.stem(x)                     # [B, T, S, d]
        x = self.spatial(x)                  # [B, T, n_s, d]
        return self.temporal(x, t_valid)     # [B, R, n_s, d]


class MLA(nn.Module):
    """Multi-head Latent Attention over a wide stream (w). Q and K/V are compressed through
    separate latent ranks (q_latent, kv_latent); the up-projected value goes back to w on
    output. This is the low-rank factorization of WideBlock's down/up attention remap:
    W_qk absorbs W_q^U W_k^U so scores are computed in the kv-latent space directly.
    """

    def __init__(self, w, num_heads, q_latent, kv_latent):
        super().__init__()
        assert w % num_heads == 0, f"w {w} not divisible by num_heads {num_heads}"
        self.num_heads, self.kv_latent = num_heads, kv_latent
        self.head_dim = w // num_heads
        self.Wq_d = nn.Linear(w, q_latent)                        # w -> q_latent
        self.W_qk = nn.Linear(q_latent, num_heads * kv_latent)    # absorbed W_q^U W_k^U
        self.Wkv_d = nn.Linear(w, kv_latent)                      # w -> kv_latent
        self.Wv_u = nn.Linear(kv_latent, num_heads * self.head_dim)
        self.Wo = nn.Linear(num_heads * self.head_dim, w)         # heads -> w (wide out)

    def forward(self, x):
        B, R, _ = x.shape
        C_q = self.Wq_d(x)                                        # [B, R, q_latent]
        C_kv = self.Wkv_d(x)                                      # [B, R, kv_latent]
        q = self.W_qk(C_q).view(B, R, self.num_heads, self.kv_latent).transpose(1, 2)
        scores = torch.matmul(q, C_kv[:, None].transpose(-2, -1)) / math.sqrt(self.kv_latent)
        attn = torch.softmax(scores, dim=-1)                     # [B, H, R, R]
        V = self.Wv_u(C_kv).view(B, R, self.num_heads, self.head_dim).transpose(1, 2)
        o = torch.matmul(attn, V).transpose(1, 2).reshape(B, R, -1)  # [B, R, H*head_dim]
        return self.Wo(o)                                        # [B, R, w]


class WideBlock(nn.Module):
    """Pre-norm transformer block whose residual stream stays WIDE (w = 3*n_s*d): MLA
    mixes over the R time axis through low-rank latents, and the FFN runs at full width.
    """

    def __init__(self, w, num_heads, q_latent, kv_latent):
        super().__init__()
        self.n1 = nn.LayerNorm(w)
        self.attn = MLA(w, num_heads, q_latent, kv_latent)
        self.n2 = nn.LayerNorm(w)
        self.ffn = _mlp(w, w, 4 * w)             # wide FFN: w -> 4w -> w

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.ffn(self.n2(x))


class Fusion(nn.Module):
    """Flattened cross-modal fuse: tokens are [B, R, w] with w = 3*n_s*d. A learned
    positional embedding over the R time axis is added, then a stack of MLA WideBlocks
    attends over R (residual/FFN stay wide, attention runs through low-rank latents), then
    pool over R -> [B, w] and read out w -> d.

    Hiding a modality = zeroing its n_s*d slice of the wide token before the stack.
    """

    def __init__(self, d, n_s_total, d_out, R, depth=4, n_heads=8,
                 q_latent=192, kv_latent=64):
        super().__init__()
        w = n_s_total * d                        # 3*n_s*d
        self.pos = nn.Parameter(torch.randn(R, w) * 0.02)   # learned time posenc over R
        self.blocks = nn.ModuleList(
            [WideBlock(w, n_heads, q_latent, kv_latent) for _ in range(depth)])
        self.norm = nn.LayerNorm(w)
        self.readout = nn.Linear(w, d_out)       # w -> d (final latent)

    def forward(self, mods, hide=()):
        # mods: dict k -> [B, R, n_s_k, d]; concat along sensor axis, flatten to wide tokens.
        parts = [torch.zeros_like(mods[k]) if k in hide else mods[k] for k in MODALITIES]
        x = torch.cat(parts, dim=2)              # [B, R, n_s_total, d]
        B, R, n_tot, d = x.shape
        x = x.reshape(B, R, n_tot * d) + self.pos  # [B, R, w] + posenc over R
        for blk in self.blocks:
            x = blk(x)                           # [B, R, w]
        return self.readout(self.norm(x).mean(1))  # pool R -> [B, w] -> [B, d]


class MMPerceiverBTSP(nn.Module):
    """Cross-modal JEPA over [B, T, S, P] inputs. Objective identical to mm_perceiver3."""

    def __init__(self, d=256, n_s=8, R=8, spatial="perceiver", fuse_depth=4,
                 fuse_heads=8, fuse_q_latent=192, fuse_kv_latent=64,
                 stem_kernel=3, lamb=0.02, ema=0.99, n_slices=512):
        super().__init__()
        self.mods = nn.ModuleDict({
            "v": Modality(P_V, d, n_s, R, spatial, stem_kernel),
            "m": Modality(P_M, d, n_s, R, spatial, stem_kernel),
            "e": Modality(P_E, d, n_s, R, spatial, stem_kernel),
        })
        n_s_v, n_s_m, n_s_e = (self.mods[k].n_s for k in MODALITIES)
        n_s_total = n_s_v + n_s_m + n_s_e
        self.fusion = Fusion(d, n_s_total, d, R, fuse_depth, fuse_heads,
                             fuse_q_latent, fuse_kv_latent)

        # EMA-target stems (one per modality) for the cross-prediction targets.
        self.tgt = nn.ModuleDict({k: copy.deepcopy(self.mods[k].stem) for k in MODALITIES})
        for t in self.tgt.values():
            for p in t.parameters():
                p.requires_grad = False

        self.pred = nn.ModuleDict({k: _mlp(d, d) for k in MODALITIES})
        self.sigreg = SlicedEppsPulley(num_slices=n_slices)
        self.lamb, self.ema, self.R = lamb, ema, R

    @torch.no_grad()
    def update_target(self):
        for k in MODALITIES:
            for po, pt in zip(self.mods[k].stem.parameters(), self.tgt[k].parameters()):
                pt.mul_(self.ema).add_(po.detach(), alpha=1 - self.ema)

    @staticmethod
    def unpack(batch, dev):
        """Cache packet -> canonical [B, T, S, P] tensors + validity masks.

        rgb   [B,1,196,768]      already [B,T=1,S=196,P=768]
        motor [B,1,8,3] + mask   -> [B,1,8,6] (vals*mask ++ mask bits), t_valid all-True
        ee    [B,13,15] + [B,13]  -> [B,13,1,15], t_valid = ee_mask (mask over TIME)
        """
        to = lambda k: batch[k].to(dev, non_blocking=True)
        rgb = to("rgb").float()                              # [B,1,196,768]
        motor, m_mask = to("motor").float(), to("motor_mask")
        motor = motor.squeeze(1) if motor.dim() == 4 else motor   # [B,8,3]
        mfeat = torch.cat([motor * m_mask, m_mask.float()], -1)   # [B,8,6]
        mfeat = mfeat.unsqueeze(1)                           # [B,1,8,6]
        ee = to("ee").float().unsqueeze(2)                   # [B,13,1,15]
        e_mask = to("ee_mask")                               # [B,13] True=valid
        return rgb, mfeat, m_mask, ee, e_mask

    def _encode(self, rgb, mfeat, ee, e_mask):
        v_valid = torch.ones(rgb.shape[0], rgb.shape[1], dtype=torch.bool, device=rgb.device)
        m_valid = torch.ones(mfeat.shape[0], mfeat.shape[1], dtype=torch.bool, device=rgb.device)
        return {
            "v": self.mods["v"](rgb, v_valid),               # [B,R,n_s,d]
            "m": self.mods["m"](mfeat, m_valid),
            "e": self.mods["e"](ee, e_mask),                 # ee_mask over time
        }

    def _target(self, x, t_valid, k):
        """EMA-target embedding for one modality: target stem then pool over T and S."""
        with torch.no_grad():
            z = self.tgt[k](x)                               # [B,T,S,d]
            m = t_valid.unsqueeze(-1).unsqueeze(-1).float()  # [B,T,1,1]
            return (z * m).sum((1, 2)) / m.sum((1, 2)).clamp(min=1.0) / z.shape[2]

    def forward(self, rgb, mfeat, m_mask, ee, e_mask):
        e_any = e_mask.any(-1)                               # [B] samples with any valid ee
        v_valid = torch.ones(rgb.shape[:2], dtype=torch.bool, device=rgb.device)
        m_valid = torch.ones(mfeat.shape[:2], dtype=torch.bool, device=rgb.device)

        mods = self._encode(rgb, mfeat, ee, e_mask)

        tv = self._target(rgb, v_valid, "v")
        tm = self._target(mfeat, m_valid, "m")
        te = self._target(ee, e_mask, "e")

        z_no_v = self.fusion(mods, hide=("v",))
        z_no_m = self.fusion(mods, hide=("m",))
        z_no_e = self.fusion(mods, hide=("e",))

        inv = (self.pred["v"](z_no_v) - tv).square().mean() \
            + (self.pred["m"](z_no_m) - tm).square().mean()
        if bool(e_any.any()):
            inv = inv + (self.pred["e"](z_no_e[e_any]) - te[e_any]).square().mean()

        # online per-modal embeddings for SIGReg: pool the encoded latents over R and n_s.
        ev = mods["v"].mean((1, 2))
        em = mods["m"].mean((1, 2))
        z_full = self.fusion(mods)
        sig = self.sigreg(ev) + self.sigreg(em) + self.sigreg(z_full)
        if int(e_any.sum()) >= 8:
            sig = sig + self.sigreg(mods["e"].mean((1, 2))[e_any])
        loss = inv + self.lamb * sig
        return {"loss": loss, "inv": inv.detach(), "sig": sig.detach(),
                "z": z_full.detach()}

    @torch.no_grad()
    def embed_vision(self, rgb, mfeat, m_mask, ee, e_mask):
        """z_v: fused latent from VISION ONLY (motor + ee hidden) — the eval latent."""
        mods = self._encode(rgb, mfeat, ee, e_mask)
        return self.fusion(mods, hide=("m", "e"))
