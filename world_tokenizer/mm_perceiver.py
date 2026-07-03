"""Stage 2 — Perceiver cross-modal JEPA (the real encoder).

M learnable queries cross-attend (stable_pretraining CrossAttention) to
[vision patch tokens (196) + state token (1)] -> M fused world tokens (the bottleneck).
Trained by cross-modal masked latent prediction: block one modality's context columns, predict
that modality's EMA-target embedding from the fused latent of the other. Losses: cross-modal
prediction (predict-don't-equate) + per-modal SIGReg + joint SIGReg on the fused latent.
"""
import copy

import torch
import torch.nn as nn

from stable_pretraining.backbone.vit import CrossAttention
from stable_pretraining.methods.lejepa import SlicedEppsPulley


def _mlp(i, o, h=None):
    h = h or o
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class PerceiverFuse(nn.Module):
    """M queries cross-attend to context tokens (Perceiver-style) -> pooled fused latent."""

    def __init__(self, d, n_queries=8, n_heads=8, depth=2):
        super().__init__()
        self.q = nn.Parameter(torch.randn(n_queries, d) * 0.02)  # learnable queries [M, d]
        self.ca = nn.ModuleList([CrossAttention(d, d, n_heads) for _ in range(depth)])
        self.n1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(depth)])
        self.ffn = nn.ModuleList([_mlp(d, d, 4 * d) for _ in range(depth)])
        self.n2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(depth)])

    def forward(self, context, attn_mask=None):
        # context: [B, T, d] (T = n_patch + 1 = 197); attn_mask: [M, T] bool, True=blocked
        x = self.q.unsqueeze(0).expand(context.shape[0], -1, -1)  # [M, d] -> [B, M, d]
        for ca, n1, ffn, n2 in zip(self.ca, self.n1, self.ffn, self.n2):
            # cross-attn: queries [B, M, d] attend to context [B, T, d] -> [B, M, d]
            x = x + ca(n1(x), context, attn_mask=attn_mask)      # [B, M, d]
            x = x + ffn(n2(x))                                   # [B, M, d]
        return x.mean(1)  # pool over M queries -> [B, d]


class MMPerceiver(nn.Module):
    def __init__(self, d=256, vis_dim=768, state_dim=28, n_patch=196, n_queries=8,
                 lamb=0.02, ema=0.99, n_slices=512):
        super().__init__()
        self.proj_v = nn.Linear(vis_dim, d)                 # vision proj: 768 -> d
        self.proj_s = nn.Linear(state_dim, d)               # state proj:  28 -> d
        self.mod = nn.Parameter(torch.randn(2, d) * 0.02)   # modality embeddings [2, d] (vision, state)
        self.fuse = PerceiverFuse(d, n_queries)
        self.pred_s = _mlp(d, d)   # fused(vision) [B,d] -> predict state target [B,d]
        self.pred_v = _mlp(d, d)   # fused(state)  [B,d] -> predict vision target [B,d]
        self.tgt_v = copy.deepcopy(self.proj_v)
        self.tgt_s = copy.deepcopy(self.proj_s)
        for p in list(self.tgt_v.parameters()) + list(self.tgt_s.parameters()):
            p.requires_grad = False
        self.sigreg = SlicedEppsPulley(num_slices=n_slices)
        self.lamb, self.ema, self.n_patch, self.n_queries = lamb, ema, n_patch, n_queries

    @torch.no_grad()
    def update_target(self):
        for o, t in [(self.proj_v, self.tgt_v), (self.proj_s, self.tgt_s)]:
            for po, pt in zip(o.parameters(), t.parameters()):
                pt.mul_(self.ema).add_(po.detach(), alpha=1 - self.ema)

    def _context(self, patch, state):
        # patch: [B, 196, vis_dim=768]; state: [B, state_dim=28]
        vt = self.proj_v(patch) + self.mod[0]                    # [B,196,768] -> [B,196,d], + mod emb [d]
        st = (self.proj_s(state) + self.mod[1]).unsqueeze(1)     # [B,28] -> [B,d] -> [B,1,d]
        return torch.cat([vt, st], dim=1)                        # [B, 197, d]  (196 vision + 1 state)

    def _mask(self, block_state, device):
        """[n_queries, n_patch+1] = [M, 197] bool, True=blocked. block_state=True hides the state token."""
        m = torch.zeros(self.n_queries, self.n_patch + 1, dtype=torch.bool, device=device)  # [M, 197]
        if block_state:
            m[:, self.n_patch:] = True     # hide state col (last) -> fuse from vision only
        else:
            m[:, :self.n_patch] = True     # hide vision cols (first 196) -> fuse from state only
        return m

    def forward(self, patch, state):
        # patch: [B, 196, 768]; state: [B, 28]
        ctx = self._context(patch, state)                        # [B, 197, d]
        with torch.no_grad():
            tv = self.tgt_v(patch).mean(1)                       # [B,196,768] -> [B,196,d] -> pooled [B,d]
            ts = self.tgt_s(state)                               # [B,28] -> [B,d]
        z_from_v = self.fuse(ctx, self._mask(block_state=True, device=patch.device))   # state hidden  [B,d]
        z_from_s = self.fuse(ctx, self._mask(block_state=False, device=patch.device))  # vision hidden [B,d]
        # predictions [B,d] vs targets [B,d] -> MSE -> scalar
        inv = (self.pred_s(z_from_v) - ts).square().mean() + (self.pred_v(z_from_s) - tv).square().mean()
        ev = self.proj_v(patch).mean(1)                          # online vision emb, pooled [B,d]
        es = self.proj_s(state)                                  # online state emb [B,d]
        z_full = self.fuse(ctx)                                  # both modalities visible [B,d]
        sig = self.sigreg(ev) + self.sigreg(es) + self.sigreg(z_full)  # per-modal + joint SIGReg, scalar
        loss = inv + self.lamb * sig                             # scalar
        return {"loss": loss, "inv": inv.detach(), "sig": sig.detach(),
                "z": z_full.detach(), "ev": ev.detach(), "es": es.detach()}

    @torch.no_grad()
    def embed(self, patch, state):
        # patch: [B, 196, 768]; state: [B, 28] -> fused latent [B, d]
        return self.fuse(self._context(patch, state))
