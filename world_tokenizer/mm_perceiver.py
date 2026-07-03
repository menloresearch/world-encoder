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
        self.q = nn.Parameter(torch.randn(n_queries, d) * 0.02)
        self.ca = nn.ModuleList([CrossAttention(d, d, n_heads) for _ in range(depth)])
        self.n1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(depth)])
        self.ffn = nn.ModuleList([_mlp(d, d, 4 * d) for _ in range(depth)])
        self.n2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(depth)])

    def forward(self, context, attn_mask=None):
        x = self.q.unsqueeze(0).expand(context.shape[0], -1, -1)
        for ca, n1, ffn, n2 in zip(self.ca, self.n1, self.ffn, self.n2):
            x = x + ca(n1(x), context, attn_mask=attn_mask)
            x = x + ffn(n2(x))
        return x.mean(1)  # [B, d]


class MMPerceiver(nn.Module):
    def __init__(self, d=256, vis_dim=768, state_dim=28, n_patch=196, n_queries=8,
                 lamb=0.02, ema=0.99, n_slices=512):
        super().__init__()
        self.proj_v = nn.Linear(vis_dim, d)
        self.proj_s = nn.Linear(state_dim, d)
        self.mod = nn.Parameter(torch.randn(2, d) * 0.02)   # modality embeddings [vision, state]
        self.fuse = PerceiverFuse(d, n_queries)
        self.pred_s = _mlp(d, d)   # fused(vision) -> predict state target
        self.pred_v = _mlp(d, d)   # fused(state)  -> predict vision target
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
        vt = self.proj_v(patch) + self.mod[0]                    # [B,196,d]
        st = (self.proj_s(state) + self.mod[1]).unsqueeze(1)     # [B,1,d]
        return torch.cat([vt, st], dim=1)                        # [B,197,d]

    def _mask(self, block_state, device):
        """[n_queries, n_patch+1] bool, True=blocked. block_state=True hides the state token."""
        m = torch.zeros(self.n_queries, self.n_patch + 1, dtype=torch.bool, device=device)
        if block_state:
            m[:, self.n_patch:] = True     # hide state -> fuse from vision only
        else:
            m[:, :self.n_patch] = True     # hide vision -> fuse from state only
        return m

    def forward(self, patch, state):
        ctx = self._context(patch, state)
        with torch.no_grad():
            tv = self.tgt_v(patch).mean(1)                       # vision target (pooled) [B,d]
            ts = self.tgt_s(state)                               # state target [B,d]
        z_from_v = self.fuse(ctx, self._mask(block_state=True, device=patch.device))   # state hidden
        z_from_s = self.fuse(ctx, self._mask(block_state=False, device=patch.device))  # vision hidden
        inv = (self.pred_s(z_from_v) - ts).square().mean() + (self.pred_v(z_from_s) - tv).square().mean()
        ev = self.proj_v(patch).mean(1)
        es = self.proj_s(state)
        z_full = self.fuse(ctx)                                  # both modalities visible
        sig = self.sigreg(ev) + self.sigreg(es) + self.sigreg(z_full)  # per-modal + joint SIGReg
        loss = inv + self.lamb * sig
        return {"loss": loss, "inv": inv.detach(), "sig": sig.detach(),
                "z": z_full.detach(), "ev": ev.detach(), "es": es.detach()}

    @torch.no_grad()
    def embed(self, patch, state):
        return self.fuse(self._context(patch, state))           # fused latent [B,d]
