"""Warm-start ViTv2 backbone + LeJEPA-video head.

Pre-flight findings baked in:
  * ``AutoModel.from_pretrained(..., trust_remote_code=True)`` FAILS on this checkpoint
    (its modelling file uses absolute imports of sibling modules ``hf_src`` /
    ``configuration_vitv2`` that transformers' check_imports rejects). The working
    path is: snapshot_download -> sys.path -> direct import of ViTv2PretrainedModel.
  * The model returns a DICT; the CLS embedding is ``out["latent"]`` (768-d).
  * stable-pretraining's LeJEPA is timm-only (no custom-backbone arg), so we compose:
    our backbone + a projector + the library's SlicedEppsPulley, and reuse the exact
    LeJEPA loss via ``LeJEPA._compute_loss`` (don't reimplement the loss).
"""
import functools
import sys

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download

from stable_pretraining.methods import LeJEPA
from stable_pretraining.methods.lejepa import SlicedEppsPulley

CKPT = "OK-AI/lejepa-vitb16-pretrain-in1k"


@functools.lru_cache(maxsize=1)
def _vitv2_repo():
    """Download (cached) the checkpoint repo and put it on sys.path for direct import."""
    repo = snapshot_download(CKPT, allow_patterns=["*.py", "*.json", "*.safetensors"])
    if repo not in sys.path:
        sys.path.insert(0, repo)
    return repo


def load_vitv2(pretrained=True):
    repo = _vitv2_repo()
    from modelling_vitv2 import ViTv2PretrainedModel  # noqa: E402  (needs sys.path first)

    if pretrained:
        return ViTv2PretrainedModel.from_pretrained(repo)
    from configuration_vitv2 import ViTv2Config  # noqa: E402

    return ViTv2PretrainedModel(ViTv2Config.from_pretrained(repo))


class ViTv2Backbone(nn.Module):
    """Adapter: makes the HF ViTv2 behave like a timm backbone (returns CLS tensor [N, 768])."""

    embed_dim = 768

    def __init__(self, pretrained=True):
        super().__init__()
        self.model = load_vitv2(pretrained=pretrained)

    def forward(self, x):  # x: [N, 3, H, W]
        return self.model(x)["latent"]  # [N, 768]


def _projector(in_dim=768, out_dim=512):
    """3-layer BN+ReLU MLP, mirroring stable-pretraining's default LeJEPA projector shape."""
    return nn.Sequential(
        nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
        nn.Linear(512, 2048), nn.BatchNorm1d(2048), nn.ReLU(inplace=True),
        nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.ReLU(inplace=True),
        nn.Linear(2048, out_dim),
    )


class LeJEPAVideo(nn.Module):
    """Continue-LeJEPA on cfg3 video, warm-started from the ViTv2 checkpoint.

    Reuses the library's SIGReg (``SlicedEppsPulley``) and loss (``LeJEPA._compute_loss``).
    """

    def __init__(self, pretrained=True, proj_dim=512, n_slices=1024, lamb=0.02):
        super().__init__()
        self.backbone = ViTv2Backbone(pretrained=pretrained)
        self.projector = _projector(self.backbone.embed_dim, proj_dim)
        self.sigreg = SlicedEppsPulley(num_slices=n_slices)
        self.lamb = lamb

    def forward(self, global_views, local_views=None):
        local_views = local_views or []
        # Backbone runs per-size group (224 globals together, 96 locals together),
        # then features (both [., 768]) are concatenated.
        g = self.backbone(torch.cat(global_views))          # [n_g*B, 768]
        feats = g if not local_views else torch.cat([g, self.backbone(torch.cat(local_views))])
        proj = self.projector(feats)                         # [n_views*B, K]
        B = global_views[0].shape[0]
        n_views = len(global_views) + len(local_views)
        proj = proj.view(n_views, B, -1)                     # [V, B, K]  (view-major)
        loss, inv, sig = LeJEPA._compute_loss(proj, len(global_views), self.sigreg, self.lamb)
        # inv/sigreg are returned detached (logging only); loss keeps the graph.
        return {"loss": loss, "inv_loss": inv.detach(), "sigreg_loss": sig.detach(),
                "embedding": g.detach()}

    @torch.no_grad()
    def embed(self, images):
        """CLS embedding [N, 768] for probing / downstream use."""
        return self.backbone(images)
