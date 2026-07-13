import numpy as np
import torch

from src.models.conditioner.base import BaseConditioner


class LatentConditioner(BaseConditioner):
    """Condition PixNerd on a precomputed world-encoder latent z_v.

    z_v is passed through as a single-token continuous embedding [B, 1, dim] — exactly the
    shape the t2i `y_embedder` (a Linear) expects. Unconditional = a learned/zero null token
    for classifier-free guidance. The dataset yields the z_v vector as the label `y`.
    """

    def __init__(self, dim: int = 256):
        super().__init__()
        self.dim = dim

    def _impl_condition(self, y, metadata: dict = {}):
        if isinstance(y, (list, tuple)):
            z = torch.stack([torch.as_tensor(v) for v in y])
        else:
            z = torch.as_tensor(y)
        z = z.float()
        if z.ndim == 1:
            z = z.unsqueeze(0)
        return z.view(z.shape[0], 1, self.dim).cuda()

    def _impl_uncondition(self, y, metadata: dict = None):
        b = len(y) if isinstance(y, (list, tuple)) else int(torch.as_tensor(y).shape[0])
        return torch.zeros(b, 1, self.dim, device="cuda")
