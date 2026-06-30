"""Video-only multi-crop Dataset for LeJEPA.

The OK-AI/lejepa-vitb16 checkpoint was trained DINOv2-style: 2 global 224 crops +
N local 96 crops, ImageNet normalization. We match that. Set ``n_local=0`` for the
first debug run (all views 224 -> no variable-resolution path through the ViT).
"""
import glob
import os

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

# ImageNet-1k stats — the checkpoint was pretrained with these.
_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class MultiCropRGB(Dataset):
    """Each RGB frame -> ``n_global`` global crops + ``n_local`` local crops."""

    def __init__(self, frames_root, n_global=2, n_local=6, gsize=224, lsize=96):
        self.paths = sorted(
            glob.glob(os.path.join(frames_root, "**", "*.jpg"), recursive=True)
        )
        assert self.paths, f"no .jpg frames found under {frames_root}"
        self.n_global, self.n_local = n_global, n_local
        self.g = T.Compose(
            [
                T.RandomResizedCrop(gsize, scale=(0.4, 1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.4, 0.4, 0.2, 0.1),
                T.ToTensor(),
                _NORM,
            ]
        )
        self.l = T.Compose(
            [
                T.RandomResizedCrop(lsize, scale=(0.05, 0.4)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.4, 0.4, 0.2, 0.1),
                T.ToTensor(),
                _NORM,
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        g = [self.g(img) for _ in range(self.n_global)]
        l = [self.l(img) for _ in range(self.n_local)]
        return g, l


def collate(batch):
    """Group crops by type (global/local differ in size, so can't stack together)."""
    gs = torch.stack([torch.stack(g) for g, _ in batch])  # [B, n_global, 3, 224, 224]
    has_local = len(batch[0][1]) > 0
    ls = (
        torch.stack([torch.stack(l) for _, l in batch]) if has_local else None
    )  # [B, n_local, 3, 96, 96] or None
    return gs, ls


def split_views(gs, ls):
    """[B, V, C, H, W] tensors -> list of V tensors [B, C, H, W] (what the model expects)."""
    global_views = [gs[:, i] for i in range(gs.shape[1])]
    local_views = [ls[:, i] for i in range(ls.shape[1])] if ls is not None else []
    return global_views, local_views
