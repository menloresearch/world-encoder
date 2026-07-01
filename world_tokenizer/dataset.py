"""Video-only multi-crop data for LeJEPA.

Two backends, same (gs, ls) batch format:
  * MultiCropRGB  — map-style over loose .jpg frames (debug / small runs)
  * make_wds_loader — WebDataset over tar shards (fast sequential NFS reads, full runs)

The checkpoint was trained DINOv2-style: 2 global 224 crops + N local 96 crops, ImageNet
norm. Set n_local=0 for the first runs (all views 224 -> no variable-resolution path).
"""
import glob
import os

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # in1k stats


def _global_tf(gsize=224):
    return T.Compose([
        T.RandomResizedCrop(gsize, scale=(0.4, 1.0)),
        T.RandomHorizontalFlip(), T.ColorJitter(0.4, 0.4, 0.2, 0.1), T.ToTensor(), _NORM,
    ])


def _local_tf(lsize=96):
    return T.Compose([
        T.RandomResizedCrop(lsize, scale=(0.05, 0.4)),
        T.RandomHorizontalFlip(), T.ColorJitter(0.4, 0.4, 0.2, 0.1), T.ToTensor(), _NORM,
    ])


class MultiCropRGB(Dataset):
    """Map-style: each RGB frame -> n_global global crops + n_local local crops."""

    def __init__(self, frames_root, n_global=2, n_local=6, gsize=224, lsize=96):
        self.paths = sorted(glob.glob(os.path.join(frames_root, "**", "*.jpg"), recursive=True))
        assert self.paths, f"no .jpg frames found under {frames_root}"
        self.n_global, self.n_local = n_global, n_local
        self.g, self.l = _global_tf(gsize), _local_tf(lsize)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return ([self.g(img) for _ in range(self.n_global)],
                [self.l(img) for _ in range(self.n_local)])


def collate(batch):
    """Group crops by type (global/local differ in size, so can't stack together)."""
    gs = torch.stack([torch.stack(g) for g, _ in batch])  # [B, n_global, 3, 224, 224]
    has_local = len(batch[0][1]) > 0
    ls = torch.stack([torch.stack(l) for _, l in batch]) if has_local else None
    return gs, ls


def split_views(gs, ls):
    """[B, V, C, H, W] -> list of V tensors [B, C, H, W] (what the model expects)."""
    global_views = [gs[:, i] for i in range(gs.shape[1])]
    local_views = [ls[:, i] for i in range(ls.shape[1])] if ls is not None else []
    return global_views, local_views


def make_wds_loader(shards, *, n_global=2, n_local=0, batch_size=64, num_workers=8,
                    seed=0, gsize=224, lsize=96, shuffle_buf=2000):
    """Infinite (resampled) WebDataset loader yielding (gs, ls) batches.

    resampled=True + per-rank `seed` is the DDP-safe pattern (no uneven-batch hang); the
    caller caps batches per epoch. workersplitter defaults to split_by_worker.
    """
    import webdataset as wds

    gtf, ltf = _global_tf(gsize), _local_tf(lsize)

    def _crop(sample):
        (img,) = sample
        return ([gtf(img) for _ in range(n_global)], [ltf(img) for _ in range(n_local)])

    ds = (
        wds.WebDataset(shards, resampled=True, shardshuffle=True, seed=seed,
                       nodesplitter=wds.split_by_node, empty_check=False)
        .shuffle(shuffle_buf)
        .decode("pil")
        .to_tuple("jpg")
        .map(_crop)
        .batched(batch_size, collation_fn=collate)
    )
    return wds.WebLoader(ds, batch_size=None, num_workers=num_workers,
                         pin_memory=True, persistent_workers=num_workers > 0)
