"""Video-only multi-crop data for LeJEPA.

Two backends, same (gs, ls) batch format:
  * MultiCropRGB  — map-style over loose .jpg frames (debug / small runs)
  * make_wds_loader — WebDataset over tar shards (fast sequential NFS reads, full runs)

The checkpoint was trained DINOv2-style: 2 global 224 crops + N local 96 crops, ImageNet
norm. Set n_local=0 for the first runs (all views 224 -> no variable-resolution path).
"""
import glob
import os
import re

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from world_tokenizer.chunk_state import IN_HAND_OF_CFG

_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # in1k stats


def _scene_group(name):
    """task_..._user_..._scene_..._cfg_... -> (cfg,task,user) group key (drop scene)."""
    return re.sub(r"_scene_\d+", "", name)


def _external_cam(scene_dir, cfg):
    """Sorted-first EXTERNAL camera dir (wrist/in-hand excluded) — matches the serial
    precompute_chunks/SceneChunks pick for the multimodal caches (one fixed external view)."""
    inhand = IN_HAND_OF_CFG.get(cfg, set())
    cams = sorted(d for d in os.listdir(scene_dir) if d.startswith("cam_"))
    for c in cams:
        if c[len("cam_"):] not in inhand:
            return c
    return cams[0] if cams else None


def split_frame_paths(frames_base, cfgs, split_map, want="train", per_scene=0):
    """External-cam .jpg paths for the WANTED split's (cfg,task,user) groups across cfgs.

    Holdout-aware: only scenes whose group is `want` in the frozen split map are kept, so a
    vision finetune trains on the SAME train groups the multimodal encoder used (never test).
    Wrist views are excluded (one fixed external view, mirroring the chunk-packet pipeline).
    per_scene>0 evenly strides at most that many frames per scene (bounds NFS small-file IO)."""
    paths = []
    for n in cfgs:
        root = os.path.join(frames_base, f"cfg{n}")
        if not os.path.isdir(root):
            continue
        for s in sorted(d for d in os.listdir(root) if d.startswith("task_")):
            if split_map.get(_scene_group(s)) != want:
                continue
            cam = _external_cam(os.path.join(root, s), n)
            if cam is None:
                continue
            fs = sorted(glob.glob(os.path.join(root, s, cam, "color", "*.jpg")))
            if per_scene and len(fs) > per_scene:
                fs = fs[:: max(1, len(fs) // per_scene)][:per_scene]
            paths.extend(fs)
    return paths


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

    def __init__(self, frames_root=None, n_global=2, n_local=6, gsize=224, lsize=96, paths=None):
        # `paths` (holdout-aware, from split_frame_paths) takes precedence over globbing a root.
        self.paths = paths if paths is not None else sorted(
            glob.glob(os.path.join(frames_root, "**", "*.jpg"), recursive=True))
        assert self.paths, f"no .jpg frames found ({'paths' if paths is not None else frames_root})"
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
