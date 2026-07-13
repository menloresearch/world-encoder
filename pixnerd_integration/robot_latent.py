from functools import partial

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Normalize
from torchvision.transforms.functional import to_tensor

from src.data.dataset.imagenet import center_crop_fn


class RobotLatentDataset(torch.utils.data.Dataset):
    """(robot frame image, world-encoder z_v) pairs for latent-conditioned pixel decoding.

    Reads a manifest built by world_tokenizer/precompute_decode.py: `frames_txt` (one image
    path per line) + `zv_npy` [N, dim] (aligned). Returns (normalized image in [-1,1], z_v
    vector, metadata) — matching PixNerd's PixImageNet contract (pixel space, no VAE latent).
    """

    def __init__(self, frames_txt: str, zv_npy: str, resolution: int = 128):
        with open(frames_txt) as f:
            self.paths = [p for p in f.read().splitlines() if p]
        self.zv = np.load(zv_npy).astype(np.float32)
        assert len(self.paths) == len(self.zv), f"{len(self.paths)} paths vs {len(self.zv)} zv"
        self.transform = partial(center_crop_fn, image_size=resolution)
        self.normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        raw = to_tensor(self.transform(Image.open(self.paths[idx]).convert("RGB")))
        x = self.normalize(raw)                                # [-1,1], the image to diffuse
        y = torch.from_numpy(self.zv[idx])                     # [dim] condition, → LatentConditioner
        return x, y, {"raw_image": raw}
