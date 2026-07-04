"""Multi-cfg chunk dataloader: yields the stage-2 packet dict (spec in DATA.md).

Loads per-cfg caches written by preprocessing/precompute_chunks.py and serves:
  rgb        [B, 1, 196, 768] f32   frozen ViT patch tokens (chunk's frame)
  motor      [B, 1, 8, 3]     f32   7 joint rows + gripper row; C=[sin, cos, symlog dq]
  motor_mask [B, 8, 3]        bool
  ee         [B, 13, 15]      f32   100Hz F/T + TCP between ticks
  ee_mask    [B, 13]          bool
  robot_id   [B]              long  0=flexiv 1=ur5 2=franka 3=kuka
  cfg        [B]              long
  scene_idx  [B]              long  index into dataset.scenes (for held-out splits)

    train, test, ds = make_loader("/mnt/nas/data/RH20T/caches", cfgs=[1,2,3,4,5,6,7])
"""
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


class ChunkDataset(Dataset):
    def __init__(self, cache_dir, cfgs=(1, 2, 3, 4, 5, 6, 7)):
        parts = {k: [] for k in
                 ["patch", "motor", "motor_mask", "ee", "ee_mask", "robot_id", "cfg"]}
        names = []
        for n in cfgs:
            z = np.load(os.path.join(cache_dir, f"cfg{n}.npz"), allow_pickle=True)
            for k in parts:
                parts[k].append(z[k])
            names.append(z["scene"])
        self._d = {k: np.concatenate(v) for k, v in parts.items()}
        names = np.concatenate(names)
        self.scenes = sorted(set(names.tolist()))  # scene-name lookup table
        idx = {s: i for i, s in enumerate(self.scenes)}
        self._scene_idx = np.array([idx[s] for s in names], dtype=np.int64)

    def __len__(self):
        return len(self._scene_idx)

    def __getitem__(self, i):
        d = self._d
        return {
            "rgb": torch.from_numpy(d["patch"][i].astype(np.float32)).unsqueeze(0),
            "motor": torch.from_numpy(d["motor"][i]),
            "motor_mask": torch.from_numpy(d["motor_mask"][i]),
            "ee": torch.from_numpy(d["ee"][i]),
            "ee_mask": torch.from_numpy(d["ee_mask"][i]),
            "robot_id": int(d["robot_id"][i]),
            "cfg": int(d["cfg"][i]),
            "scene_idx": int(self._scene_idx[i]),
        }


def make_loader(cache_dir, cfgs=(1, 2, 3, 4, 5, 6, 7), batch=256,
                holdout_frac=0.3, seed=0, num_workers=4):
    """Scene-held-out train/test DataLoaders (+ the underlying dataset)."""
    ds = ChunkDataset(cache_dir, cfgs)
    rng = np.random.RandomState(seed)
    scene_ids = rng.permutation(len(ds.scenes))
    test_scenes = set(scene_ids[:max(1, round(len(scene_ids) * holdout_frac))].tolist())
    is_test = np.isin(ds._scene_idx, list(test_scenes))
    mk = lambda mask, shuf: DataLoader(Subset(ds, np.flatnonzero(mask).tolist()),
                                       batch_size=batch, shuffle=shuf,
                                       num_workers=num_workers, drop_last=shuf)
    return mk(~is_test, True), mk(is_test, False), ds
