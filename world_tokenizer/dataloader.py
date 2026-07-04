"""Multi-cfg chunk dataloader: yields the stage-2 packet dict (spec in DATA.md).

Loads per-cfg caches written by preprocessing/precompute_chunks.py and serves:
  rgb        [B, 1, 196, 768] f32   frozen ViT patch tokens (chunk's frame)
  motor      [B, 1, 8, 3]     f32   7 joint rows + gripper row; C=[sin, cos, symlog dq]
  motor_mask [B, 8, 3]        bool
  ee         [B, 13, 15]      f32   100Hz F/T + TCP between ticks
  ee_mask    [B, 13]          bool
  robot_id   [B]              long  0=flexiv 1=ur5 2=franka 3=kuka
  cfg        [B]              long
  scene_idx  [B]              long  index into dataset.scenes
  group_idx  [B]              long  index into dataset.groups = (cfg, task, user)
  ts         [B]              long  tick timestamp (ms) — for Δt-based selection

Train/test split is HELD-OUT BY GROUP (cfg, task, user), stratified per cfg: the same
user repeating the same task up to 10x produces near-duplicate scenes, so scenes are
not independent — all repetitions land on one side of the split. Metrics
(metrics/METRICS.md) are computed on the test loader later; triplet negatives must be
drawn from test groups only.

    train, test, ds = make_loader("/mnt/nas/data/RH20T/caches", cfgs=[1,2,3,4,5,6,7])
"""
import os
import re

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


def scene_group(scene_name):
    """task_0001_user_0005_scene_0003_cfg_0001 -> task_0001_user_0005_cfg_0001."""
    return re.sub(r"_scene_\d+", "", scene_name)


class ChunkDataset(Dataset):
    def __init__(self, cache_dir, cfgs=(1, 2, 3, 4, 5, 6, 7)):
        parts = {k: [] for k in
                 ["patch", "motor", "motor_mask", "ee", "ee_mask", "robot_id", "cfg", "ts"]}
        names = []
        for n in cfgs:
            z = np.load(os.path.join(cache_dir, f"cfg{n}.npz"), allow_pickle=True)
            for k in parts:
                parts[k].append(z[k])
            names.append(z["scene"])
        self._d = {k: np.concatenate(v) for k, v in parts.items()}
        names = np.concatenate(names)

        self.scenes = sorted(set(names.tolist()))              # scene lookup
        self.groups = sorted({scene_group(s) for s in self.scenes})  # (cfg,task,user) lookup
        sidx = {s: i for i, s in enumerate(self.scenes)}
        gidx = {g: i for i, g in enumerate(self.groups)}
        self._scene_idx = np.array([sidx[s] for s in names], dtype=np.int64)
        self._group_idx = np.array([gidx[scene_group(s)] for s in names], dtype=np.int64)

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
            "group_idx": int(self._group_idx[i]),
            "ts": int(d["ts"][i]),
        }


def split_groups(groups, holdout_frac=0.3, seed=0):
    """Test group names, stratified per cfg (each cfg holds out ~holdout_frac of
    its (task, user) groups). Group names end in _cfg_NNNN."""
    rng = np.random.RandomState(seed)
    test = set()
    by_cfg = {}
    for g in groups:
        by_cfg.setdefault(g.rsplit("_cfg_", 1)[1], []).append(g)
    for cfg_groups in by_cfg.values():
        perm = rng.permutation(sorted(cfg_groups))
        test.update(perm[:max(1, round(len(perm) * holdout_frac))].tolist())
    return test


def make_loader(cache_dir, cfgs=(1, 2, 3, 4, 5, 6, 7), batch=256,
                holdout_frac=0.3, seed=0, num_workers=4):
    """Group-held-out (cfg, task, user) train/test DataLoaders (+ the dataset)."""
    ds = ChunkDataset(cache_dir, cfgs)
    test_groups = split_groups(ds.groups, holdout_frac, seed)
    test_gidx = {i for i, g in enumerate(ds.groups) if g in test_groups}
    is_test = np.isin(ds._group_idx, list(test_gidx))
    mk = lambda mask, shuf: DataLoader(Subset(ds, np.flatnonzero(mask).tolist()),
                                       batch_size=batch, shuffle=shuf,
                                       num_workers=num_workers, drop_last=shuf)
    return mk(~is_test, True), mk(is_test, False), ds
