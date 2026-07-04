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
not independent — all repetitions land on one side of the split. The assignment is
FROZEN in splits/holdout_v1.csv (generated once by preprocessing/make_split.py from
the raw scene listing, committed) so the split is replicable regardless of which
chunks were cached. Metrics (metrics/METRICS.md) are computed on the test loader
later; triplet negatives must be drawn from test groups only.

    train, test, ds = make_loader("/mnt/nas/data/RH20T/caches", cfgs=[1,2,3,4,5,6,7])
"""
import csv
import os
import re

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

SPLIT_CSV = os.path.join(os.path.dirname(__file__), "..", "splits", "holdout_v1.csv")


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


def load_split(split_csv=SPLIT_CSV):
    """{group_name: "train"|"test"} from the frozen split CSV."""
    with open(split_csv, newline="") as f:
        return {r["group"]: r["split"] for r in csv.DictReader(f)}


def make_loader(cache_dir, cfgs=(1, 2, 3, 4, 5, 6, 7), batch=256,
                split_csv=SPLIT_CSV, num_workers=4):
    """Train/test DataLoaders (+ the dataset), split by the frozen group CSV.

    Every group found in the caches must appear in the CSV — an unknown group is an
    error (regenerate the CSV deliberately via preprocessing/make_split.py, don't
    let new data silently land on either side)."""
    ds = ChunkDataset(cache_dir, cfgs)
    split = load_split(split_csv)
    unknown = [g for g in ds.groups if g not in split]
    assert not unknown, f"{len(unknown)} groups missing from {split_csv}: {unknown[:5]}"
    test_gidx = [i for i, g in enumerate(ds.groups) if split[g] == "test"]
    is_test = np.isin(ds._group_idx, test_gidx)
    mk = lambda mask, shuf: DataLoader(Subset(ds, np.flatnonzero(mask).tolist()),
                                       batch_size=batch, shuffle=shuf,
                                       num_workers=num_workers, drop_last=shuf)
    return mk(~is_test, True), mk(is_test, False), ds
