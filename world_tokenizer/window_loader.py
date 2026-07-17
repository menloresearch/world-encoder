"""Windowing loader for the temporal (Phase-2) trainer.

Wraps the same per-cfg chunk caches as dataloader.py, but instead of one tick per sample it
yields a WINDOW of `window` consecutive same-scene chunks (ordered by ts), sliding with `stride`.
A window never straddles a ts gap > `gap_ms` (TEMPORAL_ARCH.md §7.1 windowing constraint) — on the
subsampled cache normal spacing is ~1.7 s, so gap_ms=5000 only splits genuine breaks.

Per window (stacked over C = window):
  rgb        [C, 196, 768] f32     motor      [C, 8, 3] f32     motor_mask [C, 8, 3] bool
  ee         [C, 13, 15]  f32      ee_mask    [C, 13]  bool
  t_ms       [C] f32   (tick ts minus window-start ts; the continuous-time input)
  robot_id/cfg/scene_idx/group_idx  int
Split is by the frozen (cfg,task,user) group CSV, same as dataloader.make_loader.

    tr, te, ds = make_window_loader("/mnt/nas/data/RH20T/caches", cfgs=[3], window=8, stride=4)
"""
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from world_tokenizer.dataloader import SPLIT_CSV, load_split, scene_group


class ChunkWindowDataset(Dataset):
    def __init__(self, cache_dir, cfgs=(1, 2, 3, 4, 5, 6, 7), window=8, stride=4, gap_ms=5000):
        keys = ["patch", "motor", "motor_mask", "ee", "ee_mask", "robot_id", "cfg", "ts"]
        parts = {k: [] for k in keys}
        names = []
        for n in cfgs:
            z = np.load(os.path.join(cache_dir, f"cfg{n}.npz"), allow_pickle=True)
            for k in keys:
                parts[k].append(z[k])
            names.append(z["scene"])
        self._d = {k: np.concatenate(v) for k, v in parts.items()}
        names = np.concatenate(names)
        self.window = window

        self.scenes = sorted(set(names.tolist()))
        self.groups = sorted({scene_group(s) for s in self.scenes})
        sidx = {s: i for i, s in enumerate(self.scenes)}
        gidx = {g: i for i, g in enumerate(self.groups)}
        self._scene_idx = np.array([sidx[s] for s in names], dtype=np.int64)
        self._group_idx = np.array([gidx[scene_group(s)] for s in names], dtype=np.int64)

        # build windows: per scene, sort by ts, split at gaps > gap_ms, slide window/stride
        ts = self._d["ts"]
        self.windows, self.win_group = [], []
        for s in self.scenes:
            si = sidx[s]
            idx = np.where(self._scene_idx == si)[0]
            idx = idx[np.argsort(ts[idx])]                       # ts order within scene
            if len(idx) < window:
                continue
            # segment boundaries where consecutive dt exceeds gap_ms
            dt = np.diff(ts[idx])
            brk = np.flatnonzero(dt > gap_ms) + 1
            for seg in np.split(idx, brk):
                for st in range(0, len(seg) - window + 1, stride):
                    w = seg[st:st + window]
                    self.windows.append(w)
                    self.win_group.append(self._group_idx[w[0]])
        self.win_group = np.array(self.win_group, dtype=np.int64)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        w = self.windows[i]
        d = self._d
        ts = d["ts"][w].astype(np.float64)
        return {
            "rgb": torch.from_numpy(d["patch"][w].astype(np.float32)),          # [C,196,768]
            "motor": torch.from_numpy(d["motor"][w]).squeeze(1),                # [C,8,3]
            "motor_mask": torch.from_numpy(d["motor_mask"][w]),                 # [C,8,3]
            "ee": torch.from_numpy(d["ee"][w]),                                 # [C,13,15]
            "ee_mask": torch.from_numpy(d["ee_mask"][w]),                       # [C,13]
            "t_ms": torch.from_numpy((ts - ts[0]).astype(np.float32)),          # [C]
            "robot_id": int(d["robot_id"][w[0]]),
            "cfg": int(d["cfg"][w[0]]),
            "scene_idx": int(self._scene_idx[w[0]]),
            "group_idx": int(self._group_idx[w[0]]),
        }


def make_window_loader(cache_dir, cfgs=(1, 2, 3, 4, 5, 6, 7), window=8, stride=4,
                       batch=64, split_csv=SPLIT_CSV, num_workers=4, gap_ms=5000):
    ds = ChunkWindowDataset(cache_dir, cfgs, window, stride, gap_ms)
    split = load_split(split_csv)
    unknown = [g for g in ds.groups if g not in split]
    assert not unknown, f"{len(unknown)} groups missing from split CSV: {unknown[:5]}"
    test_gidx = [i for i, g in enumerate(ds.groups) if split[g] == "test"]
    is_test = np.isin(ds.win_group, test_gidx)
    mk = lambda mask, shuf: DataLoader(Subset(ds, np.flatnonzero(mask).tolist()),
                                       batch_size=batch, shuffle=shuf,
                                       num_workers=num_workers, drop_last=shuf)
    return mk(~is_test, True), mk(is_test, False), ds


def unpack(batch, dev):
    return (batch["rgb"].to(dev, non_blocking=True),
            batch["motor"].to(dev, non_blocking=True),
            batch["motor_mask"].to(dev, non_blocking=True),
            batch["ee"].to(dev, non_blocking=True),
            batch["ee_mask"].to(dev, non_blocking=True),
            batch["t_ms"].to(dev, non_blocking=True))
