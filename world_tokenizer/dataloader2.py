"""Multi-cfg chunk dataloader (VARIABLE chunk size / temporal variant).

This is the T-axis variant of the loader. For the single-timestep, backward-
compatible loader that the Phase-1 trainer/eval expect (motor_mask [B,8,3], no
`chunk_size`), use `dataloader.py`. This module adds a leading time axis T set by
`chunk_size` and, for symmetry with `motor`, gives `motor_mask` a T axis too
([B,T,8,3]) — so a consumer of this loader must squeeze/handle that T axis.

Loads per-cfg caches written by preprocessing/precompute_chunks.py and serves a
packet whose leading time axis T is set by `chunk_size` (see below):
  rgb        [B, T, 196, 768] f32   frozen ViT patch tokens, one frame per tick
  motor      [B, T, 8, 3]     f32   7 joint rows + gripper row; C=[sin, cos, symlog dq]
  motor_mask [B, T, 8, 3]     bool
  ee         [B, T*13, 15]    f32   100Hz F/T + TCP; per-tick windows concatenated
  ee_mask    [B, T*13]        bool
  robot_id   [B]              long  0=flexiv 1=ur5 2=franka 3=kuka
  cfg        [B]              long
  scene_idx  [B]              long  index into dataset.scenes (anchor tick's scene)
  group_idx  [B]              long  index into dataset.groups = (cfg, task, user)
  ts         [B]              long  anchor (first) tick timestamp (ms) — Δt-based selection

CHUNK SIZE. `chunk_size` is a duration in SECONDS, restricted to positive multiples of
0.1s (100ms) with a single decimal place — 0.1, 0.2, 0.3, ... are legal; 0.15, 0.25,
0, negatives are rejected. Native state ticks run at ~10Hz (~100ms apart), so one tick
≈ 0.1s and a chunk of `chunk_size` seconds is T = round(chunk_size/0.1) CONSECUTIVE
ticks stitched together at load time. chunk_size=0.1 -> T=1 is the single-timestep
default (leading axis of length 1, matching the old per-tick packet apart from
motor_mask, which now also carries the T axis for symmetry with motor).

Stitching is STRICT and never fabricates data: T ticks are stitched only when they are
the same scene AND temporally adjacent (0 < Δt <= MAX_STEP_MS ≈ one native tick). An
anchor tick without a full run of T contiguous ticks ahead of it is DROPPED — nothing
is zero-padded, so every emitted chunk is a real contiguous window and there is no
vision frame without a valid state behind it.

REQUIRES DENSE CACHES for T>1. precompute_chunks.py subsamples ticks per scene
(--chunks-per-scene, default 15): with more ticks than that in a scene the cached
records are strided ~seconds apart, so almost no two are 100ms-adjacent and T>1 will
yield an (near-)empty dataset. To use chunk_size>0.1, regenerate the caches with
--chunks-per-scene large enough that stride==1 (>= the max ticks/scene), so consecutive
cached records are consecutive native ticks. T=1 is unaffected.

Train/test split is HELD-OUT BY GROUP (cfg, task, user), stratified per cfg: the same
user repeating the same task up to 10x produces near-duplicate scenes, so scenes are
not independent — all repetitions land on one side of the split. The assignment is
FROZEN in splits/holdout_v1.csv (generated once by preprocessing/make_split.py from
the raw scene listing, committed) so the split is replicable regardless of which
chunks were cached. Metrics (metrics/METRICS.md) are computed on the test loader
later; triplet negatives must be drawn from test groups only.

    train, test, ds = make_loader("/mnt/nas/data/RH20T/caches", cfgs=[1,2,3,4,5,6,7])
    train, test, ds = make_loader(".../caches", chunk_size=0.3)  # 0.3s = 3-tick chunks
"""
import csv
import os
import re

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

SPLIT_CSV = os.path.join(os.path.dirname(__file__), "..", "splits", "holdout_v1.csv")

TICK_MS = 100      # native state-tick period (~10Hz); one tick ≈ 0.1s
MAX_STEP_MS = 150  # two ticks are "adjacent" iff 0 < Δt <= this (~1.5 native ticks);
                   # larger gaps mean a subsample stride or a recording gap -> don't stitch


def ticks_per_chunk(chunk_size):
    """Chunk duration (seconds) -> number of consecutive ticks T.

    chunk_size must be a positive multiple of 0.1s (100ms) with one decimal place:
    0.1->1, 0.2->2, 0.3->3, ... . Rejects 0.15/0.25 (finer than 100ms), 0, negatives."""
    steps = round(chunk_size * 10)
    if steps < 1 or abs(chunk_size * 10 - steps) > 1e-9:
        raise ValueError(
            f"chunk_size must be a positive multiple of 0.1s (100ms) with a single "
            f"decimal place, e.g. 0.1, 0.2, 0.3; got {chunk_size!r}")
    return int(steps)


def scene_group(scene_name):
    """task_0001_user_0005_scene_0003_cfg_0001 -> task_0001_user_0005_cfg_0001."""
    return re.sub(r"_scene_\d+", "", scene_name)


class ChunkDataset(Dataset):
    def __init__(self, cache_dir, cfgs=(1, 2, 3, 4, 5, 6, 7), chunk_size=0.1):
        self.n_ticks = ticks_per_chunk(chunk_size)   # T
        self.chunk_size = chunk_size
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

        # Stitch map: each row is the T record indices that form one chunk. Records are
        # stored per scene in ascending tick order, so record i+1 is tick i's temporal
        # successor within a scene. A valid anchor i needs T-1 consecutive adjacencies.
        self._members = self._build_members()   # [n_chunks, T]
        self.anchors = self._members[:, 0]       # anchor (first) record index per chunk

    def _build_members(self):
        n, T = len(self._scene_idx), self.n_ticks
        idx = np.arange(n)
        if T == 1:
            return idx[:, None]
        ts, sc = self._d["ts"].astype(np.int64), self._scene_idx
        dt = ts[1:] - ts[:-1]
        adj = (sc[1:] == sc[:-1]) & (dt > 0) & (dt <= MAX_STEP_MS)  # [n-1] i~i+1 adjacent
        w = T - 1
        if len(adj) < w:
            return np.empty((0, T), dtype=np.int64)
        # anchor i valid iff adj[i..i+w-1] all True; count False in each length-w window
        false_cs = np.concatenate([[0], np.cumsum(~adj)])          # [n]
        cap = n - w                                                # anchors in [0, cap)
        run_ok = (false_cs[np.arange(cap) + w] - false_cs[:cap]) == 0
        anchors = np.flatnonzero(run_ok)
        return anchors[:, None] + np.arange(T)[None, :]

    def __len__(self):
        return len(self._members)

    def __getitem__(self, i):
        d, T = self._d, self.n_ticks
        m = self._members[i]                       # [T] record indices, contiguous ticks
        a = int(m[0])                              # anchor record
        return {
            "rgb": torch.from_numpy(d["patch"][m].astype(np.float32)),        # [T,196,768]
            "motor": torch.from_numpy(d["motor"][m].reshape(T, *d["motor"].shape[2:])),  # [T,8,3]
            "motor_mask": torch.from_numpy(d["motor_mask"][m]),               # [T,8,3]
            "ee": torch.from_numpy(d["ee"][m].reshape(-1, d["ee"].shape[-1])),  # [T*13,15]
            "ee_mask": torch.from_numpy(d["ee_mask"][m].reshape(-1)),         # [T*13]
            "robot_id": int(d["robot_id"][a]),
            "cfg": int(d["cfg"][a]),
            "scene_idx": int(self._scene_idx[a]),
            "group_idx": int(self._group_idx[a]),
            "ts": int(d["ts"][a]),
        }


def load_split(split_csv=SPLIT_CSV):
    """{group_name: "train"|"test"} from the frozen split CSV."""
    with open(split_csv, newline="") as f:
        return {r["group"]: r["split"] for r in csv.DictReader(f)}


def make_loader(cache_dir, cfgs=(1, 2, 3, 4, 5, 6, 7), batch=256,
                split_csv=SPLIT_CSV, num_workers=4, chunk_size=0.1):
    """Train/test DataLoaders (+ the dataset), split by the frozen group CSV.

    chunk_size (seconds, multiple of 0.1 — see module docstring) sets the stitched
    chunk length T; T>1 needs dense caches (stride==1) or the loaders come up empty.

    Every group found in the caches must appear in the CSV — an unknown group is an
    error (regenerate the CSV deliberately via preprocessing/make_split.py, don't
    let new data silently land on either side)."""
    ds = ChunkDataset(cache_dir, cfgs, chunk_size=chunk_size)
    split = load_split(split_csv)
    unknown = [g for g in ds.groups if g not in split]
    assert not unknown, f"{len(unknown)} groups missing from {split_csv}: {unknown[:5]}"
    test_gidx = [i for i, g in enumerate(ds.groups) if split[g] == "test"]
    anchor_group = ds._group_idx[ds.anchors]       # per-chunk group, aligned to dataset idx
    is_test = np.isin(anchor_group, test_gidx)
    mk = lambda mask, shuf: DataLoader(Subset(ds, np.flatnonzero(mask).tolist()),
                                       batch_size=batch, shuffle=shuf,
                                       num_workers=num_workers, drop_last=shuf)
    return mk(~is_test, True), mk(is_test, False), ds
