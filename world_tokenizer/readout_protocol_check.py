"""Protocol-recovery check for the 07-21 action-readout ladder (script was never committed).

The 07-21 session ladder (brain-internal#1): proprio 0.176 / random-fuse 0.287 / fused 0.335 /
raw-coarse 0.413. The 07-27 reproducible protocol (robocasa_pretrain_v03.gates) INVERTS the
raw-vs-fused ordering (0.500 vs 0.573). "Commanded 12-D x 5 sequence" is ambiguous — this runs
the plausible protocol variants under one harness to see which (if any) recovers the 07-21
ordering + magnitudes:

  A: 5 CONSECUTIVE frames (t..t+4, 0.25 s)      — the 07-27 protocol
  B: 5 STRIDED samples (t, t+5, .., t+20, 1.25 s)
  C: single action at t
  D: B without episode-held-out split (random 80/20 frame split — leaky, but plausible)

    python -m world_tokenizer.readout_protocol_check
"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_tokenizer.seese3_probe import Z_CACHE, FRAME_IDX, coarse_pool  # noqa: E402

DATASET = ("/mnt/nas/data/robocasa/datasets/v1.0/target/atomic/"
           "PickPlaceCounterToCabinet/20250811/lerobot")
PATCH_NPZ = "/mnt/nas/data/robocasa/kepler_cache/pnp_patches.npz"


def frame_actions(dataset):
    import glob
    import pandas as pd
    frames = {}
    for pq in sorted(glob.glob(os.path.join(dataset, "data", "chunk-*", "*.parquet"))):
        df = pd.read_parquet(pq, columns=["action", "episode_index"])
        frames[int(df["episode_index"].iloc[0])] = np.stack(
            df["action"].to_numpy()).astype(np.float32)
    return frames


def targets(frames, ep, t, offsets):
    ys, valid = [], []
    for e, tt in zip(ep, t):
        a = frames[int(e)]
        idx = tt + np.asarray(offsets)
        if idx.max() < len(a):
            ys.append(a[idx].reshape(-1))
            valid.append(True)
        else:
            ys.append(np.zeros(a.shape[1] * len(offsets), dtype=np.float32))
            valid.append(False)
    return np.stack(ys), np.array(valid)


def fit(rep, y, tr, te):
    m = RidgeCV(alphas=np.logspace(-2, 4, 7)).fit(rep[tr], y[tr])
    return float(r2_score(y[te], m.predict(rep[te]), multioutput="variance_weighted"))


def main():
    d = np.load(PATCH_NPZ)
    state, ep, t = d["state"].astype(np.float32), d["ep"], d["t"]
    fc = json.load(open(FRAME_IDX))
    zc_meta = json.load(open(f"{Z_CACHE}/meta.json"))
    zmm = np.memmap(f"{Z_CACHE}/z.f32", dtype=np.float32, mode="r",
                    shape=(fc["total_frames"], zc_meta["d"]))
    z = np.array(zmm[np.array([fc["episodes"][str(e)]["start"] for e in ep]) + t])
    coarse = coarse_pool(d["patch"])
    frames = frame_actions(DATASET)

    reps = {"proprio": state, "raw_coarse": coarse, "z_v02": z}
    protos = {"A_5consec": list(range(5)), "B_5strided": [0, 5, 10, 15, 20],
              "C_single": [0]}

    ep_split_tr = (ep % 10) != 0
    rng = np.random.default_rng(0)
    frame_split_tr = rng.random(len(ep)) < 0.8

    print(f"{'proto':>12} {'split':>7} | " + " ".join(f"{k:>10}" for k in reps))
    for pname, offs in protos.items():
        y, valid = targets(frames, ep, t, offs)
        for sname, tr_mask in [("episode", ep_split_tr), ("frame", frame_split_tr)]:
            tr = valid & tr_mask
            te = valid & ~tr_mask
            row = [fit(rep, y, tr, te) for rep in reps.values()]
            print(f"{pname:>12} {sname:>7} | " + " ".join(f"{v:>10.3f}" for v in row),
                  flush=True)


if __name__ == "__main__":
    main()
