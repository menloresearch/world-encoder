"""Multi-camera chunk cache (V0.2.md Build 1): like precompute_chunks.py but K EXTERNAL cameras per tick.

For each tick-anchored chunk, encode K external cameras' frames (nearest-timestamp matched — cams are NOT
filename-synced) with the frozen ViT-B/16 and stack. Same chunks/scenes/subsampling as the single-view cache
so the multi-cam gate is apples-to-apples vs single-view v0.1. Output caches/cfg<N>_mc.shard<S>.npz with:
  patch (N,K,196,768) fp16 | motor (N,1,8,3) | motor_mask (N,8,3) | ee (N,13,15) | ee_mask (N,13)
  | robot_id (N) | cfg (N) | scene (N) str | ts (N) int64 | cam_serials (N,K) str

Scene-sharded for the 8-GPU fan-out: --shard S --nshards M processes scenes[S::M]. Merge shards after
with merge_multicam.py (or np.concatenate the shard npzs).

    # one GPU per shard:
    for g in 0..7: CUDA_VISIBLE_DEVICES=$g python preprocessing/precompute_multicam.py \
        --cfgs 6 7 --n-cams 4 --shard $g --nshards 8 &
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_tokenizer.chunk_state import IN_HAND_OF_CFG, ROBOT_OF_CFG, SceneChunks  # noqa: E402


def external_cams(frames_scene_dir, wrist):
    """Sorted external camera dirs (exclude wrist serials, require a color/ subdir)."""
    out = []
    for c in sorted(glob.glob(os.path.join(frames_scene_dir, "cam_*"))):
        serial = os.path.basename(c).replace("cam_", "")
        if serial in wrist or not os.path.isdir(os.path.join(c, "color")):
            continue
        out.append(c)
    return out


def cam_ts_array(cam_dir):
    return np.array(sorted(int(f.split(".")[0]) for f in os.listdir(os.path.join(cam_dir, "color"))
                           if f.endswith(".jpg")), dtype=np.int64)


def nearest(arr, ts0, tol):
    j = int(np.searchsorted(arr, ts0))
    cand = [arr[k] for k in (j - 1, j) if 0 <= k < len(arr)]
    if not cand:
        return None
    best = min(cand, key=lambda t: abs(int(t) - int(ts0)))
    return int(best) if abs(int(best) - int(ts0)) <= tol else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfgs", type=int, nargs="+", default=[6, 7])
    ap.add_argument("--raw-root", default="/mnt/nas/data/RH20T/raw")
    ap.add_argument("--frames-root", default="/mnt/nas/data/RH20T/frames")
    ap.add_argument("--out-dir", default="/mnt/nas/data/RH20T/caches")
    ap.add_argument("--n-cams", type=int, default=4, help="K external cameras per tick")
    ap.add_argument("--tol-ms", type=int, default=100)
    ap.add_argument("--chunks-per-scene", type=int, default=15)
    ap.add_argument("--max-scenes", type=int, default=0, help="0 = all (debug cap)")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    import torch  # deferred
    from world_tokenizer.model import load_vitv2
    from world_tokenizer.precompute_patch import patch_embed

    os.makedirs(args.out_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_vitv2(pretrained=True).to(dev).eval()
    K = args.n_cams

    for n in args.cfgs:
        out = os.path.join(args.out_dir, f"cfg{n}_mc.shard{args.shard}of{args.nshards}.npz")
        if os.path.exists(out):
            print(f"cfg{n} shard{args.shard}: exists, skip", flush=True); continue
        raw = os.path.join(args.raw_root, f"RH20T_cfg{n}")
        scenes = sorted(d for d in os.listdir(raw) if d.startswith("task_") and "_human" not in d)
        if args.max_scenes:
            scenes = scenes[:: max(1, len(scenes) // args.max_scenes)][:args.max_scenes]
        scenes = scenes[args.shard::args.nshards]                       # scene shard for this GPU

        paths, motors, masks, ees, ee_masks, names, ticks, serials = [], [], [], [], [], [], [], []
        skipped = 0
        for si, s in enumerate(scenes, 1):
            try:
                sc = SceneChunks(os.path.join(raw, s), exclude=IN_HAND_OF_CFG.get(n, ()))
            except Exception:
                skipped += 1; continue
            if len(sc) == 0:
                skipped += 1; continue
            fsdir = os.path.join(args.frames_root, f"cfg{n}", s)
            cams = external_cams(fsdir, IN_HAND_OF_CFG.get(n, set()))
            if len(cams) < K:
                skipped += 1; continue
            cams = cams[:K]
            cts = {c: cam_ts_array(c) for c in cams}
            if any(len(v) < 2 for v in cts.values()):
                skipped += 1; continue
            stride = max(1, len(sc) // args.chunks_per_scene)
            for i in list(range(len(sc)))[::stride][:args.chunks_per_scene]:
                ts0 = int(sc.ticks[i])
                matched = [nearest(cts[c], ts0, args.tol_ms) for c in cams]
                if any(m is None for m in matched):
                    continue
                fps = [os.path.join(c, "color", f"{m}.jpg") for c, m in zip(cams, matched)]
                if not all(os.path.exists(fp) for fp in fps):
                    continue
                motor, mask, ee, ee_mask = sc.chunk(i)
                paths.extend(fps)                                       # K paths, cam-major within chunk
                motors.append(motor); masks.append(mask); ees.append(ee); ee_masks.append(ee_mask)
                names.append(s); ticks.append(ts0)
                serials.append([os.path.basename(c).replace("cam_", "") for c in cams])
            if si % 100 == 0:
                print(f"cfg{n} shard{args.shard}: [{si}/{len(scenes)}] {len(motors)} chunks", flush=True)

        if not motors:
            print(f"cfg{n} shard{args.shard}: no chunks", flush=True); continue
        N = len(motors)
        print(f"cfg{n} shard{args.shard}: embedding {len(paths)} frames ({N} chunks x {K} cams, "
              f"{skipped} scenes skipped)", flush=True)
        patch = patch_embed(model, paths, dev, bs=args.batch)           # (N*K,196,768) fp16
        patch = patch.reshape(N, K, 196, 768)
        np.savez(out, patch=patch,
                 motor=np.stack(motors), motor_mask=np.stack(masks),
                 ee=np.stack(ees), ee_mask=np.stack(ee_masks),
                 robot_id=np.full(N, ROBOT_OF_CFG[n], dtype=np.int64),
                 cfg=np.full(N, n, dtype=np.int64),
                 scene=np.array(names), ts=np.array(ticks, dtype=np.int64),
                 cam_serials=np.array(serials))
        print(f"cfg{n} shard{args.shard}: SAVED {N} chunks ({patch.nbytes/1e9:.1f} GB) -> {out}", flush=True)


if __name__ == "__main__":
    main()
