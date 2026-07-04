"""Precompute stage-2 chunk caches: per-cfg npz of (ViT patch tokens + chunked state).

Supersedes world_tokenizer/precompute_patch.py for stage 2. For each cfg: sample
--chunks-per-scene tick-anchored chunks per robot scene (see chunk_state.py), embed
each chunk's frame with the frozen ViT-B/16, and write caches/cfg<N>.npz with:
  patch (N,196,768) fp16 | motor (N,1,8,3) | motor_mask (N,8,3) | ee (N,13,15)
  | ee_mask (N,13) | robot_id (N) | cfg (N) | scene (N) str | ts (N) int64 tick ms
Resumable per cfg (existing npz is skipped). Scenes missing state files are skipped;
chunks whose frame jpg is missing are skipped.

    python preprocessing/precompute_chunks.py --cfgs 1 2 3 4 5 6 7
    python preprocessing/precompute_chunks.py --cfgs 3 --max-scenes 5 --chunks-per-scene 2  # smoke
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_tokenizer.chunk_state import ROBOT_OF_CFG, SceneChunks  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfgs", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--raw-root", default="/mnt/nas/data/RH20T/raw")
    ap.add_argument("--frames-root", default="/mnt/nas/data/RH20T/frames")
    ap.add_argument("--out-dir", default="/mnt/nas/data/RH20T/caches")
    ap.add_argument("--chunks-per-scene", type=int, default=15)
    ap.add_argument("--max-scenes", type=int, default=0, help="0 = all (debug cap)")
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    import torch  # deferred: cheap --help without torch
    from world_tokenizer.model import load_vitv2
    from world_tokenizer.precompute_patch import patch_embed

    os.makedirs(args.out_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_vitv2(pretrained=True).to(dev).eval()

    for n in args.cfgs:
        out = os.path.join(args.out_dir, f"cfg{n}.npz")
        if os.path.exists(out):
            print(f"cfg{n}: {out} exists, skipping")
            continue
        raw = os.path.join(args.raw_root, f"RH20T_cfg{n}")
        scenes = sorted(d for d in os.listdir(raw)
                        if d.startswith("task_") and "_human" not in d)
        if args.max_scenes:
            scenes = scenes[:: max(1, len(scenes) // args.max_scenes)][:args.max_scenes]

        paths, motors, masks, ees, ee_masks, names, ticks = [], [], [], [], [], [], []
        skipped = 0
        for si, s in enumerate(scenes, 1):
            try:
                sc = SceneChunks(os.path.join(raw, s))
            except Exception:
                skipped += 1
                continue
            if len(sc) == 0:
                skipped += 1
                continue
            stride = max(1, len(sc) // args.chunks_per_scene)
            for i in list(range(len(sc)))[::stride][:args.chunks_per_scene]:
                fp = os.path.join(args.frames_root, f"cfg{n}", s,
                                  f"cam_{sc.serial}", "color", f"{sc.ticks[i]}.jpg")
                if not os.path.exists(fp):
                    continue
                motor, mask, ee, ee_mask = sc.chunk(i)
                paths.append(fp); motors.append(motor); masks.append(mask)
                ees.append(ee); ee_masks.append(ee_mask); names.append(s)
                ticks.append(int(sc.ticks[i]))
            if si % 200 == 0:
                print(f"cfg{n}: [{si}/{len(scenes)}] {len(paths)} chunks", flush=True)

        if not paths:
            print(f"cfg{n}: no chunks, skipping")
            continue
        print(f"cfg{n}: embedding {len(paths)} frames ({skipped} scenes skipped)", flush=True)
        patch = patch_embed(model, paths, dev, bs=args.batch)  # (N,196,768) fp16
        np.savez(out, patch=patch,
                 motor=np.stack(motors), motor_mask=np.stack(masks),
                 ee=np.stack(ees), ee_mask=np.stack(ee_masks),
                 robot_id=np.full(len(paths), ROBOT_OF_CFG[n], dtype=np.int64),
                 cfg=np.full(len(paths), n, dtype=np.int64),
                 scene=np.array(names), ts=np.array(ticks, dtype=np.int64))
        print(f"cfg{n}: SAVED {len(paths)} chunks -> {out}", flush=True)


if __name__ == "__main__":
    main()
