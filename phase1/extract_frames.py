"""RH20T mp4 -> timestamped jpg frames (color only; depth was not downloaded).

rh20t_api.extract has NO CLI (its __main__ is a hardcoded sample), so this is a thin
wrapper around its functions. Output layout: <dest>/<scene>/<cam_*>/color/<ts>.jpg

Examples:
    # one scene (debug — do this first):
    python -m phase1.extract_frames --scene task_0001_user_0016_scene_0001_cfg_0003
    # everything:
    python -m phase1.extract_frames --all --num-workers 16
"""
import argparse
import os
from multiprocessing import Pool

from rh20t_api.extract import convert_scene

DEF_RAW = "/mnt/nas/data/RH20T/cfg3_raw/RH20T_cfg3"
DEF_DEST = "/mnt/nas/data/RH20T/cfg3_frames"


def _is_scene(name, include_human=False):
    # scenes look like task_..._scene_..._cfg_0003 (and _human variants); skip calib/etc.
    if not (name.startswith("task_") and "scene_" in name):
        return False
    return include_human or not name.endswith("_human")


def _convert_one(raw_root, dest, scene):
    src = os.path.join(raw_root, scene)
    convert_scene(src, os.path.join(dest, scene), scene_depth_dir=None)
    return scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default=DEF_RAW)
    ap.add_argument("--dest", default=DEF_DEST)
    ap.add_argument("--scene", default=None, help="single scene folder to convert (debug)")
    ap.add_argument("--all", action="store_true", help="convert every scene under --raw-root")
    ap.add_argument("--include-human", action="store_true", help="also extract _human demo scenes")
    ap.add_argument("--num-workers", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    if args.scene:
        print("converting", args.scene)
        _convert_one(args.raw_root, args.dest, args.scene)
        print("done ->", os.path.join(args.dest, args.scene))
        return

    if not args.all:
        ap.error("pass --scene <name> for one scene, or --all for everything")

    scenes = sorted(s for s in os.listdir(args.raw_root) if _is_scene(s, args.include_human))
    print(f"{len(scenes)} scenes -> {args.dest} (workers={args.num_workers})")
    with Pool(args.num_workers) as pool:
        for i, s in enumerate(
            pool.starmap(_convert_one, [(args.raw_root, args.dest, s) for s in scenes])
        ):
            print(f"[{i + 1}/{len(scenes)}] {s}")


if __name__ == "__main__":
    main()
