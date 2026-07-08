"""Analyze per-cfg differences across RH20T configs (read-only; regenerates DATA.md numbers).

For each cfg: robot specs from rh20t_api's configs.json, the transformed/*.npy value
shapes probed from sample scenes (joint vector length is the key cross-cfg hazard),
scene counts, preprocessing status (frames/ + shards/), and audio presence.

    python preprocessing/analyze_cfgs.py                    # markdown tables, 3 scenes/cfg
    python preprocessing/analyze_cfgs.py --full-scan        # + exhaustive missing-joint.npy sweep
"""
import argparse
import glob
import json
import os

import numpy as np


def default_conf():
    """configs.json ships at the root of the rh20t_api repo; fall back to the NAS copy."""
    try:
        import rh20t_api
        return os.path.join(os.path.dirname(os.path.dirname(rh20t_api.__file__)),
                            "configs", "configs.json")
    except ImportError:
        return "/mnt/nas/data/RH20T/deps/rh20t_api/configs/configs.json"


def robot_scenes(raw_dir):
    """Robot (non-human) scene dirs. NOTE: '_human' substring, not endswith — cfg1/2
    have `_human_2` variants that endswith() misses."""
    return sorted(d for d in os.listdir(raw_dir)
                  if d.startswith("task_") and "_human" not in d)


def probe_scene(scene_dir):
    """Shapes of the 4 transformed/*.npy files SceneState reads, from one scene."""
    T = os.path.join(scene_dir, "transformed")
    out = {"files": sorted(os.listdir(T)) if os.path.isdir(T) else []}
    try:
        joint = np.load(os.path.join(T, "joint.npy"), allow_pickle=True).item()
        serial = sorted(joint)[0]
        first = joint[serial][sorted(joint[serial])[0]]
        out["serials"] = len(joint)
        out["joint_len"] = len(np.atleast_1d(first))
    except Exception as e:
        out["joint_len"] = f"ERR {type(e).__name__}"
    try:
        tcp = np.load(os.path.join(T, "tcp_base.npy"), allow_pickle=True).item()
        entry = tcp[sorted(tcp)[0]][0]
        out["tcp_len"] = len(entry["tcp"])
    except Exception as e:
        out["tcp_len"] = f"ERR {type(e).__name__}"
    try:
        ft = np.load(os.path.join(T, "force_torque_base.npy"), allow_pickle=True).item()
        entry = ft[sorted(ft)[0]][0]
        out["ft_len"] = len(entry["zeroed"])
    except Exception as e:
        out["ft_len"] = f"ERR {type(e).__name__}"
    try:
        grip = np.load(os.path.join(T, "gripper.npy"), allow_pickle=True).item()
        s = sorted(grip)[0]
        out["gripper_cmd"] = "gripper_command" in grip[s][sorted(grip[s])[0]]
    except Exception as e:
        out["gripper_cmd"] = f"ERR {type(e).__name__}"
    out["audio"] = len(glob.glob(os.path.join(scene_dir, "audio_mixed", "*.wav")))
    out["cams"] = len(glob.glob(os.path.join(scene_dir, "cam_*")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default="/mnt/nas/data/RH20T/raw")
    ap.add_argument("--data-root", default="/mnt/nas/data/RH20T",
                    help="holds frames/cfgN and shards/cfgN")
    ap.add_argument("--conf", default=None, help="rh20t_api configs.json")
    ap.add_argument("--cfgs", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--scenes-per-cfg", type=int, default=3)
    ap.add_argument("--full-scan", action="store_true",
                    help="check every robot scene for a missing transformed/joint.npy")
    args = ap.parse_args()

    confs = {c["conf_num"]: c for c in json.load(open(args.conf or default_conf()))}

    print("| cfg | robot | joints | gripper | sensor | joint.npy len | tcp | ft | grip_cmd "
          "| cams | audio |")
    print("|-----|-------|--------|---------|--------|---------------|-----|----|---------"
          "|------|-------|")
    counts, missing = {}, {}
    for n in args.cfgs:
        c = confs.get(n, {})
        jf = c.get("robot_joint_field", "?")
        dof = jf[1] - jf[0] if isinstance(jf, list) else "?"
        raw_dir = os.path.join(args.raw_root, f"RH20T_cfg{n}")
        if not os.path.isdir(raw_dir):
            print(f"| {n} | (no raw dir) |")
            continue
        scenes = robot_scenes(raw_dir)
        step = max(1, len(scenes) // max(1, args.scenes_per_cfg))
        probes = [probe_scene(os.path.join(raw_dir, s)) for s in scenes[::step][:args.scenes_per_cfg]]
        agree = lambda k: "/".join(sorted({str(p.get(k)) for p in probes}))
        print(f"| {n} | {c.get('robot', '?')} | {dof} | {c.get('gripper', '?')} "
              f"| {c.get('sensor', '?')} | {agree('joint_len')} | {agree('tcp_len')} "
              f"| {agree('ft_len')} | {agree('gripper_cmd')} | {agree('cams')} | {agree('audio')} |")

        all_dirs = [d for d in os.listdir(raw_dir) if d.startswith("task_")]
        counts[n] = (len(all_dirs), len(scenes),
                     sum(1 for d in all_dirs if "_human" in d and not d.endswith("_human")))
        if args.full_scan:
            missing[n] = [s for s in scenes
                          if not os.path.exists(os.path.join(raw_dir, s, "transformed", "joint.npy"))]

    print("\n| cfg | scenes (all) | robot | _human_2 variants | frames dirs | shards | samples |")
    print("|-----|--------------|-------|-------------------|-------------|--------|---------|")
    for n in args.cfgs:
        if n not in counts:
            continue
        total, robot, h2 = counts[n]
        fdir = os.path.join(args.data_root, "frames", f"cfg{n}")
        nframes = len(os.listdir(fdir)) if os.path.isdir(fdir) else "-"
        sdir = os.path.join(args.data_root, "shards", f"cfg{n}")
        ntars = len(glob.glob(os.path.join(sdir, "*.tar")))
        cf = os.path.join(sdir, "count.txt")
        nsamp = open(cf).read().strip() if os.path.exists(cf) else "-"
        print(f"| {n} | {total} | {robot} | {h2} | {nframes} | {ntars} | {nsamp} |")

    if args.full_scan:
        print("\nRobot scenes missing transformed/joint.npy:")
        for n, ms in missing.items():
            print(f"  cfg{n}: {len(ms)}" + (f" -> {ms}" if ms else ""))


if __name__ == "__main__":
    main()
