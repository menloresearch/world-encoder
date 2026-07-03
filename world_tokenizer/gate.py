"""Phase 1 data gate — confirm force/torque <-> video alignment on one scene.

The contact force spike must line up with the video frame where the gripper makes
contact. Saves:
  gate_ft.png     — |force| & |torque| over the episode, contact peak marked
  gate_frames.png — frames around the peak (timestamped) to eyeball the contact

Run: python -m world_tokenizer.gate --scene task_0001_user_0016_scene_0001_cfg_0003
"""
import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from rh20t_api.configurations import load_conf  # noqa: E402
from rh20t_api.scene import RH20TScene  # noqa: E402

RAW = "/mnt/nas/data/RH20T/cfg3_raw/RH20T_cfg3"
FRAMES = "/mnt/nas/data/RH20T/cfg3_frames"
CONF = os.path.join(os.environ["WAE_ROOT"], "deps/rh20t_api/configs/configs.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="task_0001_user_0016_scene_0001_cfg_0003")
    ap.add_argument("--cam", default=None, help="camera serial; default = the one with most frames")
    ap.add_argument("--out", default=os.environ["WAE_ROOT"])
    args = ap.parse_args()

    confs = load_conf(CONF)
    scene = RH20TScene(os.path.join(RAW, args.scene), confs)

    fdir = os.path.join(FRAMES, args.scene)
    cams = {os.path.basename(c).replace("cam_", ""): c for c in glob.glob(os.path.join(fdir, "cam_*"))}
    cam = args.cam or max(cams, key=lambda c: len(os.listdir(os.path.join(cams[c], "color"))))
    cdir = os.path.join(cams[cam], "color")
    ts = sorted(int(os.path.splitext(f)[0]) for f in os.listdir(cdir))
    print(f"scene={args.scene} cam={cam} frames={len(ts)} "
          f"span={(ts[-1] - ts[0]) / 1000:.1f}s conf={scene._conf.conf_num} ({scene._conf.robot})")

    # sanity: getters return sane values at the mid timestamp
    mid = ts[len(ts) // 2]
    for name, fn in [("tcp", scene.get_tcp_aligned), ("joints", scene.get_joint_angles_aligned),
                     ("ft", scene.get_ft_aligned), ("gripper", scene.get_gripper)]:
        try:
            print(f"  {name}@mid:", np.round(np.atleast_1d(fn(mid)), 3))
        except Exception as e:
            print(f"  {name}@mid FAILED: {e!r}")

    try:
        print("  joints(raw)@mid:", np.round(scene.get_joints_angles(mid), 3))
    except Exception as e:
        print(f"  joints(raw) FAILED: {e!r}")

    # force/torque magnitude over the episode (tolerate frames outside the F/T range)
    def _ft(t):
        try:
            v = scene.get_ft_aligned(t, serial="base", zeroed=True)
            v = np.asarray(v, dtype=float)
            return v if v.shape == (6,) and not np.isnan(v).any() else np.full(6, np.nan)
        except Exception:
            return np.full(6, np.nan)

    F = np.array([_ft(t) for t in ts])
    valid = ~np.isnan(F).any(axis=1)
    print(f"F/T valid frames: {int(valid.sum())}/{len(ts)}")
    fmag, tmag = np.linalg.norm(F[:, :3], axis=1), np.linalg.norm(F[:, 3:], axis=1)
    peak = int(np.nanargmax(fmag))
    print(f"|force| range [{np.nanmin(fmag):.2f}, {np.nanmax(fmag):.2f}] | "
          f"peak @ frame {peak}, ts={ts[peak]} ({(ts[peak] - ts[0]) / 1000:.1f}s)")

    rel = (np.array(ts) - ts[0]) / 1000.0
    plt.figure(figsize=(9, 3))
    plt.plot(rel, fmag, label="|force|")
    plt.plot(rel, tmag, label="|torque|", alpha=0.6)
    plt.axvline(rel[peak], color="r", ls="--", label="contact peak")
    plt.xlabel("time (s)"); plt.ylabel("magnitude"); plt.legend()
    plt.title(f"{args.scene} | cam {cam} — F/T over episode")
    plt.tight_layout()
    p1 = os.path.join(args.out, "gate_ft.png"); plt.savefig(p1, dpi=90); print("saved", p1)

    idxs = sorted(set([0, max(0, peak - 8), max(0, peak - 4), peak,
                       min(len(ts) - 1, peak + 4), len(ts) - 1]))
    ims = [Image.open(os.path.join(cdir, f"{ts[i]}.jpg")) for i in idxs]
    w, h = ims[0].size
    canvas = Image.new("RGB", (w * len(ims), h + 18), (0, 0, 0))
    for j, (i, im) in enumerate(zip(idxs, ims)):
        canvas.paste(im, (j * w, 18))
        ImageDraw.Draw(canvas).text((j * w + 4, 2),
                                    f"{(ts[i] - ts[0]) / 1000:.1f}s{'  <PEAK>' if i == peak else ''}",
                                    fill=(255, 255, 0))
    p2 = os.path.join(args.out, "gate_frames.png"); canvas.save(p2); print("saved", p2)


if __name__ == "__main__":
    main()
