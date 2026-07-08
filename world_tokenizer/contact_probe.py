"""Robot-relevant, UNSATURATED probe: predict CONTACT vs no-contact from the frozen CLS embedding.

The task-id probe is saturated by ImageNet features (baseline 0.91), so it can't tell whether
cfg3 finetuning added *robot-relevant* structure. Contact is a better test: a single RGB frame's
force state is NOT something ImageNet already encodes. Label = |force| > median (balanced, chance
0.5). Force from RH20TScene.get_ft_aligned (the getter that works). Scene-held-out split.

    python -m world_tokenizer.contact_probe --ckpts e0 e6 --ckpt-template /mnt/.../phase1_lr2e5_ckpt_{}.pt
"""
import argparse
import glob
import os
from collections import defaultdict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

import rh20t_api
from rh20t_api.configurations import load_conf
from rh20t_api.scene import RH20TScene
from world_tokenizer.model import LeJEPAVideo
from world_tokenizer.probe_curve import CKPT, FRAMES, embed

RAW = "/mnt/nas/data/RH20T/raw/RH20T_cfg3"
# configs.json ships at the root of the rh20t_api repo, next to the package dir
CONF = os.path.join(os.path.dirname(os.path.dirname(rh20t_api.__file__)), "configs", "configs.json")


def sample_with_force(raw_root, frames_root, confs, per_scene, max_scenes):
    """(path, scene, |force|) for strided frames across scenes; force via get_ft_aligned."""
    rows = []
    scenes = sorted(d for d in os.listdir(frames_root) if d.startswith("task_"))
    for sc in scenes[:max_scenes]:
        cams = sorted(glob.glob(os.path.join(frames_root, sc, "cam_*", "color")))
        if not cams:
            continue
        try:
            scene = RH20TScene(os.path.join(raw_root, sc), confs)
        except Exception:
            continue
        fs = sorted(os.listdir(cams[0]))
        stride = max(1, len(fs) // per_scene)
        for f in fs[::stride][:per_scene]:
            ts = int(f.split(".")[0])
            try:
                ft = np.asarray(scene.get_ft_aligned(ts, serial="base", zeroed=True), dtype=float)
                fmag = float(np.linalg.norm(ft[:3]))
            except Exception:
                continue
            if np.isnan(fmag):
                continue
            rows.append((os.path.join(cams[0], f), sc, fmag))
    return rows


def scene_split(scenes_of_rows, test_frac=0.3, seed=0):
    uniq = sorted(set(scenes_of_rows))
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    k = max(1, round(len(uniq) * test_frac))
    test = set(uniq[:k])
    tr = [i for i, s in enumerate(scenes_of_rows) if s not in test]
    te = [i for i, s in enumerate(scenes_of_rows) if s in test]
    return tr, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-root", default=FRAMES)
    ap.add_argument("--raw-root", default=RAW, help="dir of raw RH20T_cfg* scene folders")
    ap.add_argument("--conf", default=CONF, help="rh20t_api configs.json")
    ap.add_argument("--ckpts", nargs="+", default=["e0", "e6"])
    ap.add_argument("--ckpt-template", default=CKPT)
    ap.add_argument("--per-scene", type=int, default=20)
    ap.add_argument("--max-scenes", type=int, default=250)
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    confs = load_conf(args.conf)
    rows = sample_with_force(args.raw_root, args.frames_root, confs, args.per_scene, args.max_scenes)
    paths = [r[0] for r in rows]
    scenes = [r[1] for r in rows]
    fmag = np.array([r[2] for r in rows])
    thr = np.median(fmag)
    y = (fmag > thr).astype(int)  # balanced binary: contact vs no-contact
    tr, te = scene_split(scenes)
    print(f"{len(rows)} frames | contact thr(|F|)={thr:.2f}N | "
          f"train={len(tr)} test={len(te)} | pos-rate test={y[te].mean():.2f} | chance=0.50", flush=True)

    net = LeJEPAVideo(pretrained=True).to(dev).eval()
    print(f"{'ckpt':>5} | {'linear':>7} | {'kNN(20)':>8}")
    for c in args.ckpts:
        if c != "e0":
            net.load_state_dict(torch.load(args.ckpt_template.format(c), map_location=dev)["model"])
        X = embed(net, paths, dev)
        lin = LogisticRegression(max_iter=2000).fit(X[tr], y[tr]).score(X[te], y[te])
        knn = KNeighborsClassifier(n_neighbors=args.k).fit(X[tr], y[tr]).score(X[te], y[te])
        print(f"{c:>5} | {lin:7.3f} | {knn:8.3f}", flush=True)


if __name__ == "__main__":
    main()
