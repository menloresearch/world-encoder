"""Linear-probe curve — did continued LeJEPA improve the representation, and where does it plateau?

Trains a linear classifier on the frozen 768-d CLS embedding to predict task id (66 classes),
evaluating the warm-start baseline + epoch checkpoints on the SAME sample.

IMPORTANT: split is BY SCENE, not by frame — entire scenes are held out for test, so the probe
measures generalization to unseen episodes (a random frame split leaks near-duplicate frames and
trivially scores ~1.0). Same scene split + same embeddings extraction for every checkpoint.

    python -m world_tokenizer.probe_curve --ckpts e0 e3 e6 e10
"""
import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.linear_model import LogisticRegression

from world_tokenizer.model import LeJEPAVideo

FRAMES = "/mnt/nas/data/RH20T/frames/cfg3"
CKPT = "/mnt/nas/data/RH20T/checkpoints/phase1_ckpt_{}.pt"
_NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
_TF = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), _NORM])
_TASK = re.compile(r"task_(\d+)")


def sample_frames(frames_root, per_scene, max_per_class):
    """Strided frames from each scene; track (path, task, scene) so we can split by scene."""
    per, rows = defaultdict(int), []
    for sc in sorted(d for d in os.listdir(frames_root) if d.startswith("task_")):
        t = int(_TASK.search(sc).group(1))
        if per[t] >= max_per_class:
            continue
        cams = sorted(glob.glob(os.path.join(frames_root, sc, "cam_*", "color")))
        if not cams:
            continue
        fs = sorted(os.listdir(cams[0]))
        stride = max(1, len(fs) // per_scene)
        for f in fs[::stride][:per_scene]:
            rows.append((os.path.join(cams[0], f), t, sc)); per[t] += 1
            if per[t] >= max_per_class:
                break
    return rows


def scene_split(rows, test_frac=0.3, seed=0):
    """Per task, assign whole scenes to train/test. Tasks with <2 scenes are dropped."""
    task_scenes = defaultdict(set)
    for _, t, s in rows:
        task_scenes[t].add(s)
    rng = np.random.RandomState(seed)
    train_s, test_s, dropped = set(), set(), 0
    for t, scs in task_scenes.items():
        scs = sorted(scs)
        if len(scs) < 2:
            dropped += 1
            continue
        rng.shuffle(scs)
        k = max(1, round(len(scs) * test_frac))
        test_s.update(scs[:k]); train_s.update(scs[k:])
    tr = [i for i, (_, _, s) in enumerate(rows) if s in train_s]
    te = [i for i, (_, _, s) in enumerate(rows) if s in test_s]
    return tr, te, dropped


@torch.no_grad()
def embed(net, paths, dev, bs=256):
    out = []
    for i in range(0, len(paths), bs):
        imgs = torch.stack([_TF(Image.open(p).convert("RGB")) for p in paths[i:i + bs]])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(net.embed(imgs.to(dev)).float().cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-root", default=FRAMES)
    ap.add_argument("--ckpts", nargs="+", default=["e0", "e3", "e6", "e10"])
    ap.add_argument("--per-scene", type=int, default=12)
    ap.add_argument("--max-per-class", type=int, default=300)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    rows = sample_frames(args.frames_root, args.per_scene, args.max_per_class)
    tr, te, dropped = scene_split(rows)
    paths = [r[0] for r in rows]
    y = np.array([r[1] for r in rows])
    n_classes = len(set(y[tr]))
    print(f"{len(rows)} frames | {n_classes} tasks (dropped {dropped} with <2 scenes) | "
          f"train={len(tr)} test={len(te)} frames | chance = {1 / n_classes:.3f}", flush=True)

    net = LeJEPAVideo(pretrained=True).to(dev).eval()
    print(f"{'ckpt':>6} | {'probe acc':>9} | vs chance")
    for c in args.ckpts:
        if c != "e0":
            net.load_state_dict(torch.load(CKPT.format(c), map_location=dev)["model"])
        X = embed(net, paths, dev)
        clf = LogisticRegression(max_iter=3000, C=1.0).fit(X[tr], y[tr])
        acc = clf.score(X[te], y[te])
        print(f"{c:>6} | {acc:9.3f} | {acc * n_classes:5.1f}x", flush=True)


if __name__ == "__main__":
    main()
