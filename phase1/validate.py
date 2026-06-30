"""Phase 1 validation — "did the data work?" (no decoder yet).

Linear probe: freeze the backbone, extract CLS embeddings (768-d), fit a linear head to
predict the task id parsed from each frame's path. Beating chance => the latent carries
real signal => data + encoder are good. (Contact/no-contact probe needs F/T alignment via
RH20TScene.get_ft_aligned — left as a follow-up; see README.)

    python -m phase1.validate --frames-root <frames> --ckpt <phase1_ckpt.pt> --max-per-class 300
"""
import argparse
import glob
import os
import re

import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from phase1.model import LeJEPAVideo

_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
_TF = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), _NORM])
_TASK = re.compile(r"task_(\d+)")


def _task_id(path):
    m = _TASK.search(path)
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--ckpt", default=None, help="phase1 checkpoint; omit to probe the warm-start")
    ap.add_argument("--max-per-class", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # gather a balanced sample of frames per task
    paths, labels, per = [], [], {}
    for p in sorted(glob.glob(os.path.join(args.frames_root, "**", "*.jpg"), recursive=True)):
        t = _task_id(p)
        if t < 0 or per.get(t, 0) >= args.max_per_class:
            continue
        per[t] = per.get(t, 0) + 1
        paths.append(p)
        labels.append(t)
    n_classes = len(set(labels))
    assert n_classes >= 2, f"need >=2 tasks to probe, found {n_classes}"
    print(f"{len(paths)} frames over {n_classes} tasks | chance ~= {1 / n_classes:.3f}")

    net = LeJEPAVideo(pretrained=True).to(dev).eval()
    if args.ckpt:
        net.load_state_dict(torch.load(args.ckpt, map_location=dev)["model"])

    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), args.batch_size):
            imgs = torch.stack([_TF(Image.open(p).convert("RGB")) for p in paths[i : i + args.batch_size]])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                e = net.embed(imgs.to(dev))
            feats.append(e.float().cpu())
    X = torch.cat(feats).numpy()

    Xtr, Xte, ytr, yte = train_test_split(X, labels, test_size=0.3, random_state=0, stratify=labels)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
    acc = clf.score(Xte, yte)
    print(f"linear-probe task-id accuracy = {acc:.3f}  (chance = {1 / n_classes:.3f})")
    print("PASS" if acc > 2 / n_classes else "WEAK — inspect data/embeddings")


if __name__ == "__main__":
    main()
