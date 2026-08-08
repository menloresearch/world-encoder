"""The same 18-image probe on geometry-pretrained encoders: VGGT and VGGT-Omega.

Two inference modes per model:

  single   each photo encoded alone. This is the analogue of swapping a
           monocular VLA vision tower for a geometry-pretrained one, and the
           mode the blog's claim rests on.
  multi    all 18 photos encoded jointly as one scene. A reconstruction
           setting, not VLA training — camera identity becomes *more* explicit,
           which is expected.

Distances are reported separately for the camera token, the register tokens,
the patch grid and all tokens, and for each of those split into the frame
stream and the global-attention stream (VGGT's final token is the two
concatenated).

Both models need the external package on PYTHONPATH plus a checkpoint:

  # public VGGT-1B, weights pulled from HF
  .vggt-venv/bin/python probe_vggt.py --model vggt --img-dir /path/to/gridset

  # gated VGGT-Omega-1B-512, local source checkout + .pt
  PYTHONPATH=/tmp/vggt-omega .vggt-venv/bin/python probe_vggt.py \
      --model omega --checkpoint /path/to/vggt_omega_1b_512.pt \
      --img-dir /path/to/gridset
"""
import argparse
import glob
import itertools
import json
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance

HERE = os.path.dirname(os.path.abspath(__file__))

CAMS = list("ABCDEF")
LAYS = ["1", "2", "3"]
STEMS = [c + l for l in LAYS for c in CAMS]

OMEGA_CHECKPOINT_FILE = "vggt_omega_1b_512.pt"

MODELS = {
    "vggt": {
        "model_id": "facebook/VGGT-1B",
        "prefix": "vggt",
        "single_mode": "single_image_pad_518",
        "multi_mode": "all18_multiview_pad_518",
    },
    "omega": {
        "model_id": "facebook/VGGT-Omega",
        "prefix": "vggt_omega",
        "single_mode": "single_image_balanced_512",
        "multi_mode": "all18_multiview_balanced_512",
    },
}

# metric name -> (token slice, distance over flattened vector or aligned grid)
METRICS = []
for _stream in ("", "frame_", "global_"):
    METRICS += [
        (f"{_stream}camera_token", f"{_stream}camera", "flat"),
        (f"{_stream}register_tokens", f"{_stream}register", "grid"),
        (f"{_stream}patch_tokens_aligned", f"{_stream}patch", "grid"),
        (f"{_stream}all_tokens_aligned", f"{_stream}all", "grid"),
    ]


def cosine_grid(a, b):
    return float((1.0 - F.cosine_similarity(a, b, dim=-1)).mean())


def cosine_flat(a, b):
    return float(1.0 - F.cosine_similarity(a.flatten()[None],
                                           b.flatten()[None]).item())


def distances(fa, fb):
    return {name: (cosine_flat if kind == "flat" else cosine_grid)(fa[key], fb[key])
            for name, key, kind in METRICS}


def group_pairs():
    placement = [(c + a, c + b) for c in CAMS
                 for a, b in itertools.combinations(LAYS, 2)]
    camera = [(a + l, b + l) for l in LAYS
              for a, b in itertools.combinations(CAMS, 2)]
    all_pairs = list(itertools.combinations(STEMS, 2))
    both = [p for p in all_pairs if p not in placement and p not in camera]
    return all_pairs, {"placement": placement, "camera": camera, "both": both}


def summarize(rows, groups, metric):
    out = {}
    for name, pairs in groups.items():
        vals = [rows[f"{a}-{b}"][metric] for a, b in pairs]
        out[name] = {
            "mean": float(np.mean(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "n": len(vals),
        }
    out["ratio_cam_over_place"] = (
        out["camera"]["mean"] / max(out["placement"]["mean"], 1e-12)
    )
    return out


def build_rows(feats):
    all_pairs, groups = group_pairs()
    rows = {f"{a}-{b}": distances(feats[a], feats[b]) for a, b in all_pairs}
    groups_out = {name: summarize(rows, groups, name) for name, _, _ in METRICS}
    return rows, groups_out


def print_summary(groups_out, bright=None):
    for metric, g in groups_out.items():
        print(f"\n== {metric} ==")
        if bright is not None:
            print(f"  brightness anchor {bright[metric]:.4f}")
        print(f"  placement {g['placement']['mean']:.4f} "
              f"[{g['placement']['min']:.4f}..{g['placement']['max']:.4f}]")
        print(f"  camera    {g['camera']['mean']:.4f} "
              f"[{g['camera']['min']:.4f}..{g['camera']['max']:.4f}]")
        print(f"  both      {g['both']['mean']:.4f} "
              f"[{g['both']['min']:.4f}..{g['both']['max']:.4f}]")
        print(f"  camera/placement {g['ratio_cam_over_place']:.2f}")


def load_paths(img_dir):
    files = {os.path.basename(f)[:2]: f
             for f in glob.glob(os.path.join(img_dir, "*.JPG"))}
    missing = sorted(set(STEMS) - set(files))
    if missing:
        raise FileNotFoundError(
            f"missing grid images {missing} in {img_dir} (see README.md)")
    return {s: files[s] for s in STEMS}


def make_brightness_anchor(src_path, out_path):
    im = Image.open(src_path).convert("RGB")
    ImageEnhance.Brightness(im).enhance(0.9).save(out_path)
    return out_path


def split_tokens(z, patch_start):
    """z: [tokens, 2C] for one frame. VGGT concatenates a frame stream and a
    global-attention stream, so the second half is the global stream."""
    dim = z.shape[-1]
    if dim % 2:
        raise ValueError(f"expected concatenated frame/global features, got dim={dim}")
    half = dim // 2
    slices = {"camera": z[:1], "register": z[1:patch_start],
              "patch": z[patch_start:], "all": z}
    out = dict(slices)
    for name, tok in slices.items():
        out[f"frame_{name}"] = tok[..., :half]
        out[f"global_{name}"] = tok[..., half:]
    return out


def resolve_omega_checkpoint(path):
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return path
    patterns = [
        os.path.join(os.path.expanduser("~/.cache/huggingface/hub"),
                     "models--facebook--VGGT-Omega", "snapshots", "*",
                     OMEGA_CHECKPOINT_FILE),
        os.path.join("/tmp", "*", ".cache/huggingface/hub",
                     "models--facebook--VGGT-Omega", "snapshots", "*",
                     OMEGA_CHECKPOINT_FILE),
    ]
    matches = [m for pat in patterns for m in glob.glob(pat)]
    if not matches:
        raise FileNotFoundError(
            f"could not find {OMEGA_CHECKPOINT_FILE}; pass --checkpoint explicitly")
    return sorted(matches)[-1]


def load_backend(kind, checkpoint, omega_repo, device):
    """Returns (model, preprocess(paths) -> tensor, checkpoint_name)."""
    if kind == "vggt":
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        model_id = MODELS["vggt"]["model_id"]
        print(f"loading {model_id} on {device}", flush=True)
        model = VGGT.from_pretrained(model_id).to(device).eval()
        return model, (lambda paths: load_and_preprocess_images(paths, mode="pad")), None

    if omega_repo:
        sys.path.insert(0, omega_repo)
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    ckpt = resolve_omega_checkpoint(checkpoint)
    print(f"loading {MODELS['omega']['model_id']} from {ckpt}", flush=True)
    model = VGGTOmega().eval()
    state = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    return (model,
            (lambda paths: load_and_preprocess_images(paths, image_resolution=512)),
            os.path.basename(ckpt))


@torch.inference_mode()
def encode(model, preprocess, paths, device, dtype):
    """Encode a list of images in one aggregator pass.
    Returns per-frame token dicts and the preprocessed image shape."""
    images = preprocess(paths).to(device)
    print(f"images {tuple(images.shape)}", flush=True)
    with torch.autocast(device_type=device, dtype=dtype):
        outs, patch_start = model.aggregator(images[None])
    z = outs[-1][0].float().cpu()  # [frames, tokens, 2C]
    return ([split_tokens(z[i], patch_start) for i in range(z.shape[0])],
            tuple(images.shape[-2:]))


def run_single(model, preprocess, paths, device, dtype, cfg, ckpt_name, out_dir):
    print("\n## single-image encoding", flush=True)
    feats, shapes = {}, {}
    for stem in STEMS:
        print(f"encoding {stem}", flush=True)
        toks, shape = encode(model, preprocess, [paths[stem]], device, dtype)
        feats[stem] = toks[0]
        shapes[stem] = shape

    with tempfile.TemporaryDirectory() as tmp:
        anchor = make_brightness_anchor(paths["A1"],
                                       os.path.join(tmp, "A1_bright.jpg"))
        toks, bright_shape = encode(model, preprocess, [anchor], device, dtype)
    bright = distances(feats["A1"], toks[0])

    rows, groups_out = build_rows(feats)
    print_summary(groups_out, bright)
    return {
        "model": cfg["model_id"],
        "checkpoint": ckpt_name,
        "mode": cfg["single_mode"],
        "image_shapes": shapes,
        "brightness_anchor_shape": bright_shape,
        "brightness_anchor": bright,
        "groups_by_metric": groups_out,
        "pairs": rows,
    }, os.path.join(out_dir, f"{cfg['prefix']}_single_image.json")


def run_multiview(model, preprocess, paths, device, dtype, cfg, ckpt_name, out_dir):
    print("\n## all-18-view inference", flush=True)
    toks, shape = encode(model, preprocess, [paths[s] for s in STEMS],
                         device, dtype)
    feats = {stem: toks[i] for i, stem in enumerate(STEMS)}
    rows, groups_out = build_rows(feats)
    print_summary(groups_out)
    return {
        "model": cfg["model_id"],
        "checkpoint": ckpt_name,
        "mode": cfg["multi_mode"],
        "image_shape": shape,
        "groups_by_metric": groups_out,
        "pairs": rows,
    }, os.path.join(out_dir, f"{cfg['prefix']}_all18_multiview.json")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=sorted(MODELS), default="vggt")
    p.add_argument("--img-dir", default=os.environ.get(
        "GRIDSET_DIR", os.path.join(HERE, "gridset")))
    p.add_argument("--results", default=os.path.join(HERE, "results"))
    p.add_argument("--checkpoint", default=None,
                   help="VGGT-Omega .pt (auto-discovered in HF cache if omitted)")
    p.add_argument("--omega-repo", default="/tmp/vggt-omega",
                   help="VGGT-Omega source checkout to put on sys.path")
    p.add_argument("--single-only", action="store_true")
    p.add_argument("--multi-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.single_only and args.multi_only:
        raise ValueError("choose at most one of --single-only or --multi-only")
    if not torch.cuda.is_available():
        raise RuntimeError("VGGT inference needs a CUDA GPU")
    device = "cuda"
    dtype = (torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8
             else torch.float16)

    cfg = MODELS[args.model]
    os.makedirs(args.results, exist_ok=True)
    paths = load_paths(args.img_dir)
    model, preprocess, ckpt_name = load_backend(
        args.model, args.checkpoint, args.omega_repo, device)

    runs = []
    if not args.multi_only:
        runs.append(run_single)
    if not args.single_only:
        runs.append(run_multiview)
    for run in runs:
        out, out_path = run(model, preprocess, paths, device, dtype, cfg,
                            ckpt_name, args.results)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
