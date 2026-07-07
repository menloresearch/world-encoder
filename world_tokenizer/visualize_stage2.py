"""Stage 2 visualization suite.

Creates qualitative and quantitative figures for the video+robot_state Perceiver result:

  * stage2_r2_bar.png                 - raw vision vs PCA control vs Perceiver z_v
  * embedding_pca_scatter.png         - PCA scatter colored by task and force/state
  * embedding_pca_scree.png           - explained variance curves
  * nearest_neighbors_raw_vs_zv.png   - photo nearest-neighbor comparison
  * photo_pca_map_<feature>.png       - image thumbnails laid out in PCA space
  * stage2_embeddings.npz             - features used by the figures

Run after the normal environment setup:

    source /mnt/nas/data/RH20T/env.sh
    python -m world_tokenizer.visualize_stage2 \
        --cache /dev/shm/wae_tmp/mm_patch.npz \
        --out /mnt/nas/data/RH20T/stage2_vis \
        --fit-perceiver --epochs 40

The existing cache may not contain frame paths. If so, pass --infer-paths and the script will
replay precompute_patch's sampling recipe, then verify that the scene sequence still matches.
"""
import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageOps  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from world_tokenizer.mm_perceiver import MMPerceiver  # noqa: E402
from world_tokenizer.state import FT_DIMS, SceneState  # noqa: E402

RAW = "/mnt/nas/data/RH20T/cfg3_raw/RH20T_cfg3"
FRAMES = "/mnt/nas/data/RH20T/cfg3_frames"
TASK = re.compile(r"task_(\d+)")
RESAMPLE = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)

STAGE2_R2 = {
    "raw vision\n768-d": (0.257, 0.075),
    "PCA vision\n256-d": (0.134, 0.047),
    "Perceiver z_v\n256-d": (0.551, 0.018),
}


def _str_array(x):
    return np.array([v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in x])


def task_ids(scene):
    out = []
    for s in scene:
        m = TASK.search(str(s))
        out.append(int(m.group(1)) if m else -1)
    return np.array(out)


def scene_masks(scene, seed, frac=0.3):
    uniq = sorted(set(scene.tolist()))
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    test = set(uniq[: max(1, round(len(uniq) * frac))])
    tr = np.array([s not in test for s in scene])
    te = np.array([s in test for s in scene])
    return tr, te


def rankme(Z):
    s = torch.linalg.svdvals(torch.as_tensor(np.asarray(Z), dtype=torch.float32))
    p = s / s.sum() + 1e-5
    return float(torch.exp(-(p * torch.log(p)).sum()))


def load_cache(path):
    dd = np.load(path, allow_pickle=True)
    out = {"patch": dd["patch"], "state": dd["state"], "scene": _str_array(dd["scene"])}
    if "path" in dd.files:
        out["path"] = _str_array(dd["path"])
    if "timestamp" in dd.files:
        out["timestamp"] = dd["timestamp"]
    return out


def infer_paths(scene, frames_root, raw_root, per_scene):
    paths, scenes, timestamps = [], [], []
    for sc in sorted(d for d in os.listdir(frames_root) if d.startswith("task_")):
        cams = sorted(glob.glob(os.path.join(frames_root, sc, "cam_*", "color")))
        if not cams:
            continue
        try:
            st = SceneState(os.path.join(raw_root, sc))
        except Exception:
            continue
        fs = sorted(os.listdir(cams[0]))
        stride = max(1, len(fs) // per_scene)
        for f in fs[::stride][:per_scene]:
            ts = int(f.split(".")[0])
            try:
                v = st.state(ts)
            except Exception:
                continue
            if np.isfinite(v).all():
                paths.append(os.path.join(cams[0], f))
                scenes.append(sc)
                timestamps.append(ts)

    scenes = np.array(scenes)
    if len(scenes) != len(scene) or not np.all(scenes == scene):
        print("WARNING: inferred paths do not match cache scene order; photo panels disabled.", flush=True)
        return None, None
    return np.array(paths), np.array(timestamps)


def standardize(X):
    return StandardScaler().fit_transform(np.asarray(X, dtype=np.float32))


def pca_coords(X, n=2):
    return PCA(n_components=n, random_state=0).fit_transform(standardize(X))


def sample_indices(n, max_points, seed):
    if n <= max_points:
        return np.arange(n)
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(n, max_points, replace=False))


def fit_perceiver(patch, state, scene, args, dev):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    trm, _ = scene_masks(scene, args.seed)
    tr_idx = np.where(trm)[0]

    net = MMPerceiver(d=args.d, n_queries=args.queries).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    for ep in range(args.epochs):
        net.train()
        perm = np.random.permutation(tr_idx)
        losses = []
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            P = torch.tensor(patch[b], device=dev).float()
            S = torch.tensor(state[b], device=dev).float()
            out = net(P, S)
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            opt.step()
            net.update_target()
            losses.append(float(out["loss"].detach().cpu()))
        if (ep + 1) % args.log_every == 0 or ep == 0 or ep + 1 == args.epochs:
            print(f"perceiver epoch {ep + 1}/{args.epochs} loss={np.mean(losses):.4f}", flush=True)
    return net


@torch.no_grad()
def embed_perceiver(net, patch, state, dev, batch=512):
    zv, zfull = [], []
    net.eval()
    for i in range(0, len(patch), batch):
        P = torch.tensor(patch[i:i + batch], device=dev).float()
        S = torch.tensor(state[i:i + batch], device=dev).float()
        ctx = net._context(P, S)
        zv.append(net.fuse(ctx, net._mask(block_state=True, device=dev)).cpu().numpy())
        zfull.append(net.fuse(ctx).cpu().numpy())
    return np.concatenate(zv), np.concatenate(zfull)


def plot_metric_bar(outdir):
    labels = list(STAGE2_R2)
    means = [STAGE2_R2[k][0] for k in labels]
    errs = [STAGE2_R2[k][1] for k in labels]
    colors = ["#64748b", "#94a3b8", "#0f766e"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, means, yerr=errs, capsize=5, color=colors, edgecolor="#1f2937", linewidth=0.8)
    ax.set_ylabel("R2 -> robot state, scene-held-out")
    ax.set_ylim(0, max(means) + max(errs) + 0.14)
    ax.set_title("Stage 2: cross-modal latent predicts robot state")
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025,
                f"{m:.3f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "stage2_r2_bar.png"), dpi=180)
    plt.close(fig)


def plot_scatter_panel(features, scene, state, outdir, max_points, seed):
    idx = sample_indices(len(scene), max_points, seed)
    tasks = task_ids(scene)[idx]
    force = np.linalg.norm(state[:, FT_DIMS], axis=1)[idx]
    grip = state[:, -1][idx]

    names = list(features)
    fig, axes = plt.subplots(len(names), 3, figsize=(13, 3.9 * len(names)), squeeze=False)
    for r, name in enumerate(names):
        coords = pca_coords(features[name][idx])
        for c, (color, label, cmap) in enumerate([
            (tasks, "task id", "tab20"),
            (force, "symlog F/T magnitude", "viridis"),
            (grip, "symlog gripper command", "magma"),
        ]):
            ax = axes[r, c]
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=color, s=8, alpha=0.75,
                            cmap=cmap, linewidths=0)
            ax.set_title(f"{name} PCA, colored by {label}")
            ax.set_xticks([])
            ax.set_yticks([])
            if c > 0:
                cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
                cb.ax.tick_params(labelsize=8)
        axes[r, 0].set_ylabel(name)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "embedding_pca_scatter.png"), dpi=180)
    plt.close(fig)


def plot_scree(features, outdir, max_components=32):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, X in features.items():
        n = min(max_components, X.shape[0], X.shape[1])
        pca = PCA(n_components=n, random_state=0).fit(standardize(X))
        ax.plot(np.arange(1, n + 1), np.cumsum(pca.explained_variance_ratio_), marker="o",
                markersize=3, label=f"{name} (RankMe {rankme(X):.0f})")
    ax.set_xlabel("PCA components")
    ax.set_ylabel("cumulative explained variance")
    ax.set_ylim(0, 1.02)
    ax.grid(color="#e5e7eb", linewidth=0.8)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Embedding variance profile")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "embedding_pca_scree.png"), dpi=180)
    plt.close(fig)


def open_thumb(path, size):
    im = Image.open(path).convert("RGB")
    im = ImageOps.contain(im, (size, size), RESAMPLE)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return canvas


def select_queries(state, n_queries):
    force = np.linalg.norm(state[:, FT_DIMS], axis=1)
    if len(force) <= n_queries:
        return np.arange(len(force))
    qs = np.linspace(0.05, 0.95, n_queries)
    return np.array([int(np.argmin(np.abs(force - np.quantile(force, q)))) for q in qs])


def nearest_indices(X, queries, scene, k, cross_scene):
    Xn = standardize(X)
    out = []
    for q in queries:
        d = ((Xn - Xn[q]) ** 2).sum(1)
        d[q] = np.inf
        if cross_scene:
            d[scene == scene[q]] = np.inf
        out.append(np.argsort(d)[:k])
    return out


def plot_neighbor_grid(paths, scene, state, features, outdir, k, n_queries, cross_scene):
    if "raw vision" not in features or "Perceiver z_v" not in features:
        return
    queries = select_queries(state, n_queries)
    raw_nn = nearest_indices(features["raw vision"], queries, scene, k, cross_scene)
    zv_nn = nearest_indices(features["Perceiver z_v"], queries, scene, k, cross_scene)

    thumb, gap, label_h = 128, 8, 24
    cols = 1 + k + k
    rows = len(queries)
    W = cols * thumb + (cols + 1) * gap
    H = rows * (thumb + label_h) + (rows + 1) * gap + 28
    canvas = Image.new("RGB", (W, H), "#f8fafc")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 6), "query | nearest in raw vision | nearest in Perceiver z_v", fill="#111827")

    for r, q in enumerate(queries):
        y = 28 + gap + r * (thumb + label_h + gap)
        row = [q] + list(raw_nn[r]) + list(zv_nn[r])
        for c, idx in enumerate(row):
            x = gap + c * (thumb + gap)
            try:
                im = open_thumb(paths[idx], thumb)
            except Exception:
                im = Image.new("RGB", (thumb, thumb), "#e5e7eb")
            canvas.paste(im, (x, y))
            border = "#f59e0b" if c == 0 else "#64748b" if c <= k else "#0f766e"
            draw.rectangle([x, y, x + thumb - 1, y + thumb - 1], outline=border, width=3)
            tag = "Q" if c == 0 else f"R{c}" if c <= k else f"Z{c - k}"
            draw.text((x + 4, y + thumb + 4), tag, fill="#111827")
    canvas.save(os.path.join(outdir, "nearest_neighbors_raw_vs_zv.png"))


def plot_photo_map(paths, coords, outpath, max_images, seed, title):
    rng = np.random.RandomState(seed)
    idx = sample_indices(len(paths), max_images, seed)
    rng.shuffle(idx)
    xy = coords[idx]
    xy = (xy - xy.min(0)) / (np.ptp(xy, axis=0) + 1e-6)

    W, H, thumb = 1600, 1100, 96
    canvas = Image.new("RGB", (W, H), "#f8fafc")
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 14), title, fill="#111827")
    used = set()
    for i, (u, v) in zip(idx, xy):
        x = int(30 + u * (W - thumb - 60))
        y = int(50 + (1 - v) * (H - thumb - 80))
        cell = (round(x / (thumb * 0.75)), round(y / (thumb * 0.75)))
        if cell in used:
            continue
        used.add(cell)
        try:
            im = open_thumb(paths[i], thumb)
        except Exception:
            continue
        canvas.paste(im, (x, y))
        draw.rectangle([x, y, x + thumb - 1, y + thumb - 1], outline="#334155", width=1)
    canvas.save(outpath)


def plot_photo_maps(paths, features, outdir, max_images, seed):
    for name, X in features.items():
        if name not in {"raw vision", "Perceiver z_v"}:
            continue
        coords = pca_coords(X)
        safe = name.lower().replace(" ", "_")
        plot_photo_map(paths, coords, os.path.join(outdir, f"photo_pca_map_{safe}.png"),
                       max_images, seed, f"{name}: photos laid out by PCA")


def save_embeddings(outdir, features, scene, state, paths=None, timestamp=None):
    payload = {k.lower().replace(" ", "_"): v for k, v in features.items()}
    payload["scene"] = scene
    payload["state"] = state
    if paths is not None:
        payload["path"] = paths
    if timestamp is not None:
        payload["timestamp"] = timestamp
    np.savez(os.path.join(outdir, "stage2_embeddings.npz"), **payload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/dev/shm/wae_tmp/mm_patch.npz")
    ap.add_argument("--out", default="/mnt/nas/data/RH20T/stage2_vis")
    ap.add_argument("--frames-root", default=FRAMES)
    ap.add_argument("--raw-root", default=RAW)
    ap.add_argument("--per-scene", type=int, default=15,
                    help="used only when inferring paths for old caches")
    ap.add_argument("--infer-paths", action="store_true")
    ap.add_argument("--fit-perceiver", action="store_true")
    ap.add_argument("--perceiver-ckpt", default=None)
    ap.add_argument("--save-perceiver", default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--queries", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--plot-max", type=int, default=4000)
    ap.add_argument("--sprite-max", type=int, default=180)
    ap.add_argument("--neighbors", type=int, default=5)
    ap.add_argument("--queries-per-grid", type=int, default=8)
    ap.add_argument("--same-scene-neighbors", action="store_true",
                    help="allow nearest neighbors from the same scene")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cache = load_cache(args.cache)
    patch = cache["patch"]
    state = cache["state"]
    scene = cache["scene"]
    vmean = patch.astype(np.float32).mean(1)

    paths = cache.get("path")
    timestamp = cache.get("timestamp")
    if paths is None and args.infer_paths:
        paths, timestamp = infer_paths(scene, args.frames_root, args.raw_root, args.per_scene)
    if paths is not None:
        paths = _str_array(paths)
        ok = np.array([os.path.exists(p) for p in paths])
        if not ok.all():
            print(f"WARNING: {len(ok) - int(ok.sum())} cached paths are not readable.", flush=True)

    print(f"cache: {len(scene)} frames | {len(set(scene.tolist()))} scenes | patch {patch.shape}", flush=True)
    features = {"raw vision": vmean}

    if args.perceiver_ckpt or args.fit_perceiver:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if args.perceiver_ckpt:
            net = MMPerceiver(d=args.d, n_queries=args.queries).to(dev)
            ckpt = torch.load(args.perceiver_ckpt, map_location=dev)
            net.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        else:
            net = fit_perceiver(patch, state, scene, args, dev)
        zv, zfull = embed_perceiver(net, patch, state, dev)
        features["Perceiver z_v"] = zv
        features["Perceiver full"] = zfull
        if args.save_perceiver:
            torch.save({"model": net.state_dict(), "args": vars(args)}, args.save_perceiver)

    save_embeddings(args.out, features, scene, state, paths, timestamp)
    plot_metric_bar(args.out)
    plot_scatter_panel(features, scene, state, args.out, args.plot_max, args.seed)
    plot_scree(features, args.out)
    if paths is not None:
        plot_neighbor_grid(paths, scene, state, features, args.out, args.neighbors,
                           args.queries_per_grid, cross_scene=not args.same_scene_neighbors)
        plot_photo_maps(paths, features, args.out, args.sprite_max, args.seed)
    else:
        print("No frame paths available; skipped photo grids. Rebuild cache or pass --infer-paths.", flush=True)

    print("saved visualizations ->", args.out, flush=True)


if __name__ == "__main__":
    main()
