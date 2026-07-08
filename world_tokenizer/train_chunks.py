"""Phase-1 trainer: MMPerceiverChunks on the chunk caches (the matrix runs).

Train on --train-cfgs, then evaluate the frozen encoder's VISION-ONLY latent z_v on
every embodiment's held-out groups (probe fit on that embodiment's train groups,
never on test) -> one row of the 5x4 transfer matrix. Baselines per embodiment: raw
pooled ViT features (768) and PCA-256 of them — both fit on train rows only (the POC
fit PCA on all rows; fixed here). RankMe on the test z_v. Multi-seed.

Probe targets come from the packet itself: the 24 motor numbers (restricted to the
embodiment's globally-valid dims) and, where the embodiment has F/T data, the
15-dim mean of valid ee slots (R2 computed only over samples that have ee).

    python -m world_tokenizer.train_chunks --train-cfgs 3 4 --tag ur5 \
        --out-dir /mnt/nas/data/RH20T/checkpoints/phase1
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_tokenizer.dataloader import ChunkDataset, load_split, make_loader, scene_group  # noqa: E402
from world_tokenizer.mm_perceiver2 import MMPerceiverChunks, masked_mean  # noqa: E402

EMBODIMENTS = {"flexiv": (1, 2), "ur5": (3, 4), "franka": (5,), "kuka": (6, 7)}


def rankme(z):
    """Effective rank of latents z [N, d]; guarded for degenerate inputs."""
    z = np.asarray(z, dtype=np.float64)
    z = z - z.mean(0)
    s = np.linalg.svd(z, compute_uv=False)
    if s.sum() <= 0:
        return 1.0
    p = s / s.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def encode_zv(model, ds, dev, bs=512):
    """z_v for every sample of a ChunkDataset, plus pooled raw vision features."""
    d = ds._d
    n = len(ds)
    zs, raws = [], []
    model.eval()
    for i in tqdm(range(0, n, bs), desc="encode", mininterval=10, leave=False):
        sl = slice(i, min(i + bs, n))
        rgb = torch.from_numpy(d["patch"][sl].astype(np.float32)).to(dev)
        motor = torch.from_numpy(d["motor"][sl]).squeeze(1).to(dev)
        m_mask = torch.from_numpy(d["motor_mask"][sl]).to(dev)
        ee = torch.from_numpy(d["ee"][sl]).to(dev)
        e_mask = torch.from_numpy(d["ee_mask"][sl]).to(dev)
        zs.append(model.embed_vision(rgb, motor, m_mask, ee, e_mask).cpu().numpy())
        raws.append(rgb.mean(1).cpu().numpy())
    return np.concatenate(zs), np.concatenate(raws)


def probe_r2(x_tr, y_tr, x_te, y_te):
    """Standardized ridge probe R2 (uniform average over target dims)."""
    if len(x_tr) == 0 or len(x_te) == 0 or y_tr.shape[1] == 0:
        return float("nan")
    sx = StandardScaler().fit(x_tr)
    reg = Ridge(alpha=10.0).fit(sx.transform(x_tr), y_tr)
    return float(r2_score(y_te, reg.predict(sx.transform(x_te)),
                          multioutput="uniform_average"))


# ------------------------- vision-only finetune baseline eval -------------------------
# Re-encode each cache record's ORIGINAL frame through a LeJEPA-finetuned ViT backbone and
# probe its mean-pooled patch tokens (768) on the SAME held-out groups / targets as the
# multimodal eval. This drops the finetuned vision encoder in as a new row alongside the
# frozen raw ViT (768), PCA-256, and the multimodal z_v.

class _FrameDS(torch.utils.data.Dataset):
    """Frozen-eval transform = the exact preprocessing precompute_patch used (apples-to-apples
    with the cached `raw` features): Resize256 -> CenterCrop224 -> in1k norm."""
    def __init__(self, paths):
        import torchvision.transforms as T
        self.paths = paths
        self.tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                             T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


@torch.no_grad()
def encode_vision(embed_fn, paths, dev, bs=256, workers=12):
    """Vision features [N, d] via `embed_fn(imgs)->[B,d]`. Frame IO over NFS is the
    bottleneck, so decode/transform runs in `workers` DataLoader processes."""
    from torch.utils.data import DataLoader
    dl = DataLoader(_FrameDS(paths), batch_size=bs, num_workers=workers, pin_memory=True)
    out = []
    for imgs in tqdm(dl, desc="encode-vision", mininterval=10, leave=False):
        imgs = imgs.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(embed_fn(imgs).float().cpu().numpy())
    return np.concatenate(out)


def _record_frame_paths(ds, frames_base, idx):
    """External-cam frame path for the records in `idx`. The external cam (wrist excluded,
    sorted-first) read from the frames dir matches the SceneChunks serial precompute used
    (verified 0 mismatch across all 7 cfgs) — a cheap listdir vs 3 NFS npy reads/scene."""
    from world_tokenizer.dataset import _external_cam
    d = ds._d
    scene_of = np.array(ds.scenes)[ds._scene_idx]            # [N] scene name per record
    cam = {}
    def cam_of(scene, cfg):
        if scene not in cam:
            sd = os.path.join(frames_base, f"cfg{cfg}", scene)
            cam[scene] = _external_cam(sd, cfg) if os.path.isdir(sd) else None
        return cam[scene]
    paths = []
    for i in idx:
        cfg, s, ts = int(d["cfg"][i]), scene_of[i], int(d["ts"][i])
        c = cam_of(s, cfg)
        paths.append(None if c is None else
                     os.path.join(frames_base, f"cfg{cfg}", s, c, "color", f"{ts}.jpg"))
    return paths


def eval_vision_embodiment(embed_fn, cache_dir, cfgs, split, dev, frames_base,
                           bs=256, eval_max=0, workers=12):
    """Probe finetuned vision (ft, via `embed_fn`) vs frozen raw ViT (768) vs PCA-256 on one
    embodiment's held-out groups; probes fit on ITS train groups (same protocol as
    eval_embodiment). eval_max>0 strides the records (smoke / cheap runs; changes the rows)."""
    ds = ChunkDataset(cache_dir, cfgs)
    d = ds._d
    is_test = np.array([split[g] == "test" for g in
                        (ds.groups[i] for i in ds._group_idx)])

    sel = np.arange(len(ds))                                 # strided subsample (cheap evals)
    if eval_max and len(sel) > eval_max:
        sel = sel[:: max(1, len(sel) // eval_max)][:eval_max]
    paths = _record_frame_paths(ds, frames_base, sel)
    have = np.array([p is not None and os.path.exists(p) for p in paths])
    sel = sel[have]                                          # record indices we actually encode
    ft = encode_vision(embed_fn, [p for p, h in zip(paths, have) if h], dev, bs=bs, workers=workers)
    raw = d["patch"][sel].astype(np.float32).mean(1)         # frozen ViT, same records
    is_test = is_test[sel]

    motor = d["motor"].reshape(len(ds), -1)[sel]
    mvalid = d["motor_mask"].reshape(len(ds), -1).all(0)
    y_motor = motor[:, mvalid]
    e_any = d["ee_mask"].any(1)[sel]
    y_ee = masked_mean(torch.from_numpy(d["ee"]), torch.from_numpy(d["ee_mask"])).numpy()[sel]

    tr, te = ~is_test, is_test
    pca = PCA(n_components=min(256, raw.shape[1], tr.sum()), random_state=0).fit(raw[tr])
    p_tr, p_te = pca.transform(raw[tr]), pca.transform(raw[te])

    out = {"n_test": int(te.sum()), "n_drop": int((~have).sum()),
           "rankme_ft": rankme(ft[te]), "rankme_raw": rankme(raw[te])}
    for name, (xtr, xte) in {"ft": (ft[tr], ft[te]), "raw": (raw[tr], raw[te]),
                             "pca256": (p_tr, p_te)}.items():
        out[f"{name}_r2_motor"] = probe_r2(xtr, y_motor[tr], xte, y_motor[te])
        etr, ete = tr & e_any, te & e_any
        if etr.sum() > 50 and ete.sum() > 50:
            src = {"ft": ft, "raw": raw}.get(name)
            if src is None:
                out[f"{name}_r2_ee"] = probe_r2(pca.transform(raw[etr]), y_ee[etr],
                                                pca.transform(raw[ete]), y_ee[ete])
            else:
                out[f"{name}_r2_ee"] = probe_r2(src[etr], y_ee[etr], src[ete], y_ee[ete])
    return out


def _load_vision_embed(ckpt_path, dev):
    """Load a --vision-ckpt and return (embed_fn(imgs)->[B,d], d, kind). Auto-detects:
      * HEAD-ONLY (LeJEPAVisionHead): frozen ViT + proj_v -> proj_v(patch).mean(1)  [d=256]
      * FULL-FINETUNE (LeJEPAVideo): mean-pooled finetuned patch tokens              [768]"""
    from world_tokenizer.model import LeJEPAVideo, LeJEPAVisionHead
    ck = torch.load(ckpt_path, map_location=dev)
    sd = ck["model"] if "model" in ck else ck
    if any(k.endswith("proj_v.weight") or k == "proj_v.weight" for k in sd):
        d = sd[[k for k in sd if k.endswith("proj_v.weight")][0]].shape[0]
        net = LeJEPAVisionHead(d=d, pretrained=False).to(dev).eval()
        net.load_state_dict(sd)
        return (lambda imgs: net.embed(imgs)), d, "head"
    net = LeJEPAVideo(pretrained=False).to(dev).eval()
    net.load_state_dict(sd)
    bb = net.backbone
    return (lambda imgs: bb.model(imgs)["patch_latent"].mean(1)), 768, "full-finetune"


def run_vision_eval(args, dev):
    """--vision-ckpt path: probe a finetuned vision encoder (head or full) per embodiment."""
    ck = torch.load(args.vision_ckpt, map_location="cpu")
    embed_fn, ft_dim, kind = _load_vision_embed(args.vision_ckpt, dev)

    split = load_split()
    have = [n for n in range(1, 8)
            if os.path.exists(os.path.join(args.cache_dir, f"cfg{n}.npz"))]
    eval_embs = {k: tuple(c for c in v if c in have) for k, v in EMBODIMENTS.items()}
    eval_embs = {k: v for k, v in eval_embs.items()
                 if v and (not args.eval_embodiments or k in args.eval_embodiments)}
    run_dir = os.path.join(args.out_dir, args.tag)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[{args.tag}] VISION-CKPT {args.vision_ckpt} | {kind} ft_dim={ft_dim} "
          f"| eval {eval_embs} | dev {dev}", flush=True)

    results = {"vision_ckpt": args.vision_ckpt, "kind": kind, "ft_dim": ft_dim,
               "step": ck.get("step") if isinstance(ck, dict) else None, "embodiments": {}}
    for emb, cfgs in eval_embs.items():
        res = eval_vision_embodiment(embed_fn, args.cache_dir, cfgs, split, dev,
                                     args.frames_base, bs=args.enc_bs, eval_max=args.eval_max,
                                     workers=args.workers)
        results["embodiments"][emb] = res
        print(f"  [{emb}] " + " ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in res.items()), flush=True)
        with open(os.path.join(run_dir, "vision_eval.json"), "w") as f:
            json.dump(results, f, indent=1)
    print("VISION-EVAL DONE ->", os.path.join(run_dir, "vision_eval.json"), flush=True)


def eval_embodiment(model, cache_dir, cfgs, split, dev):
    """Probe z_v vs raw vs PCA-256 on one embodiment; probes fit on ITS train groups."""
    ds = ChunkDataset(cache_dir, cfgs)
    is_test = np.array([split[g] == "test" for g in
                        (ds.groups[i] for i in ds._group_idx)])
    zv, raw = encode_zv(model, ds, dev)

    d = ds._d
    motor = d["motor"].reshape(len(ds), -1)                     # [N,24]
    mvalid = d["motor_mask"].reshape(len(ds), -1).all(0)        # globally-valid dims
    y_motor = motor[:, mvalid]
    e_any = d["ee_mask"].any(1)
    ee_t = torch.from_numpy(d["ee"])
    y_ee = masked_mean(ee_t, torch.from_numpy(d["ee_mask"])).numpy()

    tr, te = ~is_test, is_test
    pca = PCA(n_components=min(256, raw.shape[1], tr.sum()), random_state=0).fit(raw[tr])
    p_tr, p_te = pca.transform(raw[tr]), pca.transform(raw[te])

    out = {"n_test": int(te.sum()), "rankme_zv": rankme(zv[te])}
    for name, (xtr, xte) in {"zv": (zv[tr], zv[te]), "raw": (raw[tr], raw[te]),
                             "pca256": (p_tr, p_te)}.items():
        out[f"{name}_r2_motor"] = probe_r2(xtr, y_motor[tr], xte, y_motor[te])
        etr, ete = tr & e_any, te & e_any
        if etr.sum() > 50 and ete.sum() > 50:
            xtr_e = {"zv": zv, "raw": raw}.get(name)
            if xtr_e is None:                                   # pca: transform subset
                out[f"{name}_r2_ee"] = probe_r2(pca.transform(raw[etr]), y_ee[etr],
                                                pca.transform(raw[ete]), y_ee[ete])
            else:
                out[f"{name}_r2_ee"] = probe_r2(xtr_e[etr], y_ee[etr],
                                                xtr_e[ete], y_ee[ete])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/mnt/nas/data/RH20T/caches")
    ap.add_argument("--train-cfgs", type=int, nargs="+",
                    help="cfgs to train the Perceiver on (required unless --vision-ckpt)")
    ap.add_argument("--tag", required=True, help="run name, e.g. ur5 / all")
    ap.add_argument("--out-dir", default="/mnt/nas/data/RH20T/checkpoints/phase1")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--queries", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    # vision-only finetune baseline eval (no MMPerceiver training when --vision-ckpt is set)
    ap.add_argument("--vision-ckpt", help="LeJEPA-finetuned ViT ckpt: probe its features instead "
                    "of training/evaluating the multimodal Perceiver")
    ap.add_argument("--frames-base", default="/mnt/nas/data/RH20T/frames")
    ap.add_argument("--eval-embodiments", nargs="+", default=None,
                    help="subset of {flexiv,ur5,franka,kuka} to eval (default all present)")
    ap.add_argument("--enc-bs", type=int, default=256, help="vision encode batch size")
    ap.add_argument("--eval-max", type=int, default=0,
                    help="cap records/embodiment (strided) for cheap/smoke vision eval; 0=all")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.vision_ckpt:
        run_vision_eval(args, dev)
        return
    assert args.train_cfgs, "--train-cfgs is required unless --vision-ckpt is given"
    run_dir = os.path.join(args.out_dir, args.tag)
    os.makedirs(run_dir, exist_ok=True)
    split = load_split()
    have = [n for n in range(1, 8)
            if os.path.exists(os.path.join(args.cache_dir, f"cfg{n}.npz"))]
    eval_embs = {k: tuple(c for c in v if c in have)
                 for k, v in EMBODIMENTS.items()}
    eval_embs = {k: v for k, v in eval_embs.items() if v}
    print(f"[{args.tag}] train cfgs {args.train_cfgs} | eval {eval_embs} | dev {dev}",
          flush=True)

    train_loader, _, _ = make_loader(args.cache_dir, tuple(args.train_cfgs),
                                     batch=args.batch, num_workers=args.workers)
    results = {"args": vars(args), "seeds": {}}
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = MMPerceiverChunks(d=args.d, n_queries=args.queries).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        t0 = time.time()
        for ep in tqdm(range(args.epochs), desc=f"seed {seed} train", mininterval=5):
            model.train()
            ep_loss, nb = 0.0, 0
            for batch in train_loader:
                out = model(*MMPerceiverChunks.unpack(batch, dev))
                opt.zero_grad()
                out["loss"].backward()
                opt.step()
                model.update_target()
                ep_loss += float(out["loss"]) ; nb += 1
            if ep % 10 == 0 or ep == args.epochs - 1:
                print(f"  seed {seed} ep {ep}: loss {ep_loss / max(nb, 1):.4f}", flush=True)

        torch.save(model.state_dict(), os.path.join(run_dir, f"seed{seed}.pt"))
        res = {}
        for emb, cfgs in eval_embs.items():
            res[emb] = eval_embodiment(model, args.cache_dir, cfgs, split, dev)
            print(f"  seed {seed} [{emb}] " + " ".join(
                f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in res[emb].items()), flush=True)
        res["train_minutes"] = (time.time() - t0) / 60
        results["seeds"][seed] = res
        with open(os.path.join(run_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=1)

    # mean+-std summary over seeds
    print(f"\n=== [{args.tag}] mean±std over {args.seeds} seeds ===", flush=True)
    for emb in eval_embs:
        for key in ("zv_r2_motor", "raw_r2_motor", "pca256_r2_motor",
                    "zv_r2_ee", "raw_r2_ee", "pca256_r2_ee", "rankme_zv"):
            vals = [results["seeds"][s][emb][key] for s in results["seeds"]
                    if key in results["seeds"][s][emb]]
            if vals:
                print(f"  {emb:8s} {key:16s} {np.mean(vals):.3f} ±{np.std(vals):.3f}",
                      flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
