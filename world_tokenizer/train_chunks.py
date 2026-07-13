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


def encode_zv(model, ds, dev, bs=512, mode="vision"):
    """z_v for every sample of a ChunkDataset, plus pooled raw vision features.
    mode='vision' -> vision-only latent (eval default); 'state' -> state-only latent
    (vision hidden, motor+ee only) for the cross-modal decode demo."""
    d = ds._d
    n = len(ds)
    embed = model.embed_state if mode == "state" else model.embed_vision
    zs, raws = [], []
    model.eval()
    for i in tqdm(range(0, n, bs), desc="encode", mininterval=10, leave=False):
        sl = slice(i, min(i + bs, n))
        rgb = torch.from_numpy(d["patch"][sl].astype(np.float32)).to(dev)
        motor = torch.from_numpy(d["motor"][sl]).squeeze(1).to(dev)
        m_mask = torch.from_numpy(d["motor_mask"][sl]).to(dev)
        ee = torch.from_numpy(d["ee"][sl]).to(dev)
        e_mask = torch.from_numpy(d["ee_mask"][sl]).to(dev)
        zs.append(embed(rgb, motor, m_mask, ee, e_mask).cpu().numpy())
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
    ap.add_argument("--train-cfgs", type=int, nargs="+", required=True)
    ap.add_argument("--tag", required=True, help="run name, e.g. ur5 / all")
    ap.add_argument("--out-dir", default="/mnt/nas/data/RH20T/checkpoints/phase1")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--queries", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--vision-only", action="store_true",
                    help="ablation: train on vision alone (no state fusion) — the cross-modal control")
    ap.add_argument("--no-joint-sigreg", action="store_true",
                    help="ablation: drop joint SIGReg on the fused latent")
    args = ap.parse_args()

    import time as _time
    for _ in range(15):                                   # VM CUDA-init is flaky under load
        if torch.cuda.is_available():
            break
        _time.sleep(10)
    if not torch.cuda.is_available():
        print("!! CUDA unavailable after retries — refusing to train on CPU", flush=True)
        sys.exit(2)
    dev = "cuda"
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
        model = MMPerceiverChunks(d=args.d, n_queries=args.queries,
                                  vision_only=args.vision_only,
                                  joint_sigreg=not args.no_joint_sigreg).to(dev)
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
