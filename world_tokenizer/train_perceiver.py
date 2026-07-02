"""Stage 2 — train + eval the Perceiver cross-modal JEPA (multi-seed, with attribution baseline).

Key eval: the VISION-ONLY fused latent (state masked) -> state R². Compare to:
  raw vision(768)        -> state  (no fusion, no training)
  PCA-256(vision)        -> state  (compression-only control: isolates cross-modal from dim-reduction)
If z_v beats BOTH consistently, the gain is genuinely cross-modal, not just compression.
Encoder retrained fresh per split (no leakage). RankMe for collapse. Scene-held-out.

    python -m world_tokenizer.train_perceiver --cache /dev/shm/wae_tmp/mm_patch.npz --seeds 5
"""
import argparse

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from world_tokenizer.mm_perceiver import MMPerceiver
from world_tokenizer.state import FT_DIMS


def rankme(Z):
    s = torch.linalg.svdvals(torch.as_tensor(np.asarray(Z), dtype=torch.float32))
    p = s / s.sum() + 1e-5
    return float(torch.exp(-(p * torch.log(p)).sum()))


def scene_masks(scene, seed, frac=0.3):
    u = sorted(set(scene.tolist()))
    rng = np.random.RandomState(seed)
    rng.shuffle(u)
    te = set(u[: max(1, round(len(u) * frac))])
    return (np.array([s not in te for s in scene]), np.array([s in te for s in scene]))


def probe(X, Y, trm, tem):
    sc = StandardScaler().fit(X[trm])
    m = MLPRegressor(hidden_layer_sizes=(256,), alpha=1e-3, early_stopping=True, max_iter=300)
    m.fit(sc.transform(X[trm]), Y[trm])
    return r2_score(Y[tem], m.predict(sc.transform(X[tem])))


def run_seed(patch, state, kin, vmean, scene, seed, args, dev):
    trm, tem = scene_masks(scene, seed)
    tr_idx = np.where(trm)[0]
    net = MMPerceiver(d=args.d, n_queries=args.queries).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    for _ in range(args.epochs):
        net.train()
        perm = np.random.permutation(tr_idx)
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            P = torch.tensor(patch[b], device=dev).float()
            S = torch.tensor(state[b], device=dev).float()
            out = net(P, S)
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            opt.step()
            net.update_target()
    # eval: vision-only fused latent (state masked)
    net.eval()
    zv = []
    with torch.no_grad():
        for i in range(0, len(patch), 512):
            P = torch.tensor(patch[i:i + 512], device=dev).float()
            S = torch.tensor(state[i:i + 512], device=dev).float()
            ctx = net._context(P, S)
            zv.append(net.fuse(ctx, net._mask(block_state=True, device=dev)).cpu().numpy())
    zv = np.concatenate(zv)
    pca = PCA(n_components=args.d).fit_transform(StandardScaler().fit_transform(vmean))
    return {"zv": probe(zv, kin, trm, tem), "raw": probe(vmean, kin, trm, tem),
            "pca": probe(pca, kin, trm, tem), "rankme_zv": rankme(zv[tem])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/dev/shm/wae_tmp/mm_patch.npz")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--queries", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    dd = np.load(args.cache, allow_pickle=True)
    patch, state, scene = dd["patch"], dd["state"], dd["scene"]
    kin = np.delete(state, FT_DIMS, axis=1)
    vmean = patch.astype(np.float32).mean(1)  # (N,768) pooled vision, for raw + PCA baselines
    print(f"{len(scene)} frames | {len(set(scene.tolist()))} scenes | patch {patch.shape} | "
          f"seeds={args.seeds} d={args.d} queries={args.queries}", flush=True)

    rows = []
    for s in range(args.seeds):
        r = run_seed(patch, state, kin, vmean, scene, s, args, dev)
        rows.append(r)
        print(f"  seed {s}: z_v->state {r['zv']:.3f} | raw {r['raw']:.3f} | pca256 {r['pca']:.3f} | "
              f"RankMe z_v {r['rankme_zv']:.0f}", flush=True)

    def agg(k):
        a = np.array([r[k] for r in rows]); return a.mean(), a.std()
    print("\n=== mean±std over seeds (R2 -> state, scene-held-out) ===")
    for k, lbl in [("raw", "raw vision(768)"), ("pca", "PCA-256 vision (compression ctrl)"),
                   ("zv", "Perceiver z_v(256, cross-modal)")]:
        m, sd = agg(k); print(f"  {lbl:>34}: {m:.3f} ±{sd:.3f}")
    gz_raw = np.array([r["zv"] - r["raw"] for r in rows])
    gz_pca = np.array([r["zv"] - r["pca"] for r in rows])
    print(f"  {'z_v - raw':>34}: {gz_raw.mean():+.3f} ±{gz_raw.std():.3f} ({'all+' if (gz_raw>0).all() else 'mixed'})")
    print(f"  {'z_v - pca256 (cross-modal gain)':>34}: {gz_pca.mean():+.3f} ±{gz_pca.std():.3f} "
          f"({'all+' if (gz_pca>0).all() else 'mixed'})")
    print(f"  RankMe z_v {agg('rankme_zv')[0]:.0f}  (no collapse if high)", flush=True)


if __name__ == "__main__":
    main()
