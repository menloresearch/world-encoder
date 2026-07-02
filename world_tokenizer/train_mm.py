"""Stage 2 train + eval: train MMJepa on the mm cache; eval = held-out cross-modal prediction + RankMe.

The thesis test: after cross-modal training, does the learned VISION embedding (256-d) predict the
kinematic state about as well as RAW frozen vision (768-d, the Step-1 gate's 0.43 baseline)? If yes,
cross-modal training kept state-relevant structure while compressing 3x. RankMe must stay high (no
collapse). Scene-held-out throughout.

    python -m world_tokenizer.train_mm --cache /dev/shm/wae_tmp/mm_cache.npz --epochs 60
"""
import argparse

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from world_tokenizer.mm_jepa import MMJepa
from world_tokenizer.state import FT_DIMS


def rankme(Z):
    s = torch.linalg.svdvals(torch.as_tensor(np.asarray(Z), dtype=torch.float32))
    p = s / s.sum() + 1e-5
    return float(torch.exp(-(p * torch.log(p)).sum()))


def scene_masks(scene, seed=0, frac=0.3):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/dev/shm/wae_tmp/mm_cache.npz")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    dd = np.load(args.cache, allow_pickle=True)
    vision, state, scene = dd["vision"], dd["state"], dd["scene"]
    kin = np.delete(state, FT_DIMS, axis=1)  # 22-dim kinematic target for the cross-modal probe
    trm, tem = scene_masks(scene)
    V = torch.tensor(vision, device=dev)
    S = torch.tensor(state, device=dev)
    tr_idx = np.where(trm)[0]
    print(f"{len(scene)} frames | train {trm.sum()} / test {tem.sum()} (scene-held-out) | d={args.d}", flush=True)

    net = MMJepa(d=args.d).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    for ep in range(args.epochs):
        net.train()
        perm = np.random.permutation(tr_idx)
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            out = net(V[b], S[b])
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            opt.step()
            net.update_target()
        if ep % 15 == 0 or ep == args.epochs - 1:
            with torch.no_grad():
                ev = net.enc_v(V[tr_idx[:2000]]).cpu().numpy()
            print(f"ep{ep} loss={float(out['loss']):.4f} inv={float(out['inv']):.4f} "
                  f"sig={float(out['sig']):.2f} rankme(ev)={rankme(ev):.1f}", flush=True)

    net.eval()
    with torch.no_grad():
        ev = net.enc_v(V).cpu().numpy()
        es = net.enc_s(S).cpu().numpy()
    z = np.concatenate([ev, es], axis=1)
    vpca = PCA(n_components=32).fit_transform(StandardScaler().fit_transform(vision))

    print("\n=== held-out eval (R2, scene-held-out) ===", flush=True)
    print(f"  raw vision(768) -> state : {probe(vision, kin, trm, tem):.3f}   (Step-1 baseline)")
    print(f"  learned ev(256) -> state : {probe(ev, kin, trm, tem):.3f}   (cross-modal, compressed 3x)")
    print(f"  fused   z(512)  -> state : {probe(z, kin, trm, tem):.3f}   (contains es; sanity ~high)")
    print(f"  learned es(256) -> vision(PCA32): {probe(es, vpca, trm, tem):.3f}   (reverse)")
    print(f"  RankMe   ev={rankme(ev[tem]):.1f}  es={rankme(es[tem]):.1f}  z={rankme(z[tem]):.1f}  "
          f"(max {args.d} / {2 * args.d} for z)", flush=True)


if __name__ == "__main__":
    main()
