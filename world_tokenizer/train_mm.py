"""Stage 2 train + eval (multi-seed): train MMJepa on the mm cache; eval = held-out cross-modal
prediction + RankMe, with error bars.

Thesis test: after cross-modal training, does the learned VISION embedding (256-d) predict the
kinematic state better than RAW frozen vision (768-d)? Encoder is retrained fresh per split
(held-out scenes never seen in training). RankMe must stay high (no collapse). Scene-held-out.

    python -m world_tokenizer.train_mm --cache /dev/shm/wae_tmp/mm_cache.npz --seeds 5 --epochs 60
"""
import argparse

import numpy as np
import torch
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from world_tokenizer.mm_jepa import MMJepa
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


def run_seed(vision, state, kin, scene, seed, args, dev):
    trm, tem = scene_masks(scene, seed)
    V, S = torch.tensor(vision, device=dev), torch.tensor(state, device=dev)
    tr_idx = np.where(trm)[0]
    net = MMJepa(d=args.d).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    for _ in range(args.epochs):
        net.train()
        perm = np.random.permutation(tr_idx)
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            out = net(V[b], S[b])
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            opt.step()
            net.update_target()
    net.eval()
    with torch.no_grad():
        ev = net.enc_v(V).cpu().numpy()
        es = net.enc_s(S).cpu().numpy()
    z = np.concatenate([ev, es], axis=1)
    return {"raw": probe(vision, kin, trm, tem), "ev": probe(ev, kin, trm, tem),
            "z": probe(z, kin, trm, tem), "rankme_ev": rankme(ev[tem]), "rankme_z": rankme(z[tem])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/dev/shm/wae_tmp/mm_cache.npz")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    dd = np.load(args.cache, allow_pickle=True)
    vision, state, scene = dd["vision"], dd["state"], dd["scene"]
    kin = np.delete(state, FT_DIMS, axis=1)
    print(f"{len(scene)} frames | {len(set(scene.tolist()))} scenes | seeds={args.seeds} | d={args.d} "
          f"| retrain fresh per split (no leakage)", flush=True)

    rows = []
    for s in range(args.seeds):
        r = run_seed(vision, state, kin, scene, s, args, dev)
        rows.append(r)
        print(f"  seed {s}: raw->state {r['raw']:.3f} | ev->state {r['ev']:.3f} | "
              f"z->state {r['z']:.3f} | RankMe ev {r['rankme_ev']:.0f} z {r['rankme_z']:.0f}", flush=True)

    def agg(k):
        a = np.array([r[k] for r in rows])
        return a.mean(), a.std()
    print("\n=== mean±std over seeds (R2, scene-held-out) ===")
    for k, lbl in [("raw", "raw vision(768) -> state"), ("ev", "learned ev(256) -> state"),
                   ("z", "fused z(512) -> state")]:
        m, sd = agg(k)
        print(f"  {lbl:>28}: {m:.3f} ±{sd:.3f}")
    dev_ev_raw = np.array([r["ev"] - r["raw"] for r in rows])
    print(f"  {'ev - raw (per-seed gap)':>28}: {dev_ev_raw.mean():+.3f} ±{dev_ev_raw.std():.3f}"
          f"  ({'CONSISTENT +' if (dev_ev_raw > 0).all() else 'mixed'})")
    print(f"  RankMe ev {agg('rankme_ev')[0]:.0f} | z {agg('rankme_z')[0]:.0f}  (no collapse if high)", flush=True)


if __name__ == "__main__":
    main()
