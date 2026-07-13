"""Phase-1 extra evals from a saved checkpoint — NO training (runs on the frozen matrix
checkpoints, on cached patch features so no ViT needed):

  1.7  triplet accuracy   — anchor/pos = same scene, nearby tick; negatives tiered
       (diff scene same robot, then diff robot). Geometry check the plan owed.
  1.8  PCA figures        — z_v PCA-2D scatter colored by robot / cfg.
  kuka diagnostic         — per-motor-dim probe R² (z_v vs raw) on kuka, to locate the
                            one cell where raw beats fusion.

    python -m world_tokenizer.eval_extras --ckpt .../phase1/all/seed0.pt --out figures/all
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_tokenizer.dataloader import ChunkDataset, load_split           # noqa: E402
from world_tokenizer.mm_perceiver2 import MMPerceiverChunks                # noqa: E402
from world_tokenizer.train_chunks import encode_zv, probe_r2              # noqa: E402
from metrics.metrics import triplet_accuracy, format_triplet_report, effective_rank  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache-dir", default="/mnt/nas/data/RH20T/caches")
    ap.add_argument("--cfgs", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--queries", type=int, default=8)
    ap.add_argument("--out", default="figures/all")
    ap.add_argument("--n-triplets", type=int, default=6000)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    split = load_split()
    model = MMPerceiverChunks(d=args.d, n_queries=args.queries).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev))
    model.eval()

    ds = ChunkDataset(args.cache_dir, tuple(args.cfgs))
    is_test = np.array([split[ds.groups[i]] == "test" for i in ds._group_idx])
    zv, raw = encode_zv(model, ds, dev)                       # (N,d), (N,768)

    te = np.flatnonzero(is_test)
    zt, sidx = zv[te], ds._scene_idx[te]
    rid, cfg = ds._d["robot_id"][te], ds._d["cfg"][te]
    print(f"\n=== {args.ckpt} | {len(te)} test chunks ===", flush=True)
    print(f"RankMe(z_v test) = {effective_rank(zt)['rankme']:.1f}", flush=True)

    # ---------- 1.7 triplet accuracy ----------
    rng = np.random.default_rng(0)
    by_scene = defaultdict(list)
    for i, s in enumerate(sidx):
        by_scene[s].append(i)
    multi = [np.array(v) for v in by_scene.values() if len(v) >= 2]
    A, P, Nn, tiers = [], [], [], []
    same_r = {r: np.flatnonzero(rid == r) for r in np.unique(rid)}
    diff_r = {r: np.flatnonzero(rid != r) for r in np.unique(rid)}
    for _ in range(args.n_triplets):
        grp = multi[rng.integers(len(multi))]
        a, p = rng.choice(grp, 2, replace=False)
        if rng.random() < 0.5:
            pool = same_r[rid[a]]; pool = pool[sidx[pool] != sidx[a]]; tier = "diff_scene_same_robot"
        else:
            pool = diff_r[rid[a]]; tier = "diff_robot"
        if len(pool) == 0:
            continue
        A.append(a); P.append(p); Nn.append(pool[rng.integers(len(pool))]); tiers.append(tier)
    A, P, Nn = np.array(A), np.array(P), np.array(Nn)
    res = {d: triplet_accuracy(zt[A], zt[P], zt[Nn], tiers=tiers, distance=d)
           for d in ("mse", "cosine")}
    print("\n--- 1.7 triplet accuracy (z_v; pos = same scene / nearby) ---")
    print(format_triplet_report(res), flush=True)

    # ---------- 1.8 PCA figures ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        sub = rng.choice(len(te), min(8000, len(te)), replace=False)
        p2 = PCA(n_components=2, random_state=0).fit_transform(zt[sub] - zt.mean(0))
        names = ["flexiv", "ur5", "franka", "kuka"]
        for key, lab, title in [("robot", rid[sub], "robot"), ("cfg", cfg[sub], "cfg")]:
            fig, ax = plt.subplots(figsize=(6, 5))
            sc = ax.scatter(p2[:, 0], p2[:, 1], c=lab, s=4, cmap="tab10", alpha=0.5)
            ax.set_title(f"z_v PCA — colored by {title}")
            cb = plt.colorbar(sc); cb.set_label(title)
            fig.savefig(f"{args.out}/pca_{key}.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
        # state-colored: does the latent encode continuous world-state, not just robot id?
        motor_all = ds._d["motor"].reshape(len(ds), -1)[te]
        grip = motor_all[sub, 21]                                  # gripper: row7 ch0 (symlog width)
        ee_s, eem_s = ds._d["ee"][te][sub], ds._d["ee_mask"][te][sub]
        fmag = np.linalg.norm(ee_s[..., :6], axis=-1)              # F/T magnitude per slot
        fmag = (fmag * eem_s).sum(1) / np.clip(eem_s.sum(1), 1, None)
        has_ee = eem_s.any(1)
        for key, lab, title, m in [("gripper", grip, "gripper (symlog width)", np.ones(len(sub), bool)),
                                   ("force", fmag, "force magnitude", has_ee)]:
            if m.sum() < 50:
                continue
            fig, ax = plt.subplots(figsize=(6, 5))
            sc = ax.scatter(p2[m, 0], p2[m, 1], c=lab[m], s=4, cmap="viridis", alpha=0.5)
            ax.set_title(f"z_v PCA — colored by {title}")
            cb = plt.colorbar(sc); cb.set_label(title)
            fig.savefig(f"{args.out}/pca_{key}.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
        print(f"\n--- 1.8 PCA figures saved to {args.out}/ (robot, cfg, gripper, force) ---", flush=True)
    except Exception as e:
        print(f"\n[PCA figures skipped: {e}]", flush=True)

    # ---------- kuka diagnostic: per-motor-dim R² z_v vs raw ----------
    if 6 in args.cfgs or 7 in args.cfgs:
        kmask = np.isin(cfg, [6, 7])
        d = ds._d
        motor = d["motor"].reshape(len(ds), -1)[te][kmask]        # [Nk,24]
        mvalid = d["motor_mask"].reshape(len(ds), -1)[te][kmask].all(0)
        tr = np.flatnonzero(~is_test[np.isin(ds._d["cfg"], [6, 7])])  # not used; probe on kuka test only via split
        # simple within-kuka-test split for a per-dim probe (fit/test 70/30 by scene)
        kser = sidx[kmask]
        uniq = np.unique(kser); rng.shuffle(uniq)
        trs = set(uniq[: int(0.7 * len(uniq))].tolist())
        m_tr = np.array([s in trs for s in kser]); m_te = ~m_tr
        y = motor[:, mvalid]
        print("\n--- kuka diagnostic: per-motor-dim R² (z_v vs raw), kuka test ---")
        zk, rk = zt[kmask], raw[te][kmask]
        for name, X in [("z_v", zk), ("raw", rk)]:
            per = [probe_r2(X[m_tr], y[m_tr, j:j + 1], X[m_te], y[m_te, j:j + 1])
                   for j in range(y.shape[1])]
            print(f"  {name:4s}: mean {np.mean(per):+.3f} | per-dim " +
                  " ".join(f"{v:+.2f}" for v in per), flush=True)

    print("\nDONE eval_extras", flush=True)


if __name__ == "__main__":
    main()
