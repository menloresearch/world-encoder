"""SeeSE3-style probes (E1, V0.3.md) — does the fuse keep the 3D structure raw features have?

Adapted from SeeSE3 (arXiv 2607.14228) to our setting: the cameras are static and the ROBOT
moves, so end-effector motion stands in for camera motion.

  (a) mutual-neighborhood — within-episode k-NN overlap between representation space and
      3D EE-position space (the paper's feature-topology-vs-spatial-topology metric;
      within-episode = the static-scene analog, cross-episode layouts are confounded).
  (b) delta-SE(3) linear readout (Poincare-adapter-style) — ridge regression from
      z_{t+D} - z_t to the EE's motion (delta-pos 3 + delta-rot rotvec 3), episode-level
      split. "Linear accessibility of SE(3) motion from latent displacements."

Representations ranked:
  raw ViT patches, coarse-spatial (3cam x 2x2 mean-pool = the 0.413 action-readout ceiling row)
  pretrained fused z (kepler z cache rows, vision-only latent)
  state16 (the encoder's own 16-d proprio target; pipeline sanity — (b) should be ~1.0)

Inputs: kepler_cache/pnp_patches.npz + frame_cache/index.json + kepler_z_cache z.f32.
CPU-only (numpy + sklearn). Deltas are in probe-sample steps: stride-5 frames at 20 fps
=> 1 step = 0.25 s.

    python -m world_tokenizer.seese3_probe --out results/downstream/seese3_pnp.json
"""
import argparse
import json
import os

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

DS = "/mnt/nas/data/robocasa/datasets/v1.0/target/atomic/PickPlaceCounterToCabinet/20250811"
PATCH_NPZ = "/mnt/nas/data/robocasa/kepler_cache/pnp_patches.npz"
Z_CACHE = f"{DS}/kepler_z_cache_pnp_mc3_seed0"
FRAME_IDX = f"{DS}/frame_cache/index.json"

# state16 layout (meta/modality.json): base_pos 0:3, base_rot 3:7,
# ee_pos_rel 7:10, ee_rot_rel(quat xyzw, robosuite convention) 10:14, gripper 14:16
EE_POS, EE_QUAT = slice(7, 10), slice(10, 14)


def quat_conj(q):
    return q * np.array([-1.0, -1.0, -1.0, 1.0], dtype=q.dtype)


def quat_mul(a, b):                                   # xyzw, batched
    x1, y1, z1, w1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    x2, y2, z2, w2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2], axis=-1)


def quat_to_rotvec(q):
    q = np.where(q[..., 3:4] < 0, -q, q)              # sign-canonical
    v, w = q[..., :3], np.clip(q[..., 3], -1.0, 1.0)
    ang = 2.0 * np.arctan2(np.linalg.norm(v, axis=-1), w)
    axis = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)
    return axis * ang[..., None]


def coarse_pool(patch_f16, chunk=2000):
    """[N,3,196,768] f16 -> [N, 3*2*2*768] f32 (14x14 grid mean-pooled to 2x2)."""
    n = patch_f16.shape[0]
    out = np.empty((n, 3 * 4 * 768), dtype=np.float32)
    for i in range(0, n, chunk):
        p = patch_f16[i:i + chunk].astype(np.float32)          # [c,3,196,768]
        p = p.reshape(-1, 3, 2, 7, 2, 7, 768).mean((3, 5))     # [c,3,2,2,768]
        out[i:i + chunk] = p.reshape(len(p), -1)
    return out


def knn_sets(D, k):
    np.fill_diagonal(D, np.inf)
    return np.argsort(D, axis=1)[:, :k]


def mutual_neighborhood(rep, pos, ep, k=5):
    """Mean k-NN overlap between rep space (cosine) and 3D EE space (euclid), per episode."""
    overlaps, chance = [], []
    for e in np.unique(ep):
        m = ep == e
        n = int(m.sum())
        if n < k + 2:
            continue
        r = rep[m] / (np.linalg.norm(rep[m], axis=1, keepdims=True) + 1e-12)
        Dr = 1.0 - r @ r.T
        p = pos[m]
        Dp = np.linalg.norm(p[:, None] - p[None, :], axis=-1)
        nr, np_ = knn_sets(Dr, k), knn_sets(Dp, k)
        overlaps.append(np.mean([len(set(a) & set(b)) / k for a, b in zip(nr, np_)]))
        chance.append(k / (n - 1))
    return float(np.mean(overlaps)), float(np.mean(chance)), len(overlaps)


def delta_pairs(ep, t, delta_steps, stride=5):
    """Indices (i, j) with same episode and t_j - t_i == stride*delta_steps."""
    order = np.lexsort((t, ep))
    eo, to = ep[order], t[order]
    idx = {(e_, t_): o for e_, t_, o in zip(eo, to, order)}
    ii, jj = [], []
    for e_, t_, o in zip(eo, to, order):
        j = idx.get((e_, t_ + stride * delta_steps))
        if j is not None:
            ii.append(o), jj.append(j)
    return np.array(ii), np.array(jj)


def se3_readout(rep, state, ep, t, delta_steps, train_eps, test_eps):
    ii, jj = delta_pairs(ep, t, delta_steps)
    X = (rep[jj] - rep[ii]).astype(np.float32)
    dpos = state[jj, EE_POS] - state[ii, EE_POS]
    drot = quat_to_rotvec(quat_mul(state[jj, EE_QUAT],
                                   quat_conj(state[ii, EE_QUAT])))
    Y = np.concatenate([dpos, drot], axis=1).astype(np.float32)
    tr = np.isin(ep[ii], train_eps)
    te = np.isin(ep[ii], test_eps)
    model = RidgeCV(alphas=np.logspace(-2, 4, 7)).fit(X[tr], Y[tr])
    pred = model.predict(X[te])
    return {
        "r2_all": float(r2_score(Y[te], pred, multioutput="variance_weighted")),
        "r2_pos": float(r2_score(Y[te, :3], pred[:, :3], multioutput="variance_weighted")),
        "r2_rot": float(r2_score(Y[te, 3:], pred[:, 3:], multioutput="variance_weighted")),
        "n_pairs_test": int(te.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knn", type=int, default=5)
    ap.add_argument("--deltas", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--out", default="results/downstream/seese3_pnp.json")
    args = ap.parse_args()

    d = np.load(PATCH_NPZ)
    state, ep, t = d["state"], d["ep"], d["t"]
    print(f"{len(ep)} samples, {len(np.unique(ep))} episodes", flush=True)

    fc = json.load(open(FRAME_IDX))
    zc_meta = json.load(open(f"{Z_CACHE}/meta.json"))
    zmm = np.memmap(f"{Z_CACHE}/z.f32", dtype=np.float32, mode="r",
                    shape=(fc["total_frames"], zc_meta["d"]))
    rows = np.array([fc["episodes"][str(e)]["start"] for e in ep]) + t
    z = np.array(zmm[rows])                                       # [N,256]
    print("fused z loaded", z.shape, flush=True)

    print("pooling raw patches (coarse 3cam x 2x2)...", flush=True)
    patches = coarse_pool(d["patch"])
    print("coarse patches", patches.shape, flush=True)

    reps = {"raw_patch_coarse": patches, "fused_z": z, "state16": state}

    rng = np.random.default_rng(0)                                # scene-level split
    eps_u = rng.permutation(np.unique(ep))
    n_tr = int(0.8 * len(eps_u))
    train_eps, test_eps = eps_u[:n_tr], eps_u[n_tr:]

    results = {"knn": args.knn, "deltas_steps": args.deltas,
               "note": "1 delta step = 0.25 s (stride-5 frames @ 20 fps)", "reps": {}}
    for name, rep in reps.items():
        r = {}
        ov, ch, n_ep = mutual_neighborhood(rep, state[:, EE_POS], ep, k=args.knn)
        r["mutual_knn_overlap"] = {"overlap": ov, "chance": ch, "episodes": n_ep}
        for ds in args.deltas:
            r[f"dse3_delta{ds}"] = se3_readout(rep, state, ep, t, ds,
                                               train_eps, test_eps)
        results["reps"][name] = r
        print(f"\n== {name} ==")
        print(f"  mutual-kNN overlap {ov:.3f} (chance {ch:.3f})")
        for ds in args.deltas:
            x = r[f"dse3_delta{ds}"]
            print(f"  dSE3 @{ds * 0.25:.2f}s: R2 all {x['r2_all']:.3f} "
                  f"pos {x['r2_pos']:.3f} rot {x['r2_rot']:.3f} "
                  f"(n={x['n_pairs_test']})", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSAVED {args.out}")


if __name__ == "__main__":
    main()
