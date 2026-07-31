"""D1/D2 shortcut detectors (02_shortcut_mitigation §4) — offline, no rollouts.

Given an encoder ckpt (v0.3 MMPerceiverMCPatch or v0.4 MMPerceiverMCStateAdd, auto-
detected via the 'alpha' key) + held-out cache demos (ep % 10 == 0):

  D1  null-state / null-vision output-delta sweep. Encoder-side version of 02's D1
      (no policy head here): relative L2 delta of the unpooled latent set under
      intact / null-state (dataset mean = zeros in normalized space, RDT convention) /
      permuted-state (within-split shuffle) / null-vision (per-position mean patch
      features — the feature-space analog of RDT's background image). Reported per
      query subset (conditioned 0..3 vs unconditioned 4..7). For v0.4 ckpts the same
      sweep is run on the AUX cosine loss (03's aux-health gate: zeroing VISION must
      move the aux loss; zeroing state ideally moves it less), plus permuted ACTIONS.
      Verdict: SHORTCUT if vision-delta ~ 0 while state-delta is large.
  D2  state -> z ridge probe (RidgeCV, episode-held-out), per-token R^2.
      Verdict: COLLAPSE if conditioned-token mean R^2 >= 0.9 (02 threshold proposal).
      (02 also wants a z -> object-pose floor; object pose is not in this cache —
      noted as not-computed.)

    cd /home/menlo/brain/ishneet/world-encoder && \
    CUDA_VISIBLE_DEVICES=0 /home/menlo/brain/ishneet/robocasa/.venv/bin/python3 \
      -m world_tokenizer.v04_detector \
      --ckpt /mnt/nas/data/robocasa/kepler_ckpt/pnp_mc3_v03_lp2.0/seed0.pt \
      --dataset /mnt/nas/data/robocasa/datasets/v1.0/target/atomic/PickPlaceCounterToCabinet/20250811/lerobot
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402

from world_tokenizer.robocasa_pretrain_v03 import MMPerceiverMCPatch  # noqa: E402
from world_tokenizer.robocasa_pretrain_v04 import (  # noqa: E402
    MMPerceiverMCStateAdd, build_future_and_actions, load_actions, pool_patches)


def embed_sets(model, patch, state, dev, bs=512):
    """Unpooled latent sets [N, M, d] (v0.4 embed_set | v0.3 masked pool=False path)."""
    zs = []
    for i in range(0, len(state), bs):
        p = torch.from_numpy(patch[i:i + bs].astype(np.float32)).to(dev)
        s = torch.from_numpy(state[i:i + bs]).to(dev)
        if isinstance(model, MMPerceiverMCStateAdd):
            z = model.embed_set(p, s)
        else:
            with torch.no_grad():
                z = model.fuse(model._context(p, s), model._mask(True, dev), pool=False)
        zs.append(z.cpu().numpy())
    return np.concatenate(zs)


def aux_cos(model, patch, state, act, fut_v, dev, bs=512):
    tot, n = 0.0, 0
    for i in range(0, len(state), bs):
        p = torch.from_numpy(patch[i:i + bs].astype(np.float32)).to(dev)
        s = torch.from_numpy(state[i:i + bs]).to(dev)
        a = torch.from_numpy(act[i:i + bs]).to(dev)
        fv = torch.from_numpy(fut_v[i:i + bs]).to(dev)
        pred = model.predict_future(p, s, a)
        c = (1.0 - torch.nn.functional.cosine_similarity(pred, fv, dim=-1)).mean()
        tot += float(c) * len(s)
        n += len(s)
    return tot / n


def rel_delta(z_c, z_0):
    """Per-token mean ||z_c - z_0|| / mean ||z_0|| -> [M]."""
    num = np.linalg.norm(z_c - z_0, axis=-1).mean(0)
    den = np.linalg.norm(z_0, axis=-1).mean(0)
    return num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache",
                    default="/mnt/nas/data/robocasa/kepler_cache/pnp_patches.npz")
    ap.add_argument("--dataset",
                    default="/mnt/nas/data/robocasa/datasets/v1.0/target/atomic/"
                            "PickPlaceCounterToCabinet/20250811/lerobot")
    ap.add_argument("--n-test", type=int, default=2048)
    ap.add_argument("--n-probe-train", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    np.random.seed(args.seed)
    dev = args.device

    res = json.load(open(os.path.join(os.path.dirname(args.ckpt), "results.json")))
    a = res["args"]
    lp = res.get("lamb_patch", a.get("lamb_patch"))
    if isinstance(lp, list):
        lp = lp[0]
    mu = np.array(res["state_mu"], np.float32)
    sd_ = np.array(res["state_sd"], np.float32)

    d = np.load(args.cache)
    patch, state_raw, ep, t = d["patch"], d["state"].astype(np.float32), d["ep"], d["t"]
    state = (state_raw - mu) / sd_
    K = patch.shape[1]
    is_test = (ep % 10 == 0)
    te = np.flatnonzero(is_test)[:args.n_test]
    tr = np.flatnonzero(~is_test)
    tr = tr[np.linspace(0, len(tr) - 1, min(args.n_probe_train, len(tr))).astype(int)]

    sd_ck = torch.load(args.ckpt, map_location="cpu")
    is_v04 = "alpha" in sd_ck
    kw = dict(n_cams=K, d=a["d"], state_dim=state.shape[1], n_queries=a["queries"],
              lamb_patch=lp, mask_ratio=a["mask_ratio"])
    if is_v04:
        model = MMPerceiverMCStateAdd(
            act_len=res["h_eff"], act_dim=res["act_dim"],
            n_pred=a.get("n_pred", 16), **kw).to(dev).eval()
        model.load_state_dict(sd_ck, strict=True)
        alpha = torch.tanh(model.alpha.detach()).cpu().numpy()
        print(f"[v0.4 StateAdd ckpt] tanh(alpha) = {np.round(alpha, 4).tolist()}")
    else:
        model = MMPerceiverMCPatch(**kw).to(dev).eval()
        model.load_state_dict(sd_ck, strict=True)
        print("[v0.3 ckpt] state never enters the vision-masked forward "
              "(expect state deltas = 0 by construction)")
    M = a["queries"]
    n_cond = M // 2
    print(f"held-out demos: {len(te)} test rows ({len(np.unique(ep[te]))} episodes); "
          f"probe-train {len(tr)} rows\n")

    # ---------- D1: null/permuted sweep on the latent set ----------
    null_vis = patch[tr[:4096]].astype(np.float32).mean(0, keepdims=True)  # [1,K,196,768]
    perm = np.random.permutation(len(te))

    z0 = embed_sets(model, patch[te], state[te], dev)
    conds = {
        "null_state (mean)": (patch[te], np.zeros_like(state[te])),
        "perm_state": (patch[te], state[te][perm]),
        "null_vision (mean)": (np.broadcast_to(null_vis, patch[te].shape), state[te]),
    }
    print("D1 — relative latent delta vs intact (per query subset):")
    print(f"{'condition':<22}{'cond q0-3':>12}{'uncond q4-7':>14}{'all':>10}")
    d1 = {}
    for name, (pc, sc) in conds.items():
        r = rel_delta(embed_sets(model, pc, sc, dev), z0)
        d1[name] = r
        print(f"{name:<22}{r[:n_cond].mean():>12.4f}{r[n_cond:].mean():>14.4f}"
              f"{r.mean():>10.4f}")
    dz_state = max(d1["null_state (mean)"].mean(), d1["perm_state"].mean())
    dz_vis = d1["null_vision (mean)"].mean()
    shortcut = dz_vis < 0.02 and dz_state > 0.10
    print(f"D1 verdict: dz_state {dz_state:.4f} vs dz_vision {dz_vis:.4f} -> "
          f"{'SHORTCUT (kill per 02 §4)' if shortcut else 'OK (vision moves the latent)'}")

    # ---------- D1-aux: aux-loss sweep (v0.4 only; 03's aux-health gate) ----------
    if is_v04:
        actions = load_actions(args.dataset)
        fut_row, act_chunk, _, _ = build_future_and_actions(ep, t, actions, res["h_eff"])
        fut_te = pool_patches(patch[fut_row[te]], bs=1024)
        act_te = act_chunk[te]
        a0 = aux_cos(model, patch[te], state[te], act_te, fut_te, dev)
        a_ns = aux_cos(model, patch[te], np.zeros_like(state[te]), act_te, fut_te, dev)
        a_nv = aux_cos(model, np.broadcast_to(null_vis, patch[te].shape).copy(),
                       state[te], act_te, fut_te, dev)
        a_pa = aux_cos(model, patch[te], state[te], act_te[perm], fut_te, dev)
        print("\nD1-aux — aux cosine loss sweep (03 aux-health gate):")
        print(f"  intact {a0:.4f} | null_state {a_ns:.4f} (d {a_ns-a0:+.4f}) | "
              f"null_vision {a_nv:.4f} (d {a_nv-a0:+.4f}) | "
              f"perm_actions {a_pa:.4f} (d {a_pa-a0:+.4f})")
        aux_short = (a_nv - a0) < max(0.02, 1.0 * (a_ns - a0))
        print(f"D1-aux verdict: {'AUX-SHORTCUT (vision barely moves aux -> kill)' if aux_short else 'OK (aux runs on vision)'}")

    # ---------- D2: state -> z ridge probe, per token ----------
    z_tr = embed_sets(model, patch[tr], state[tr], dev)
    z_te = embed_sets(model, patch[te], state[te], dev)
    r2 = np.empty(M)
    for m in range(M):
        rg = RidgeCV(alphas=np.logspace(-2, 4, 7)).fit(state[tr], z_tr[:, m])
        r2[m] = r2_score(z_te[:, m], rg.predict(state[te]),
                         multioutput="variance_weighted")
    print(f"\nD2 — state->z ridge R^2 per token: {np.round(r2, 3).tolist()}")
    print(f"  conditioned q0-3 mean {r2[:n_cond].mean():.3f} | "
          f"unconditioned q4-7 mean {r2[n_cond:].mean():.3f}")
    collapse = r2[:n_cond].mean() >= 0.9
    print(f"D2 verdict: {'COLLAPSE (cond R^2 >= 0.9 -> kill; check z->obj-pose floor)' if collapse else 'OK'}"
          f"  [z->object-pose floor NOT computed: no object pose in this cache]")

    print(f"\nOVERALL: {'KILL-FLAGGED' if (shortcut or collapse) else 'no shortcut flags'}")


if __name__ == "__main__":
    main()
