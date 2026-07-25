"""Action-conditioned next-latent predictor on RoboCasa (V0.2.md N2, unblocked 2026-07-20).

The RH20T attempt failed because dq is REALIZED motion (already in z_t). RoboCasa logs the
COMMANDED 12-D action per step — genuine new information at frame t. Test: predict the
in-domain pretrained encoder's fused latent Δ cache-steps ahead (stride 5 frames = 0.25 s),
arms = carry-forward | z_t only | z_t + command sequence over the gap. If commands are a
real "X", the action arm should beat the no-action arm (unlike RH20T's dq).

    python -m world_tokenizer.robocasa_predictor \
      --cache /mnt/nas/data/robocasa/kepler_cache/pnp_patches.npz \
      --ckpt  /mnt/nas/data/robocasa/kepler_ckpt/pnp_mc3 \
      --dataset /mnt/nas/data/robocasa/datasets/v1.0/target/atomic/PickPlaceCounterToCabinet/20250811/lerobot
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_tokenizer.robocasa_pretrain import MMPerceiverMC  # noqa: E402

STRIDE = 5  # must match the cache's --stride


def encode_all(ckpt_dir, patch, state, dev, bs=512):
    meta = json.load(open(os.path.join(ckpt_dir, "results.json")))
    mu = np.array(meta["state_mu"], dtype=np.float32)
    sd = np.array(meta["state_sd"], dtype=np.float32)
    s = (state - mu) / sd
    model = MMPerceiverMC(n_cams=patch.shape[1], d=meta["args"]["d"],
                          state_dim=state.shape[1],
                          n_queries=meta["args"]["queries"]).to(dev)
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, "seed0.pt"),
                                     map_location=dev))
    model.eval()
    zs = []
    with torch.no_grad():
        for i in range(0, len(patch), bs):
            p = torch.from_numpy(patch[i:i + bs].astype(np.float32)).to(dev)
            st = torch.from_numpy(s[i:i + bs]).to(dev)
            zs.append(model.embed(p, st).cpu().numpy())
    return np.concatenate(zs)


def load_actions(dataset, ep_arr, t_arr):
    """acts[i] = the STRIDE commands issued from frame t_i on (flattened, [STRIDE*12])."""
    import pandas as pd
    per_ep = {}
    for pq in sorted(glob.glob(os.path.join(dataset, "data", "chunk-*", "*.parquet"))):
        df = pd.read_parquet(pq, columns=["action", "episode_index"])
        per_ep[int(df["episode_index"].iloc[0])] = np.stack(
            df["action"].to_numpy()).astype(np.float32)
    A = per_ep[ep_arr[0]].shape[1]
    acts = np.zeros((len(ep_arr), STRIDE * A), dtype=np.float32)
    ok = np.zeros(len(ep_arr), dtype=bool)
    for i, (e, t) in enumerate(zip(ep_arr, t_arr)):
        a = per_ep[int(e)]
        if t + STRIDE <= len(a):
            acts[i] = a[t:t + STRIDE].ravel()
            ok[i] = True
    return acts, ok


def fit_eval(x_tr, y_tr, x_te, y_te):
    """Ridge + small early-stopped MLP; return best test MSE (standardized space)."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    reg = Ridge(alpha=10.0).fit(x_tr, y_tr)
    mse_lin = float(((reg.predict(x_te) - y_te) ** 2).mean())

    xtr, xv, ytr, yv = train_test_split(x_tr, y_tr, test_size=0.1, random_state=0)
    dev = "cuda"
    net = torch.nn.Sequential(
        torch.nn.Linear(x_tr.shape[1], 512), torch.nn.GELU(),
        torch.nn.Linear(512, y_tr.shape[1])).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    xt, yt = torch.tensor(xtr, device=dev), torch.tensor(ytr, device=dev)
    xvv, yvv = torch.tensor(xv, device=dev), torch.tensor(yv, device=dev)
    best, best_sd, patience = np.inf, None, 0
    for ep in range(200):
        net.train()
        perm = torch.randperm(len(xt), device=dev)
        for i in range(0, len(xt), 1024):
            sl = perm[i:i + 1024]
            loss = ((net(xt[sl]) - yt[sl]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = float(((net(xvv) - yvv) ** 2).mean())
        if v < best - 1e-5:
            best, patience = v, 0
            best_sd = {k: p.detach().clone() for k, p in net.state_dict().items()}
        else:
            patience += 1
            if patience >= 10:
                break
    net.load_state_dict(best_sd)
    with torch.no_grad():
        mse_mlp = float(((net(torch.tensor(x_te, device=dev)).cpu().numpy()
                          - y_te) ** 2).mean())
    return min(mse_lin, mse_mlp), {"lin": mse_lin, "mlp": mse_mlp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="results/temporal/robocasa_predictor_pnp.json")
    ap.add_argument("--deltas", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()
    dev = "cuda"

    z = np.load(args.cache)
    patch, state, ep, t = z["patch"], z["state"].astype(np.float32), z["ep"], z["t"]
    print(f"{len(ep)} samples", flush=True)
    lat = encode_all(args.ckpt, patch, state, dev)
    lat = (lat - lat.mean(0)) / (lat.std(0) + 1e-6)               # standardize latent
    acts, act_ok = load_actions(args.dataset, ep, t)
    acts = (acts - acts.mean(0)) / (acts.std(0) + 1e-6)

    # index pairs: samples are stride-ordered within episode -> next row = +STRIDE frames
    res = {"n": len(ep), "deltas": {}}
    for d in args.deltas:
        src = np.arange(len(ep) - d)
        keep = (ep[src] == ep[src + d]) & (t[src + d] == t[src] + d * STRIDE)
        # action sequence over the whole gap = commands from each intermediate sample
        a_seq = np.concatenate([acts[src + j] for j in range(d)], axis=1)
        for j in range(d):
            keep &= act_ok[src + j]
        src = src[keep]; a_seq = a_seq[keep]
        is_test = (ep[src] % 10 == 0)
        x, y = lat[src], lat[src + d]
        tr, te = ~is_test, is_test
        carry = float(((x[te] - y[te]) ** 2).mean())
        m_no, d_no = fit_eval(x[tr], y[tr], x[te], y[te])
        xa = np.concatenate([x, a_seq], axis=1)
        m_act, d_act = fit_eval(xa[tr], y[tr], xa[te], y[te])
        row = {"pairs": int(len(src)), "carry_mse": carry,
               "noact_mse": m_no, "act_mse": m_act,
               "noact_over_carry": m_no / carry, "act_over_carry": m_act / carry,
               "act_over_noact": m_act / m_no, "detail": {"noact": d_no, "act": d_act}}
        res["deltas"][d] = row
        print(f"Δ{d} ({d*STRIDE} frames = {d*0.25:.2f}s): carry {carry:.3f} | "
              f"no-act {m_no:.3f} ({m_no/carry:.2f}x) | "
              f"act {m_act:.3f} ({m_act/carry:.2f}x) | act/no-act {m_act/m_no:.3f}",
              flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"SAVED {args.out}", flush=True)


if __name__ == "__main__":
    main()
