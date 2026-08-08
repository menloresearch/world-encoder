"""Certificate experiment (PROGRESS pt. 28 addendum, after arXiv 2607.27017).

Question: are z8 transitions on the DINO-WM cache learnable by ANY reasonable
estimator, or only unlearnable by their ViT predictor recipe? Train our rung-0
LatentDynamics trunk (Markov, residual, standardized) on the SAME cache,
timebase (frameskip 5) and action convention (5 raw relative actions per model
step, their normalization), then report the SAME 5-step true-action rollout
ratio measured in pt. 28 (their ViT: z8 1.045 vs dense 0.726).

  CUDA_VISIBLE_DEVICES=7 python -m world_tokenizer.pusht_dynamics_dwm_cert
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from world_tokenizer.pusht_dynamics import LatentDynamics

DATA = Path("/home/menlo/dwm_data/pusht_noise")
# dino_wm/datasets/pusht_dset.py constants (actions pre-scaled by 1/100)
ACTION_MEAN = torch.tensor([-0.0087, 0.0068])
ACTION_STD = torch.tensor([0.2019, 0.2002])
FS = 5                                   # frameskip (their timebase)


def load_split(split, n_eps=None, stem="z8_latents"):
    root = DATA / split
    if (root / f"{stem}.pth").exists():
        lat = torch.load(root / f"{stem}.pth", mmap=True, weights_only=True)
        lat = lat[:]
    else:
        parts = sorted(root.glob(f"{stem}_shard*of*.pth"))
        lat = torch.cat([torch.load(p, weights_only=True) for p in parts], 0)
    acts = torch.load(root / "rel_actions.pth").float() / 100.0
    acts = (acts - ACTION_MEAN) / ACTION_STD
    with open(root / "seq_lengths.pkl", "rb") as f:
        L = pickle.load(f)
    if n_eps:
        lat, acts, L = lat[:n_eps], acts[:n_eps], L[:n_eps]
    return lat, acts, list(L)


def windows(L, span):
    out = []
    for ep, T in enumerate(L):
        for t in range(0, T - span):
            out.append((ep, t))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-eps", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--latent-stem", default="z8_latents")
    ap.add_argument("--n-tokens", type=int, default=8)
    ap.add_argument("--out-name", default="pusht_dyn_dwm_z8")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda"

    lat_tr, act_tr, L_tr = load_split("train", args.train_eps, args.latent_stem)
    lat_va, act_va, L_va = load_split("val", stem=args.latent_stem)
    print(f"train {lat_tr.shape} val {lat_va.shape}", flush=True)

    # standardization over train latents (subsample rows for speed)
    flat = lat_tr[:200].reshape(-1, 256).float()  # in_dim fixed at 256 across arms
    mu, sd = flat.mean(0).to(dev), (flat.std(0) + 1e-6).to(dev)

    def get(lat, eps, ts):
        z = lat[eps, ts].float().to(dev)
        return (z - mu) / sd

    def acts_step(acts, eps, ts):
        # 5 raw actions covering [t, t+5) -> 10-D, their normalization
        idx = ts[:, None] + np.arange(FS)[None]
        a = acts[eps[:, None], idx]                      # [B,5,2]
        return a.reshape(len(eps), -1).to(dev)

    model = LatentDynamics(args.n_tokens, 256, act_dim=2 * FS).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    w_tr = windows(L_tr, FS)
    w_va = windows(L_va, FS)
    print(f"{len(w_tr)} train pairs, {len(w_va)} val pairs", flush=True)

    def val_mse():
        model.eval()
        se_p = se_c = m = 0
        with torch.no_grad():
            for i in range(0, len(w_va), 512):
                ep, t = w_va[i:i+512, 0], w_va[i:i+512, 1]
                z0, z1 = get(lat_va, ep, t), get(lat_va, ep, t + FS)
                r = z1 - z0
                pr = model(z0, acts_step(act_va, ep, t))
                se_p += float((pr - r).square().sum()); se_c += float(r.square().sum())
                m += r.numel()
        return se_p / m, se_c / m

    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        perm = np.random.permutation(len(w_tr))
        for i in range(0, len(perm), args.batch):
            sl = perm[i:i+args.batch]
            ep, t = w_tr[sl, 0], w_tr[sl, 1]
            z0, z1 = get(lat_tr, ep, t), get(lat_tr, ep, t + FS)
            loss = (model(z0, acts_step(act_tr, ep, t)) - (z1 - z0)).square().mean()
            opt.zero_grad(); loss.backward(); opt.step()
        vp, vc = val_mse()
        print(f"ep {epoch}: val {vp:.5f} carry {vc:.5f} ratio {vp/vc:.3f} "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)

    # THE pt. 28 metric: 5 chained model-steps (25 frames), true actions, vs carry
    model.eval()
    w5 = windows(L_va, 5 * FS)
    se_p = se_c = m = 0
    with torch.no_grad():
        for i in range(0, len(w5), 512):
            ep, t = w5[i:i+512, 0], w5[i:i+512, 1]
            z = get(lat_va, ep, t)
            for k in range(5):
                z = z + model(z, acts_step(act_va, ep, t + k * FS))
            zt = get(lat_va, ep, t + 5 * FS)
            z0 = get(lat_va, ep, t)
            se_p += float((z - zt).square().sum()); se_c += float((z0 - zt).square().sum())
            m += zt.numel()
    print(f"CERTIFICATE 5-step rollout: pred {se_p/m:.4f} carry {se_c/m:.4f} "
          f"ratio {(se_p/m)/(se_c/m):.3f}  (their ViT: z8 1.045, dense 0.726)", flush=True)

    out = Path("/mnt/nas/data/robocasa/kepler_ckpt") / args.out_name
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "mu": mu.cpu(), "sd": sd.cpu(),
                "cfg": {"n_tokens": args.n_tokens, "in_dim": 256, "act_dim": 2 * FS,
                        "frameskip": FS, "train_eps": args.train_eps}},
               out / f"seed{args.seed}.pt")
    print(f"SAVED {out}/seed{args.seed}.pt", flush=True)


if __name__ == "__main__":
    main()
