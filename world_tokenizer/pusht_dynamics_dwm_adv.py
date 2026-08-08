"""Adversarial World Modeling on our dwm trunk (PROGRESS pt. 32, after arXiv 2512.09929).

That paper diagnoses OUR pt. 31 failure verbatim — gradient/CEM optimization drives a
learned world model into off-distribution states where its errors compound — and fixes it
by finetuning the world model on FGSM-perturbed states/actions, which smooths the induced
action-loss landscape. On the DINO-WM PushT harness they report CEM 78 -> 94, Adam 54 -> 82,
GD 38 -> 56, with the latent architecture UNCHANGED. Our pt. 31 remedies (K=3 ensemble) and
pt. 31 addendum 2 (FSQ discreteness) both act on the model/representation; this one acts on
the cost landscape, which is what the exploitation exhibit actually implicates.

Finetunes from the existing z8 trunk so the ONLY delta vs the 0.12/0.08 hybrid reads is the
adversarial objective, then reports the same 5-step true-action rollout ratio as the pt. 29
certificate so the number is directly comparable (z8 0.494 / q4 0.474 / q16 0.469 / dense 0.726).

  CUDA_VISIBLE_DEVICES=3 python -m world_tokenizer.pusht_dynamics_dwm_adv
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from world_tokenizer.pusht_dynamics import LatentDynamics
from world_tokenizer.pusht_dynamics_dwm_cert import FS, load_split, windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-eps", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--latent-stem", default="z8_latents")
    ap.add_argument("--n-tokens", type=int, default=8)
    ap.add_argument("--init-ckpt",
                    default="/mnt/nas/data/robocasa/kepler_ckpt/pusht_dyn_dwm_z8/seed0.pt",
                    help="finetune from this trunk ('none' = train from scratch)")
    ap.add_argument("--adv-eps-a", type=float, default=0.1,
                    help="FGSM step on the (normalized) action input")
    ap.add_argument("--adv-eps-z", type=float, default=0.05,
                    help="FGSM step on the (standardized) latent input")
    ap.add_argument("--adv-weight", type=float, default=0.5,
                    help="loss = (1-w)*clean + w*adversarial")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out-name", default="pusht_dyn_dwm_z8adv")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda"

    lat_tr, act_tr, L_tr = load_split("train", args.train_eps, args.latent_stem)
    lat_va, act_va, L_va = load_split("val", stem=args.latent_stem)
    print(f"train {lat_tr.shape} val {lat_va.shape}", flush=True)

    model = LatentDynamics(args.n_tokens, 256, act_dim=2 * FS).to(dev)
    if args.init_ckpt != "none":
        ck = torch.load(args.init_ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model"])
        mu, sd = ck["mu"].to(dev), ck["sd"].to(dev)
        print(f"finetuning from {args.init_ckpt} (its standardization reused)", flush=True)
    else:
        flat = lat_tr[:200].reshape(-1, 256).float()
        mu, sd = flat.mean(0).to(dev), (flat.std(0) + 1e-6).to(dev)

    def get(lat, eps, ts):
        z = lat[eps, ts].float().to(dev)
        return (z - mu) / sd

    def acts_step(acts, eps, ts):
        idx = ts[:, None] + np.arange(FS)[None]
        a = acts[eps[:, None], idx]
        return a.reshape(len(eps), -1).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    w_tr = windows(L_tr, FS)
    w_va = windows(L_va, FS)
    print(f"{len(w_tr)} train pairs, {len(w_va)} val pairs", flush=True)

    def val_ratio():
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
            a = acts_step(act_tr, ep, t)
            r = z1 - z0

            # FGSM: one ascent step on the INPUTS, sign-scaled, graph discarded
            z0_g = z0.detach().requires_grad_(True)
            a_g = a.detach().requires_grad_(True)
            probe = (model(z0_g, a_g) - r).square().mean()
            gz, ga = torch.autograd.grad(probe, [z0_g, a_g])
            z0_adv = (z0 + args.adv_eps_z * gz.sign()).detach()
            a_adv = (a + args.adv_eps_a * ga.sign()).detach()

            clean = (model(z0, a) - r).square().mean()
            # target stays the TRUE residual: the model must remain accurate in the
            # neighbourhood, which is what flattens the action-loss surface CEM descends
            adv = (model(z0_adv, a_adv) - r).square().mean()
            loss = (1 - args.adv_weight) * clean + args.adv_weight * adv
            opt.zero_grad(); loss.backward(); opt.step()
        vp, vc = val_ratio()
        print(f"ep {epoch}: val {vp:.5f} carry {vc:.5f} ratio {vp/vc:.3f} "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)

    # the pt. 29 certificate metric, unchanged, so it is comparable across arms
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
          f"ratio {(se_p/m)/(se_c/m):.3f}  (z8 clean 0.494, dense 0.726)", flush=True)

    out = Path("/mnt/nas/data/robocasa/kepler_ckpt") / args.out_name
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "mu": mu.cpu(), "sd": sd.cpu(),
                "cfg": {"n_tokens": args.n_tokens, "in_dim": 256, "act_dim": 2 * FS,
                        "adv_eps_a": args.adv_eps_a, "adv_eps_z": args.adv_eps_z,
                        "adv_weight": args.adv_weight}},
               out / f"seed{args.seed}.pt")
    print(f"SAVED {out}/seed{args.seed}.pt", flush=True)


if __name__ == "__main__":
    main()
