"""v1 temporal (Phase-2) trainer: MMPerceiverTemporal over windowed chunk caches.

Cheap by design — the caches store precomputed frozen ViT patch tokens, so there is no ViT
forward; single-GPU is plenty. Trains the flat multi-rate Perceiver with masked
cross-modal-across-time latent prediction + per-timestep SIGReg (TEMPORAL_ARCH.md §5, §16),
then reports a collapse guard (RankMe + emb std) on held-out windows.

    python -m world_tokenizer.train_temporal --train-cfgs 3 4 --tag ur5 --epochs 40
    # smoke: ... --max-steps 60 --epochs 1
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_tokenizer.mm_perceiver_temporal import MMPerceiverTemporal  # noqa: E402
from world_tokenizer.window_loader import make_window_loader, unpack   # noqa: E402


def rankme(z):
    """Effective rank of z [N,d] — collapse guard (>1 healthy)."""
    z = np.asarray(z, dtype=np.float64)
    z = z - z.mean(0)
    s = np.linalg.svd(z, compute_uv=False)
    if s.sum() <= 0:
        return 1.0
    p = s / s.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


@torch.no_grad()
def health(model, loader, dev, max_batches=8):
    """Vision-only z_v on held-out windows -> RankMe + mean per-dim std."""
    model.eval()
    zs = []
    for i, b in enumerate(loader):
        zs.append(model.embed_vision(*unpack(b, dev)).float().cpu().numpy())
        if i + 1 >= max_batches:
            break
    if not zs:
        return float("nan"), float("nan")
    z = np.concatenate(zs)
    return rankme(z), float(z.std(0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/mnt/nas/data/RH20T/caches")
    ap.add_argument("--train-cfgs", type=int, nargs="+", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default="/mnt/nas/data/RH20T/checkpoints/temporal")
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0, help="max grad norm (0=off)")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--n-latents", type=int, default=64)
    ap.add_argument("--n-self", type=int, default=4)
    ap.add_argument("--mask-ratio", type=float, default=0.5)
    ap.add_argument("--mask-mode", default="mixed", choices=["mixed", "modality", "future", "cell"],
                    help="cross-modal masking scheme (§18.9); 'cell' is the old weak one")
    ap.add_argument("--raw-target", action="store_true",
                    help="unnormalized MSE target (v0.1-style) instead of L2-normalized cosine (§18.11)")
    ap.add_argument("--pred-mode", default="query", choices=["query", "v01"],
                    help="'query' temporal decoder | 'v01' per-modal MLP on pooled latent (§18.12)")
    ap.add_argument("--joint-sigreg", action="store_true",
                    help="also SIGReg the fused latent (v0.1 had it; stabilizes raw target)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0, help="hard cap (smoke)")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    for _ in range(15):                                       # VM CUDA-init is flaky under load
        if torch.cuda.is_available():
            break
        time.sleep(10)
    assert torch.cuda.is_available(), "no CUDA"
    dev = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_dir = os.path.join(args.out_dir, args.tag)
    os.makedirs(run_dir, exist_ok=True)
    tr, te, ds = make_window_loader(args.cache_dir, tuple(args.train_cfgs), window=args.window,
                                    stride=args.stride, batch=args.batch, num_workers=args.workers)
    print(f"[{args.tag}] cfgs {args.train_cfgs} | {len(ds)} windows | {len(tr)} train batches "
          f"| window {args.window} stride {args.stride} | dev {dev}", flush=True)

    model = MMPerceiverTemporal(d=args.d, n_latents=args.n_latents, n_self=args.n_self,
                                mask_ratio=args.mask_ratio, mask_mode=args.mask_mode,
                                norm_pred=not args.raw_target, pred_mode=args.pred_mode,
                                joint_sigreg=args.joint_sigreg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{args.tag}] trainable params {n_params/1e6:.2f}M", flush=True)

    log = {"args": vars(args), "steps": []}
    step, t0 = 0, time.time()
    for ep in range(args.epochs):
        model.train()
        ep_loss = ep_inv = ep_sig = 0.0
        nb = 0
        for b in tr:
            out = model(*unpack(b, dev))
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            model.update_target()
            step += 1
            ep_loss += float(out["loss"].detach()); ep_inv += float(out["inv"]); ep_sig += float(out["sig"]); nb += 1
            if step % args.log_every == 0 or step == 1:
                print(f"  e{ep} s{step} loss {float(out['loss'].detach()):.4f} "
                      f"inv {float(out['inv']):.4f} sig {float(out['sig']):.2f}", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        rm, es = health(model, te, dev)
        print(f"== e{ep} done | mean loss {ep_loss/max(nb,1):.4f} inv {ep_inv/max(nb,1):.4f} "
              f"sig {ep_sig/max(nb,1):.2f} | z_v RankMe {rm:.1f} std {es:.4f} ==", flush=True)
        log["steps"].append({"epoch": ep, "loss": ep_loss/max(nb,1), "inv": ep_inv/max(nb,1),
                             "sig": ep_sig/max(nb,1), "rankme": rm, "emb_std": es})
        torch.save({"model": model.state_dict(), "args": vars(args), "epoch": ep},
                   os.path.join(run_dir, f"seed{args.seed}.pt"))
        with open(os.path.join(run_dir, f"log_seed{args.seed}.json"), "w") as f:
            json.dump(log, f, indent=1)
        if args.max_steps and step >= args.max_steps:
            break

    print(f"[{args.tag}] DONE in {(time.time()-t0)/60:.1f} min -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
