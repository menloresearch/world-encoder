"""Phase 1 training — continue-LeJEPA on cfg3 video, single GPU (debug) first.

Recipe (locked): warm-start (don't freeze), LR 2e-4, AdamW, BF16, ~30 epochs.
BF16 needs no GradScaler (that's an fp16-only thing).

Debug first:   python -m phase1.train --frames-root <one_scene_frames> --epochs 3 --n-local 0
Scale to DDP:  torchrun --nproc_per_node=7 -m phase1.train ...   (wrap model in DDP, add
               DistributedSampler; or use the stable-pretraining Lightning trainer). GPU 0 is
               busy, so default to the 7 free GPUs via CUDA_VISIBLE_DEVICES.
"""
import argparse

import torch
from torch.utils.data import DataLoader

from phase1.dataset import MultiCropRGB, collate, split_views
from phase1.model import LeJEPAVideo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-global", type=int, default=2)
    ap.add_argument("--n-local", type=int, default=0, help="0 for the first debug run")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=0, help="stop early for a quick smoke run")
    ap.add_argument("--out", default="/mnt/nas/data/RH20T/phase1_ckpt.pt")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = MultiCropRGB(args.frames_root, n_global=args.n_global, n_local=args.n_local)
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=args.num_workers > 0, drop_last=True,
        collate_fn=collate,
    )
    print(f"{len(ds)} frames | {len(dl)} steps/epoch | bs={args.batch_size} "
          f"n_global={args.n_global} n_local={args.n_local}")

    net = LeJEPAVideo(pretrained=True).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    step = 0
    for epoch in range(args.epochs):
        net.train()
        for gs, ls in dl:
            gs = gs.to(dev, non_blocking=True)
            ls = ls.to(dev, non_blocking=True) if ls is not None else None
            global_views, local_views = split_views(gs, ls)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(global_views, local_views)
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            opt.step()

            # collapse guard: std of CLS embeddings across the batch must stay > 0.
            emb_std = out["embedding"].float().std(0).mean().item()
            step += 1
            if step % 10 == 0 or step == 1:
                print(f"e{epoch} s{step} loss={out['loss']:.4f} "
                      f"inv={out['inv_loss']:.4f} sigreg={out['sigreg_loss']:.4f} "
                      f"emb_std={emb_std:.4f}")
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    torch.save({"model": net.state_dict(), "args": vars(args)}, args.out)
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
