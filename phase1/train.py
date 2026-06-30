"""Phase 1 training — continue-LeJEPA on cfg3 video.

Single GPU (debug): python -m phase1.train --frames-root <dir> --epochs 3 --n-local 0 --max-steps 30
DDP (full run):     CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node=7 \
                        -m phase1.train --frames-root <dir> --epochs 30

GPU 0 is busy, so pin the free GPUs via CUDA_VISIBLE_DEVICES. BF16, no GradScaler
(fp16-only). LR 2e-4 per-recipe; global batch = world_size * --batch-size, so consider
scaling LR if you push world size up. Backbone is NOT frozen.
"""
import argparse
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from phase1.dataset import MultiCropRGB, collate, split_views
from phase1.model import LeJEPAVideo


def _dist_init():
    """Init process group if launched by torchrun; else single-process."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
        return True, dist.get_rank(), dist.get_world_size(), local_rank
    return False, 0, 1, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64, help="per-GPU batch size")
    ap.add_argument("--n-global", type=int, default=2)
    ap.add_argument("--n-local", type=int, default=0, help="0 for the first debug run")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (smoke runs)")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--out", default="/mnt/nas/data/RH20T/phase1_ckpt.pt")
    args = ap.parse_args()

    is_dist, rank, world, local_rank = _dist_init()
    dev = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    is_main = rank == 0

    def log(*a):
        if is_main:
            print(*a, flush=True)

    ds = MultiCropRGB(args.frames_root, n_global=args.n_global, n_local=args.n_local)
    sampler = DistributedSampler(ds, shuffle=True, drop_last=True) if is_dist else None
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0, drop_last=True, collate_fn=collate,
    )
    log(f"{len(ds)} frames | world={world} | {len(dl)} steps/epoch/gpu | "
        f"bs/gpu={args.batch_size} (global={world * args.batch_size}) "
        f"n_global={args.n_global} n_local={args.n_local}")

    net = LeJEPAVideo(pretrained=True).to(dev)
    if is_dist:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        # DDP smoke confirmed all params receive grad -> no find_unused_parameters needed.
        net = DDP(net, device_ids=[local_rank])
    raw = net.module if is_dist else net
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    step = 0
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        net.train()
        for gs, ls in dl:
            gs = gs.to(dev, non_blocking=True)
            ls = ls.to(dev, non_blocking=True) if ls is not None else None
            gv, lv = split_views(gs, ls)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(gv, lv)
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            opt.step()
            step += 1
            if is_main and (step % args.log_every == 0 or step == 1):
                emb_std = out["embedding"].float().std(0).mean().item()  # collapse guard (>0)
                log(f"e{epoch} s{step} loss={out['loss']:.4f} inv={out['inv_loss']:.4f} "
                    f"sigreg={out['sigreg_loss']:.4f} emb_std={emb_std:.4f}")
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    if is_main:
        torch.save({"model": raw.state_dict(), "args": vars(args)}, args.out)
        log("saved ->", args.out)
    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
