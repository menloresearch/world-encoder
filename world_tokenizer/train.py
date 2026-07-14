"""Phase 1 training — continue-LeJEPA on cfg3 video.

Map-style (debug): python -m world_tokenizer.train --frames-root <dir> --epochs 3 --n-local 0 --max-steps 30
WebDataset (full): CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m torch.distributed.run \
                       --nproc_per_node=7 -m world_tokenizer.train --shards /mnt/nas/data/RH20T/shards/cfg3 --epochs 30

Use the venv `python -m torch.distributed.run` (the torchrun binary is base-env). GPU 0 is
busy -> pin 1..7. BF16, no GradScaler. LR 2e-4, AdamW, backbone NOT frozen.
"""
import argparse
import glob
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from world_tokenizer.dataset import MultiCropRGB, collate, make_wds_loader, split_views
from world_tokenizer.model import LeJEPAVideo


def _dist_init():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
        return True, dist.get_rank(), dist.get_world_size(), local_rank
    return False, 0, 1, 0


def _map_stream(args, is_dist):
    """Map-style frames -> (generator over the whole run, steps_per_epoch, total_steps)."""
    ds = MultiCropRGB(args.frames_root, n_global=args.n_global, n_local=args.n_local)
    sampler = DistributedSampler(ds, shuffle=True, drop_last=True) if is_dist else None
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
                    num_workers=args.num_workers, pin_memory=True,
                    persistent_workers=args.num_workers > 0, drop_last=True, collate_fn=collate)
    spe = len(dl)

    def gen():
        for epoch in range(args.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            yield from dl

    return gen(), spe, args.epochs * spe, len(ds)


def _wds_stream(args, world, rank):
    """WebDataset shards -> (infinite generator, steps_per_epoch, total_steps)."""
    shards = sorted(glob.glob(os.path.join(args.shards, "*.tar")))
    assert shards, f"no .tar shards under {args.shards}"
    total = int(open(os.path.join(args.shards, "count.txt")).read())
    spe = args.steps_per_epoch or max(1, total // (world * args.batch_size))
    loader = make_wds_loader(shards, n_global=args.n_global, n_local=args.n_local,
                             batch_size=args.batch_size, num_workers=args.num_workers,
                             seed=1000 + rank)  # per-rank seed -> different data per rank

    def gen():
        yield from loader  # infinite (resampled); caller stops at total_steps

    return gen(), spe, args.epochs * spe, total


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--frames-root", help="map-style: dir of loose .jpg frames")
    src.add_argument("--shards", help="WebDataset: dir of .tar shards (+ count.txt)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64, help="per-GPU batch size")
    ap.add_argument("--n-global", type=int, default=2)
    ap.add_argument("--n-local", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--steps-per-epoch", type=int, default=0, help="WDS: 0=auto from count.txt")
    ap.add_argument("--max-steps", type=int, default=0, help="hard cap on total steps (smoke)")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=None,
                    help="random seed (torch/numpy/random); default None = unseeded")
    ap.add_argument("--out", default="/mnt/nas/data/RH20T/checkpoints/phase1_ckpt.pt")
    args = ap.parse_args()

    if args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    is_dist, rank, world, local_rank = _dist_init()
    dev = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    is_main = rank == 0

    def log(*a):
        if is_main:
            print(*a, flush=True)

    net = LeJEPAVideo(pretrained=True).to(dev)
    if is_dist:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = DDP(net, device_ids=[local_rank])  # all params get grad -> no find_unused
    raw = net.module if is_dist else net
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.shards:
        stream, spe, total_steps, n = _wds_stream(args, world, rank)
        src_desc = f"WDS {n} samples"
    else:
        stream, spe, total_steps, n = _map_stream(args, is_dist)
        src_desc = f"{n} frames"
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps) if not args.shards else args.max_steps
    log(f"{src_desc} | world={world} | bs/gpu={args.batch_size} (global={world * args.batch_size}) "
        f"| {spe} steps/epoch | {total_steps} total steps | n_global={args.n_global} n_local={args.n_local}")

    net.train()
    step = 0
    for gs, ls in stream:
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
            log(f"e{step // spe} s{step}/{total_steps} loss={out['loss']:.4f} "
                f"inv={out['inv_loss']:.4f} sigreg={out['sigreg_loss']:.4f} emb_std={emb_std:.4f}")
        if is_main and step % spe == 0:  # epoch-tagged checkpoint (for probing the plateau)
            ep = step // spe
            torch.save({"model": raw.state_dict(), "args": vars(args), "step": step, "epoch": ep},
                       args.out.replace(".pt", f"_e{ep}.pt"))
            log(f"== EPOCH {ep}/{args.epochs} done @ step {step}, saved _e{ep}.pt ==")
        if step >= total_steps:
            break

    if is_main:
        torch.save({"model": raw.state_dict(), "args": vars(args), "step": step}, args.out)
        log("saved ->", args.out)
    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
