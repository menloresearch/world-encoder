"""Phase 1 training — continue-LeJEPA (vision-only ViT finetune) on RH20T video.

Three data sources:
  --shards     WebDataset tar shards (fast, split-UNAWARE — shard keys carry no scene id).
  --frames-root  loose .jpg dir (debug; split-unaware).
  --cfgs       HOLDOUT-AWARE: external-cam frames of the TRAIN groups of these cfgs, per
               splits/holdout_v1.csv — the SAME train split the multimodal encoder used, so
               the finetuned ViT is an apples-to-apples vision baseline vs the Perceiver z_v.

Holdout-aware "all" (all 7 cfgs, train split only), DDP on the free GPUs:
  CUDA_VISIBLE_DEVICES=0,1,5 python -m torch.distributed.run --nproc_per_node=3 \
      -m world_tokenizer.train --cfgs 1 2 3 4 5 6 7 --per-scene 30 --epochs 10 --lr 2e-5 \
      --out /mnt/nas/data/RH20T/checkpoints/phase1_vision/vision_all.pt
Smoke:  python -m world_tokenizer.train --cfgs 3 --per-scene 5 --n-local 0 --max-steps 30

Use the venv `python -m torch.distributed.run` (the torchrun binary is base-env). GPU 0 is
often busy -> pin free ones. BF16, no GradScaler. LR ~2e-5 (2e-4 collapses RankMe), AdamW,
backbone NOT frozen.
"""
import argparse
import glob
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from world_tokenizer.dataset import (MultiCropRGB, collate, make_wds_loader, split_frame_paths,
                                      split_views)
from world_tokenizer.model import LeJEPAVideo, LeJEPAVisionHead


def _dist_init():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
        return True, dist.get_rank(), dist.get_world_size(), local_rank
    return False, 0, 1, 0


def _map_stream(args, is_dist):
    """Map-style frames -> (generator over the whole run, steps_per_epoch, total_steps).

    Two path sources: --frames-root (glob a dir, split-unaware) or --cfgs (HOLDOUT-AWARE:
    external-cam frames of the --split groups of these cfgs, per holdout_v1.csv)."""
    if args.cfgs:
        from world_tokenizer.dataloader import load_split
        paths = split_frame_paths(args.frames_base, args.cfgs, load_split(),
                                  want=args.split, per_scene=args.per_scene)
        assert paths, f"no {args.split} frames for cfgs {args.cfgs} under {args.frames_base}"
        ds = MultiCropRGB(paths=paths, n_global=args.n_global, n_local=args.n_local)
    else:
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
    src.add_argument("--frames-root", help="map-style: dir of loose .jpg frames (split-unaware)")
    src.add_argument("--shards", help="WebDataset: dir of .tar shards (+ count.txt)")
    src.add_argument("--cfgs", type=int, nargs="+",
                     help="HOLDOUT-AWARE map-style: train on --split groups of these cfgs "
                          "(external cam) per splits/holdout_v1.csv, from --frames-base")
    ap.add_argument("--frames-base", default="/mnt/nas/data/RH20T/frames",
                    help="root holding cfg1..cfg7 frame dirs (used with --cfgs)")
    ap.add_argument("--split", default="train", choices=["train", "test"],
                    help="which holdout split to train on (--cfgs)")
    ap.add_argument("--per-scene", type=int, default=0,
                    help="--cfgs: cap frames/scene (0=all); bounds NFS small-file IO")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64, help="per-GPU batch size")
    ap.add_argument("--n-global", type=int, default=2)
    ap.add_argument("--n-local", type=int, default=0)
    ap.add_argument("--head-only", "--freeze-backbone", dest="head_only", action="store_true",
                    help="freeze the ViT, train ONLY proj_v (768->d) — the matched baseline to "
                         "the multimodal encoder (LeJEPAVisionHead). Use higher LR (~1e-3).")
    ap.add_argument("--head-dim", type=int, default=256, help="proj_v output dim (--head-only)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--steps-per-epoch", type=int, default=0, help="WDS: 0=auto from count.txt")
    ap.add_argument("--max-steps", type=int, default=0, help="hard cap on total steps (smoke)")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=None,
                    help="random seed (torch/numpy/random); default None = unseeded")
    ap.add_argument("--out", default="/mnt/nas/data/RH20T/checkpoints/phase1_vision/vision_all.pt")
    args = ap.parse_args()

    if args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    is_dist, rank, world, local_rank = _dist_init()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    dev = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    is_main = rank == 0

    def log(*a):
        if is_main:
            print(*a, flush=True)

    if args.head_only:
        net = LeJEPAVisionHead(d=args.head_dim, pretrained=True).to(dev)
    else:
        net = LeJEPAVideo(pretrained=True).to(dev)
    if is_dist:
        if not args.head_only:  # head-only has no trainable BN (backbone frozen) -> skip convert
            net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = DDP(net, device_ids=[local_rank])  # frozen backbone params carry no grad
    raw = net.module if is_dist else net
    trainable = [p for p in net.parameters() if p.requires_grad]  # only proj_v when head-only
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    if args.shards:
        stream, spe, total_steps, n = _wds_stream(args, world, rank)
        src_desc = f"WDS {n} samples"
    else:
        stream, spe, total_steps, n = _map_stream(args, is_dist)
        src_desc = f"{n} frames"
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps) if not args.shards else args.max_steps
    mode = f"HEAD-ONLY d={args.head_dim} ({sum(p.numel() for p in trainable)/1e3:.0f}k params)" \
        if args.head_only else "FULL-FINETUNE"
    log(f"{mode} | {src_desc} | world={world} | bs/gpu={args.batch_size} "
        f"(global={world * args.batch_size}) | {spe} steps/epoch | {total_steps} total steps "
        f"| n_global={args.n_global} n_local={args.n_local}")

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
