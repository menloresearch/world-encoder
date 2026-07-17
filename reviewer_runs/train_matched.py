"""Matched-ALL (ALL@quarter) — answers reviewer Q2 ("is ALL's advantage just 4x data?").

Trains the ALL encoder on cfg1-7 but CAPS the training set to N frames (default 32870 =
the mean specialist's train size), same 40 epochs -> same gradient-step budget as an
average specialist. Every seed draws a fresh seeded subsample. Eval is identical to
train_chunks (vision-only z_v probed on every embodiment's held-out groups). If ALL@N
still ties specialists on-diagonal and dominates off-diagonal, the shared-encoder win is
not merely data volume.

Standalone (touches no shared file); reuses eval_embodiment/EMBODIMENTS from train_chunks.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from world_tokenizer.dataloader import ChunkDataset, load_split
from world_tokenizer.mm_perceiver2 import MMPerceiverChunks
from world_tokenizer.train_chunks import EMBODIMENTS, eval_embodiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/mnt/nas/data/RH20T/caches")
    ap.add_argument("--n", type=int, default=32870, help="capped train frames (mean specialist)")
    ap.add_argument("--tag", default="all_quarter")
    ap.add_argument("--out-dir", default="/mnt/nas/data/RH20T/checkpoints/reviewer")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    for _ in range(15):
        if torch.cuda.is_available():
            break
        time.sleep(10)
    if not torch.cuda.is_available():
        print("!! CUDA unavailable — refusing CPU train", flush=True)
        sys.exit(2)
    dev = "cuda"
    run_dir = os.path.join(args.out_dir, args.tag)
    os.makedirs(run_dir, exist_ok=True)
    split = load_split()
    ds = ChunkDataset(args.cache_dir, (1, 2, 3, 4, 5, 6, 7))
    g = [ds.groups[i] for i in ds._group_idx]
    is_te = np.array([split[x] == "test" for x in g])
    train_idx_all = np.flatnonzero(~is_te)
    n = min(args.n, len(train_idx_all))
    print(f"[{args.tag}] full-train {len(train_idx_all)} -> capped N={n} | eval {list(EMBODIMENTS)}",
          flush=True)

    results = {"args": vars(args), "n_used": int(n), "seeds": {}}
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(seed)
        sub = rng.choice(train_idx_all, size=n, replace=False)
        loader = DataLoader(Subset(ds, sub.tolist()), batch_size=args.batch, shuffle=True,
                            num_workers=args.workers, drop_last=True)
        model = MMPerceiverChunks(d=256, n_queries=8).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        t0 = time.time()
        for ep in range(args.epochs):
            model.train()
            el, nb = 0.0, 0
            for batch in loader:
                out = model(*MMPerceiverChunks.unpack(batch, dev))
                opt.zero_grad()
                out["loss"].backward()
                opt.step()
                model.update_target()
                el += float(out["loss"])
                nb += 1
            if ep % 10 == 0 or ep == args.epochs - 1:
                print(f"  seed{seed} ep{ep} loss {el / max(nb, 1):.4f}", flush=True)
        torch.save(model.state_dict(), os.path.join(run_dir, f"seed{seed}.pt"))
        res = {}
        for emb, ecfgs in EMBODIMENTS.items():
            res[emb] = eval_embodiment(model, args.cache_dir, ecfgs, split, dev)
            print(f"  seed{seed} [{emb}] " + " ".join(
                f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in res[emb].items()), flush=True)
        res["train_minutes"] = (time.time() - t0) / 60
        results["seeds"][seed] = res
        with open(os.path.join(run_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
