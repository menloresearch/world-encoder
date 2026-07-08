"""Pack extracted frames into WebDataset tar shards for fast SEQUENTIAL NFS reads.

Training off ~2.3M tiny jpgs over NFS is IO-bound (random small-file reads). Packing
them into ~10k-image tar shards turns that into large sequential reads. One-time cost.

    python -m preprocessing.make_shards --num-workers 32
"""
import argparse
import glob
import os
from multiprocessing import Pool

import webdataset as wds

DEF_FRAMES = "/mnt/nas/data/RH20T/frames/cfg3"
DEF_OUT = "/mnt/nas/data/RH20T/shards/cfg3"


def _shard_chunk(wid, paths, outdir, maxcount):
    """One worker writes its slice of frames into shard-<wid>-%05d.tar."""
    pattern = os.path.join(outdir, f"shard-{wid:03d}-%05d.tar")
    sink = wds.ShardWriter(pattern, maxcount=maxcount, verbose=0)
    n = 0
    for p in paths:
        try:
            with open(p, "rb") as f:
                raw = f.read()
            sink.write({"__key__": f"{wid:03d}{n:08d}", "jpg": raw})  # raw bytes; decode on read
            n += 1
        except Exception:
            pass
    sink.close()
    return wid, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-root", default=DEF_FRAMES)
    ap.add_argument("--out", default=DEF_OUT)
    ap.add_argument("--num-workers", type=int, default=32)
    ap.add_argument("--maxcount", type=int, default=10000)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print("globbing frames (one-time)...", flush=True)
    paths = sorted(glob.glob(os.path.join(args.frames_root, "**", "*.jpg"), recursive=True))
    print(f"{len(paths)} frames -> sharding with {args.num_workers} workers", flush=True)
    chunks = [paths[i::args.num_workers] for i in range(args.num_workers)]  # strided = balanced
    tasks = [(wid, chunks[wid], args.out, args.maxcount) for wid in range(args.num_workers)]

    total = 0
    with Pool(args.num_workers) as pool:
        for wid, n in pool.starmap(_shard_chunk, tasks):
            total += n
    nshards = len(glob.glob(os.path.join(args.out, "*.tar")))
    with open(os.path.join(args.out, "count.txt"), "w") as f:
        f.write(str(total))  # train.py reads this for steps/epoch
    print(f"SHARDS DONE: {total} samples, {nshards} shards -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
