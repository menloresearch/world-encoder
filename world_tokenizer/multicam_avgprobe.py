"""JQ follow-up #1 (V0.2.md): single-view-encode + average probe for the mc encoder.

Run the mc encoder once per camera (single-view input), average the K embeddings,
run the same force/pose/motor probes as the Build-1 gate. Separates "fusion does
real work" from "just a better backbone" — the v0.2 analogue of the mean-of-v0.1
late-fusion baseline. Inference-only on the existing kuka cache.

Caveat to attach (V0.2.md): the pre-dropout ckpt is OOD on single views (1-view
distance 0.476 > cross-moment ~0.30), so averaging degraded embeddings UNDERSTATES
fusion; this is the baseline row — rerun with --ckpt <dropout ckpt> for the real answer.

    python -m world_tokenizer.multicam_avgprobe                       # pre-dropout ckpt
    python -m world_tokenizer.multicam_avgprobe --ckpt ... --out ...  # post-dropout
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_tokenizer.dataloader import ChunkDataset, load_split  # noqa: E402
from world_tokenizer.mm_perceiver3 import MMPerceiverChunks, masked_mean  # noqa: E402
from world_tokenizer.train_chunks import probe_r2, rankme  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/mnt/nas/data/RH20T/caches")
    ap.add_argument("--train-cfgs", type=int, nargs="+", default=[6, 7])
    ap.add_argument("--ckpt",
                    default="/mnt/nas/data/RH20T/checkpoints/multicam/kuka_mc4/seed0.pt")
    ap.add_argument("--out", default="results/temporal/multicam_avgprobe.json")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    dev = "cuda"
    ds = ChunkDataset(args.cache_dir, tuple(args.train_cfgs), multicam=True)
    split = load_split()
    is_test = np.array([split[ds.groups[g]] == "test" for g in ds._group_idx])
    d, n, K = ds._d, len(ds), ds._cam_ids.shape[1]

    mc = MMPerceiverChunks(n_cam_ids=len(ds.cam_vocab)).to(dev)
    mc.load_state_dict(torch.load(args.ckpt, map_location=dev))
    mc.eval()

    # mc4 = fused all-K (recomputed here so both arms share the exact split/ckpt);
    # mc1avg = encode each camera alone, average the K embeddings (JQ's arm);
    # per-cam singles kept for the per-slot readout.
    Z = {"mc4": [], "mc1avg": []}
    Zs = [[] for _ in range(K)]
    with torch.no_grad():
        for i in tqdm(range(0, n, args.batch), desc="encode", mininterval=10):
            sl = slice(i, min(i + args.batch, n))
            rgb = torch.from_numpy(d["patch"][sl].astype(np.float32)).to(dev)
            cams = list(rgb.unbind(1))
            cam_ids = torch.from_numpy(ds._cam_ids[sl]).to(dev)
            motor = torch.from_numpy(d["motor"][sl]).squeeze(1).to(dev)
            m_mask = torch.from_numpy(d["motor_mask"][sl]).to(dev)
            ee = torch.from_numpy(d["ee"][sl]).to(dev)
            e_mask = torch.from_numpy(d["ee_mask"][sl]).to(dev)
            Z["mc4"].append(mc.embed_vision(cams, motor, m_mask, ee, e_mask,
                                            cam_ids).cpu().numpy())
            singles = [mc.embed_vision([cams[k]], motor, m_mask, ee, e_mask,
                                       cam_ids[:, [k]]).cpu().numpy() for k in range(K)]
            for k in range(K):
                Zs[k].append(singles[k])
            Z["mc1avg"].append(np.mean(singles, axis=0))
    Z = {k: np.concatenate(v) for k, v in Z.items()}
    for k in range(K):
        Z[f"mc1_cam{k}"] = np.concatenate(Zs[k])

    motor = d["motor"].reshape(n, -1)
    mvalid = d["motor_mask"].reshape(n, -1).all(0)
    y_motor = motor[:, mvalid]
    e_any = d["ee_mask"].any(1)
    y_ee = masked_mean(torch.from_numpy(d["ee"]),
                       torch.from_numpy(d["ee_mask"])).numpy()

    tr, te = ~is_test, is_test
    res = {"ckpt": args.ckpt, "n_test": int(te.sum())}
    for name, z in Z.items():
        res[f"{name}_rankme"] = rankme(z[te])
        res[f"{name}_r2_motor"] = probe_r2(z[tr], y_motor[tr], z[te], y_motor[te])
        etr, ete = tr & e_any, te & e_any
        if etr.sum() > 50 and ete.sum() > 50:
            res[f"{name}_r2_ee"] = probe_r2(z[etr], y_ee[etr], z[ete], y_ee[ete])
    # distances in units of the model's own cross-moment distance (thread convention)
    zf = torch.from_numpy(Z["mc4"][te]).float()
    za = torch.from_numpy(Z["mc1avg"][te]).float()
    cosd = lambda a, b: 1 - torch.nn.functional.cosine_similarity(a, b, dim=-1)  # noqa: E731
    cross = float(cosd(zf, zf.roll(1, 0)).mean())
    res["d_avg_to_full"] = float(cosd(za, zf).mean())
    res["d_cross_instant_full"] = cross
    res["d_avg_to_full_normalized"] = res["d_avg_to_full"] / cross

    for k, v in res.items():
        print(f"{k:28s} {v:.4f}" if isinstance(v, float) else f"{k:28s} {v}", flush=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"SAVED {args.out}", flush=True)


if __name__ == "__main__":
    main()
