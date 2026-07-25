"""View-composition analysis for the multi-cam encoder (JQ's Slack ask, 2026-07-21).

Questions: (1) does the fused embedding CONVERGE to the full-4-view embedding as views
are added? (2) do different COMPOSITIONS of the same number of views give the same or
different embeddings (view-composition invariance)? (3) how does early fusion (ours)
compare to the late-fusion baseline JQ proposed = mean of per-view v0.1 embeddings?

Metrics on held-out kuka chunks (cosine distance):
  - d(z_subset, z_full4) by subset size 1/2/3  -> convergence with view count
  - mean pairwise d between different same-size compositions of the SAME instant,
    referenced against d between DIFFERENT instants (the scale that matters):
    ratio << 1 -> embedding is mostly scene-determined (composition-invariant);
    ratio ~ 1  -> each composition is basically a different embedding.

    python -m world_tokenizer.multicam_composition
"""
import itertools
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_tokenizer.dataloader import ChunkDataset, load_split  # noqa: E402
from world_tokenizer.mm_perceiver2 import MMPerceiverChunks as V01  # noqa: E402
from world_tokenizer.mm_perceiver3 import MMPerceiverChunks as MC  # noqa: E402


def cosd(a, b):
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    return 1 - (a * b).sum(-1)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",
                    default="/mnt/nas/data/RH20T/checkpoints/multicam/kuka_mc4/seed0.pt")
    ap.add_argument("--out", default="results/temporal/multicam_composition.json")
    args = ap.parse_args()

    dev = "cuda"
    ds = ChunkDataset("/mnt/nas/data/RH20T/caches", (6, 7), multicam=True)
    split = load_split()
    is_test = np.array([split[ds.groups[g]] == "test" for g in ds._group_idx])
    idx = np.flatnonzero(is_test)[:2000]
    d = ds._d

    mc = MC(n_cam_ids=len(ds.cam_vocab)).to(dev)
    mc.load_state_dict(torch.load(args.ckpt, map_location=dev))
    v01 = V01().to(dev)
    v01.load_state_dict(torch.load(
        "/mnt/nas/data/RH20T/checkpoints/phase1/kuka/seed0.pt", map_location=dev))
    mc.eval(), v01.eval()

    subsets = [s for r in (1, 2, 3) for s in itertools.combinations(range(4), r)]
    Z = {s: [] for s in subsets}           # mc embeddings per view-subset
    Zf, Vv = [], []                        # mc full-4; v0.1 per-view stack
    bs = 128
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            sl = idx[i:i + bs]
            rgb = torch.from_numpy(d["patch"][sl].astype(np.float32)).to(dev)
            cam_ids = torch.from_numpy(ds._cam_ids[sl]).to(dev)
            motor = torch.from_numpy(d["motor"][sl]).squeeze(1).to(dev)
            m_mask = torch.from_numpy(d["motor_mask"][sl]).to(dev)
            ee = torch.from_numpy(d["ee"][sl]).to(dev)
            e_mask = torch.from_numpy(d["ee_mask"][sl]).to(dev)
            cams = list(rgb.unbind(1))
            Zf.append(mc.embed_vision(cams, motor, m_mask, ee, e_mask, cam_ids).cpu())
            for s in subsets:
                Z[s].append(mc.embed_vision([cams[k] for k in s], motor, m_mask, ee,
                                            e_mask, cam_ids[:, list(s)]).cpu())
            Vv.append(torch.stack([v01.embed_vision(c, motor, m_mask, ee, e_mask).cpu()
                                   for c in cams], dim=1))
    Zf = torch.cat(Zf)
    Z = {s: torch.cat(v) for s, v in Z.items()}
    Vv = torch.cat(Vv)                     # [N,4,256]

    res = {"ckpt": args.ckpt, "cam_vocab": ds.cam_vocab}
    # 1. convergence to full-view embedding as views are added (ours)
    for r in (1, 2, 3):
        ds_r = torch.stack([cosd(Z[s], Zf) for s in subsets if len(s) == r])
        res[f"mc_d_to_full_{r}view"] = float(ds_r.mean())
    # 1b. per-camera breakdown (JQ #3): subset-vs-full conditioned on WHICH camera is
    # the singleton; + all distances also in units of the model's own cross-moment
    # distance (thread convention, so v0.1-vs-v0.2 scales compare apples-to-apples)
    cross = float(cosd(Zf, Zf.roll(1, 0)).mean())
    res["mc_d_cross_instant_full"] = cross
    for s in subsets:
        if len(s) == 1:
            dk = float(cosd(Z[s], Zf).mean())
            res[f"mc_d_to_full_cam{s[0]}"] = dk
            res[f"mc_d_to_full_cam{s[0]}_norm"] = dk / cross
    for r in (1, 2, 3):
        res[f"mc_d_to_full_{r}view_norm"] = res[f"mc_d_to_full_{r}view"] / cross
    # late-fusion baseline: mean of v0.1 per-view embs, subset vs all-4
    Vf = Vv.mean(1)
    for r in (1, 2, 3):
        ds_r = torch.stack([cosd(Vv[:, list(s)].mean(1), Vf)
                            for s in subsets if len(s) == r])
        res[f"v01mean_d_to_full_{r}view"] = float(ds_r.mean())
    # 2. composition sensitivity at fixed size 2: same instant, different pairs
    pairs = [s for s in subsets if len(s) == 2]
    for name, embs in (("mc", Z), ("v01mean", {s: Vv[:, list(s)].mean(1) for s in pairs})):
        pp = torch.stack([cosd(embs[a], embs[b])
                          for a, b in itertools.combinations(pairs, 2)]).mean()
        # reference: same representation, DIFFERENT instant (shift by 1 within set)
        ref = torch.stack([cosd(embs[p], embs[p].roll(1, 0)) for p in pairs]).mean()
        res[f"{name}_d_composition_2view"] = float(pp)
        res[f"{name}_d_cross_instant_2view"] = float(ref)
        res[f"{name}_composition_over_instant"] = float(pp / ref)

    for k, v in res.items():
        print(f"{k:36s} {v:.4f}" if isinstance(v, float) else f"{k:36s} {v}", flush=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"SAVED {args.out}", flush=True)


if __name__ == "__main__":
    main()
