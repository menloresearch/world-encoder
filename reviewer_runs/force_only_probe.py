"""Force-only vs pose-only probe, paper-grade — answers reviewer Q4.

Reuses the FROZEN ALL encoders (phase1/all) + the data/compute-matched vision-only
control (phase1_abl/all_vo). NO training. Upgrades the one-off /tmp/force_only.py with:
  (1) per-seed error bars (mean +- std over the 5 seeds),
  (2) a paired significance test (fused vs raw, fused vs vision-only control),
  (3) the vision-only CONTROL comparison (not just raw ViT),
  (4) a pose-partialled probe ("force_resid"): predict the part of force NOT linearly
      explained by TCP pose -> shows force recovery is separate from pose, not leakage.

ee target = masked mean of the 15-dim ee block; force = dims [0:6] (F/T),
pose = dims [6:15] (TCP xyz + 6D rot); ee = full [0:15] (reconciles with paper Table 2).
Sensored robots only (flexiv/ur5/kuka); franka has no F/T.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from world_tokenizer.dataloader import ChunkDataset, load_split
from world_tokenizer.mm_perceiver2 import MMPerceiverChunks, masked_mean
from world_tokenizer.train_chunks import encode_zv, probe_r2

CACHE = "/mnt/nas/data/RH20T/caches"
FUSED = "/mnt/nas/data/RH20T/checkpoints/phase1/all"
VO = "/mnt/nas/data/RH20T/checkpoints/phase1_abl/all_vo"
OUT = "/home/menlo/brain/ishneet/world-encoder/reviewer_runs/force_only_full.json"
ROBOTS = {"flexiv": (1, 2), "ur5": (3, 4), "kuka": (6, 7)}
FORCE, POSE, EE = slice(0, 6), slice(6, 15), slice(0, 15)
SEEDS = 5
dev = "cuda" if torch.cuda.is_available() else "cpu"
split = load_split()


def resid_probe(xtr, xte, ytr_f, yte_f, ytr_p, yte_p):
    """R2 of predicting force AFTER removing the part linearly explained by pose target."""
    sp = StandardScaler().fit(ytr_p)
    rp = Ridge(alpha=10.0).fit(sp.transform(ytr_p), ytr_f)
    rtr = ytr_f - rp.predict(sp.transform(ytr_p))
    rte = yte_f - rp.predict(sp.transform(yte_p))
    return probe_r2(xtr, rtr, xte, rte)


def paired(a, b):
    a, b = np.asarray(a), np.asarray(b)
    t, p = stats.ttest_rel(a, b)
    return {"delta": float((a - b).mean()), "t": float(t), "p": float(p)}


res = {}
for rob, cfgs in ROBOTS.items():
    ds = ChunkDataset(CACHE, cfgs)
    g = [ds.groups[i] for i in ds._group_idx]
    is_te = np.array([split[x] == "test" for x in g])
    d = ds._d
    e_any = d["ee_mask"].any(1)
    y = masked_mean(torch.from_numpy(d["ee"]), torch.from_numpy(d["ee_mask"])).numpy()
    tr, te = ~is_te, is_te
    etr, ete = tr & e_any, te & e_any
    acc = {m: {"force": [], "pose": [], "ee": [], "force_resid": []}
           for m in ("fused", "vo", "raw")}
    for s in range(SEEDS):
        mf = MMPerceiverChunks(d=256, n_queries=8).to(dev)
        mf.load_state_dict(torch.load(f"{FUSED}/seed{s}.pt", map_location=dev))
        mf.eval()
        zf, raw = encode_zv(mf, ds, dev)
        del mf
        torch.cuda.empty_cache()

        mv = MMPerceiverChunks(d=256, n_queries=8, vision_only=True).to(dev)
        mv.load_state_dict(torch.load(f"{VO}/seed{s}.pt", map_location=dev))
        mv.eval()
        zvo, _ = encode_zv(mv, ds, dev)
        del mv
        torch.cuda.empty_cache()

        for m, X in [("fused", zf), ("vo", zvo), ("raw", raw)]:
            acc[m]["force"].append(probe_r2(X[etr], y[etr, FORCE], X[ete], y[ete, FORCE]))
            acc[m]["pose"].append(probe_r2(X[etr], y[etr, POSE], X[ete], y[ete, POSE]))
            acc[m]["ee"].append(probe_r2(X[etr], y[etr, EE], X[ete], y[ete, EE]))
            acc[m]["force_resid"].append(
                resid_probe(X[etr], X[ete], y[etr, FORCE], y[ete, FORCE],
                            y[etr, POSE], y[ete, POSE]))
        print(f"  {rob} seed{s} done", flush=True)

    robres = {m: {k: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                      "seeds": [float(x) for x in v]}
                  for k, v in acc[m].items()} for m in acc}
    robres["sig_force_fused_vs_raw"] = paired(acc["fused"]["force"], acc["raw"]["force"])
    robres["sig_force_fused_vs_vo"] = paired(acc["fused"]["force"], acc["vo"]["force"])
    res[rob] = robres
    del ds
    print(f"{rob}: force fused {robres['fused']['force']['mean']:+.3f} "
          f"vo {robres['vo']['force']['mean']:+.3f} raw {robres['raw']['force']['mean']:+.3f}",
          flush=True)

json.dump(res, open(OUT, "w"), indent=1)

print("\n=== FORCE-only R2 (mean+-std, 5 seeds) — fused vs vision-only control vs raw ===")
hdr = f"{'robot':7} {'fused':>14} {'vo-control':>14} {'raw':>14} {'d_raw(p)':>15} {'d_vo(p)':>15}"
print(hdr)
for r in ROBOTS:
    R = res[r]
    f, v, w = R["fused"]["force"], R["vo"]["force"], R["raw"]["force"]
    sr, sv = R["sig_force_fused_vs_raw"], R["sig_force_fused_vs_vo"]
    print(f"{r:7} {f['mean']:+.3f}+-{f['std']:.3f} {v['mean']:+.3f}+-{v['std']:.3f} "
          f"{w['mean']:+.3f}+-{w['std']:.3f} {sr['delta']:+.3f}(p{sr['p']:.3f}) "
          f"{sv['delta']:+.3f}(p{sv['p']:.3f})")
print("\n=== POSE-only, full-ee, and FORCE_|_POSE (fused vs raw) ===")
for r in ROBOTS:
    R = res[r]
    print(f"{r:7} pose f{R['fused']['pose']['mean']:+.3f}/r{R['raw']['pose']['mean']:+.3f} | "
          f"ee-full f{R['fused']['ee']['mean']:+.3f}/r{R['raw']['ee']['mean']:+.3f} | "
          f"force_|_pose f{R['fused']['force_resid']['mean']:+.3f}/r{R['raw']['force_resid']['mean']:+.3f}")
print("DONE_FORCE_ONLY_FULL", flush=True)
