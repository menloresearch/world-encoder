"""NH1 gate eval (TEMPORAL_ARCH.md §18.2 / §19.1 P3+P4): does temporal (v0.2) beat single-timestep
(v0.1) on things a single frame structurally can't do?

All representations are VISION-ONLY (the thesis: what vision alone can recover). Head-to-head:
  - v1  = temporal window latent (MMPerceiverTemporal.embed_vision over a window)
  - v0.1 = single-frame latent (mm_perceiver2.MMPerceiverChunks.embed_vision on one tick)
  - naive = carry-forward (predict future = present) — for P4 only

Tests (ridge probe R2, fit on train windows, eval on test; higher=better):
  P3  DYNAMICS (full window, target at last tick):
      dq   = joint velocity (symlog dq, motor ch2)          <- single frame has ZERO velocity info
      dFdt = force-rate = force(last) - force(last-1)        <- ditto
  P4  FUTURE FORCE (context = first Cc ticks, target = force at tick Cc-1+delta):
      for delta in --deltas ; v1 encodes the context sub-window, v0.1 encodes the last context tick.

Reject NH1 if v1 >> v0.1 on P3 (v0.1 ~0 by construction) and v1 > v0.1 & naive on P4 with the gap
growing in delta.

    python -m world_tokenizer.gate_eval --v1 .../temporal/ur5b/seed0.pt \
        --v0 .../phase1/ur5/seed0.pt --cfgs 3 4
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_tokenizer.mm_perceiver_temporal import MMPerceiverTemporal, masked_mean  # noqa: E402
from world_tokenizer.mm_perceiver2 import MMPerceiverChunks                          # noqa: E402
from world_tokenizer.train_chunks import probe_r2, rankme                            # noqa: E402
from world_tokenizer.window_loader import make_window_loader                         # noqa: E402


@torch.no_grad()
def encode(v1, v0, loader, dev, Cc, deltas):
    """Per window: v1 full-window z, v1 context z, v0.1 last-frame z, v0.1 context-last z,
    and targets (dq, dFdt, present force, future force @ each delta) with validity."""
    v1.eval(); v0.eval()
    out = {k: [] for k in ("z1_full", "z1_ctx", "z0_last", "z0_ctxlast",
                           "dq", "dq_m", "dfdt", "dfdt_ok", "f_now", "f_now_ok")}
    for d in deltas:
        out[f"f_fut{d}"] = []; out[f"f_fut{d}_ok"] = []
    for b in loader:
        rgb = b["rgb"].to(dev); motor = b["motor"].to(dev); mm = b["motor_mask"].to(dev)
        ee = b["ee"].to(dev); em = b["ee_mask"].to(dev); t = b["t_ms"].to(dev)
        B, C = rgb.shape[0], rgb.shape[1]
        last, cl = C - 1, Cc - 1
        # --- representations (vision-only) ---
        out["z1_full"].append(v1.embed_vision(rgb, motor, mm, ee, em, t).float().cpu().numpy())
        out["z1_ctx"].append(v1.embed_vision(rgb[:, :Cc], motor[:, :Cc], mm[:, :Cc],
                                             ee[:, :Cc], em[:, :Cc], t[:, :Cc]).float().cpu().numpy())
        out["z0_last"].append(v0.embed_vision(rgb[:, last], motor[:, last], mm[:, last],
                                              ee[:, last], em[:, last]).float().cpu().numpy())
        out["z0_ctxlast"].append(v0.embed_vision(rgb[:, cl], motor[:, cl], mm[:, cl],
                                                 ee[:, cl], em[:, cl]).float().cpu().numpy())
        # --- P3 targets ---
        out["dq"].append(motor[:, last, :, 2].cpu().numpy())                 # [B,8] symlog dq
        out["dq_m"].append(mm[:, last, :, 2].cpu().numpy())                  # [B,8] valid
        fL = masked_mean(ee[:, last, :, :6], em[:, last])                    # [B,6] force @ last
        fLm1 = masked_mean(ee[:, last - 1, :, :6], em[:, last - 1])
        okd = (em[:, last].any(-1) & em[:, last - 1].any(-1))
        out["dfdt"].append((fL - fLm1).cpu().numpy()); out["dfdt_ok"].append(okd.cpu().numpy())
        # --- P4 targets (force) ---
        out["f_now"].append(masked_mean(ee[:, cl, :, :6], em[:, cl]).cpu().numpy())
        out["f_now_ok"].append(em[:, cl].any(-1).cpu().numpy())
        for dl in deltas:
            tk = cl + dl
            out[f"f_fut{dl}"].append(masked_mean(ee[:, tk, :, :6], em[:, tk]).cpu().numpy())
            out[f"f_fut{dl}_ok"].append(em[:, tk].any(-1).cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", required=True, help="temporal (v0.2) checkpoint")
    ap.add_argument("--v0", required=True, help="v0.1 single-tick checkpoint")
    ap.add_argument("--cache-dir", default="/mnt/nas/data/RH20T/caches")
    ap.add_argument("--cfgs", type=int, nargs="+", required=True)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=5, help="context ticks for P4 future prediction")
    ap.add_argument("--deltas", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    dev = "cuda"

    ck1 = torch.load(args.v1, map_location=dev); a = ck1["args"]
    v1 = MMPerceiverTemporal(d=a["d"], n_latents=a["n_latents"], n_self=a["n_self"]).to(dev)
    v1.load_state_dict(ck1["model"])
    v0 = MMPerceiverChunks(d=256, n_queries=8).to(dev)
    v0.load_state_dict(torch.load(args.v0, map_location=dev))
    print(f"v1={args.v1} (ep {ck1.get('epoch')})  |  v0.1={args.v0}", flush=True)

    tr, te, _ = make_window_loader(args.cache_dir, tuple(args.cfgs), window=args.window,
                                   stride=args.stride, batch=128, num_workers=4)
    Tr = encode(v1, v0, tr, dev, args.ctx, args.deltas)
    Te = encode(v1, v0, te, dev, args.ctx, args.deltas)
    print(f"windows train {len(Tr['z1_full'])} test {len(Te['z1_full'])}", flush=True)

    res = {"rankme_v1": rankme(Te["z1_full"]), "rankme_v0": rankme(Te["z0_last"])}

    # ---- P3 dynamics ----
    mv = Tr["dq_m"].all(0)                                   # globally-valid velocity dims
    def probe(xtr, ytr, xte, yte, ok_tr=None, ok_te=None):
        if ok_tr is not None:
            xtr, ytr = xtr[ok_tr], ytr[ok_tr]; xte, yte = xte[ok_te], yte[ok_te]
        if len(xtr) < 50 or len(xte) < 50:
            return float("nan")
        return probe_r2(xtr, ytr, xte, yte)
    res["P3_dq_v1"] = probe(Tr["z1_full"], Tr["dq"][:, mv], Te["z1_full"], Te["dq"][:, mv])
    res["P3_dq_v0"] = probe(Tr["z0_last"], Tr["dq"][:, mv], Te["z0_last"], Te["dq"][:, mv])
    res["P3_dFdt_v1"] = probe(Tr["z1_full"], Tr["dfdt"], Te["z1_full"], Te["dfdt"],
                              Tr["dfdt_ok"], Te["dfdt_ok"])
    res["P3_dFdt_v0"] = probe(Tr["z0_last"], Tr["dfdt"], Te["z0_last"], Te["dfdt"],
                              Tr["dfdt_ok"], Te["dfdt_ok"])

    # ---- P4 future force ----
    for dl in args.deltas:
        ok_tr, ok_te = Tr[f"f_fut{dl}_ok"], Te[f"f_fut{dl}_ok"]
        y_tr, y_te = Tr[f"f_fut{dl}"], Te[f"f_fut{dl}"]
        res[f"P4_f{dl}_v1"] = probe(Tr["z1_ctx"], y_tr, Te["z1_ctx"], y_te, ok_tr, ok_te)
        res[f"P4_f{dl}_v0"] = probe(Tr["z0_ctxlast"], y_tr, Te["z0_ctxlast"], y_te, ok_tr, ok_te)
        # naive: predict future force = present force (carry-forward), eval on test only
        nok = ok_te & Te["f_now_ok"]
        from sklearn.metrics import r2_score
        res[f"P4_f{dl}_naive"] = (float(r2_score(y_te[nok], Te["f_now"][nok],
                                  multioutput="uniform_average")) if nok.sum() > 50 else float("nan"))

    print("\n=== NH1 GATE (vision-only; reject NH1 if v1 beats v0.1 / naive) ===", flush=True)
    print(f"  RankMe: v1 {res['rankme_v1']:.1f}  v0.1 {res['rankme_v0']:.1f}", flush=True)
    print(f"  P3 dq    R2:  v1 {res['P3_dq_v1']:.3f}   v0.1 {res['P3_dq_v0']:.3f}", flush=True)
    print(f"  P3 dF/dt R2:  v1 {res['P3_dFdt_v1']:.3f}   v0.1 {res['P3_dFdt_v0']:.3f}", flush=True)
    for dl in args.deltas:
        print(f"  P4 force @Δ{dl} R2:  v1 {res[f'P4_f{dl}_v1']:.3f}   v0.1 {res[f'P4_f{dl}_v0']:.3f}"
              f"   naive {res[f'P4_f{dl}_naive']:.3f}", flush=True)

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
        print("saved ->", args.out, flush=True)


if __name__ == "__main__":
    main()
