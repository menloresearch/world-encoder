"""v0.4 correctness gate: at alpha=0, MMPerceiverMCStateAdd == v0.3 unpooled encoder.

Loads the REAL v0.3 checkpoint into both the v0.3 class (MMPerceiverMCPatch, strict)
and the v0.4 class (strict=False; new v0.4 params keep their init, alpha stays 0), then
on real cache rows checks:

  T1  alpha=0 equivalence — v04.embed_set(p,s) vs v0.3 fuse(ctx, mask_state, pool=False)
      (the exact set forward_set()/the unpool policy consumes). PASS: max|diff| < 1e-6.
  T2  state invariance at alpha=0 — perturbing state must not move the output at all.
  T3  predictor isolation — the 8 output-query results are identical whether or not the
      16 predictor queries are run alongside them (no query cross-talk). tol 1e-5.
  T4  gate opening — with alpha=0.5, conditioned queries 0..3 move, queries 4..7 must
      stay EXACTLY fixed (4/4 split mask), and state sensitivity appears only on 0..3.

    cd /home/menlo/brain/ishneet/world-encoder && \
    CUDA_VISIBLE_DEVICES=0 /home/menlo/brain/ishneet/robocasa/.venv/bin/python3 \
      -m world_tokenizer.v04_equiv_test
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_tokenizer.robocasa_pretrain_v03 import MMPerceiverMCPatch  # noqa: E402
from world_tokenizer.robocasa_pretrain_v04 import MMPerceiverMCStateAdd  # noqa: E402

NEW_PREFIXES = ("state_mlp.", "alpha", "cond_qmask", "null_u", "pred_q",
                "pred_state.", "pred_act.", "pred_head.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",
                    default="/mnt/nas/data/robocasa/kepler_ckpt/pnp_mc3_v03_lp2.0/seed0.pt")
    ap.add_argument("--cache",
                    default="/mnt/nas/data/robocasa/kepler_cache/pnp_patches.npz")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    res = json.load(open(os.path.join(os.path.dirname(args.ckpt), "results.json")))
    a = res["args"]
    lp = res.get("lamb_patch", a["lamb_patch"] if not isinstance(a["lamb_patch"], list)
                 else a["lamb_patch"][0])
    mu = np.array(res["state_mu"], np.float32)
    sd_ = np.array(res["state_sd"], np.float32)

    d = np.load(args.cache)
    rows = np.linspace(0, len(d["ep"]) - 1, args.n).astype(int)
    patch = d["patch"][rows].astype(np.float32)
    state = (d["state"].astype(np.float32)[rows] - mu) / sd_
    K, state_dim = patch.shape[1], state.shape[1]
    dev = args.device
    p = torch.from_numpy(patch).to(dev)
    s = torch.from_numpy(state).to(dev)

    sd = torch.load(args.ckpt, map_location="cpu")
    kw = dict(n_cams=K, d=a["d"], state_dim=state_dim, n_queries=a["queries"],
              lamb_patch=lp, mask_ratio=a["mask_ratio"])
    ref = MMPerceiverMCPatch(**kw).to(dev).eval()
    ref.load_state_dict(sd, strict=True)
    v04 = MMPerceiverMCStateAdd(act_len=5, act_dim=12, n_pred=16, **kw).to(dev).eval()
    missing, unexpected = v04.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected}"
    bad = [m for m in missing if not m.startswith(NEW_PREFIXES)]
    assert not bad, f"missing non-v0.4 keys: {bad}"
    print(f"ckpt {args.ckpt}\nloaded: strict into v0.3; strict=False into v0.4 "
          f"({len(missing)} new v0.4 params uninitialized-from-ckpt, alpha=0)")
    print(f"inputs: {args.n} real cache rows, fp32, {dev}\n")

    ok = True
    with torch.no_grad():
        # T1: alpha=0 equivalence against the v0.3 unpooled vision-only set
        ref_set = ref.fuse(ref._context(p, s), ref._mask(True, dev), pool=False)
        v04_set = v04.embed_set(p, s)
        d1 = float((ref_set - v04_set).abs().max())
        t1 = d1 < 1e-6
        ok &= t1
        print(f"T1 alpha=0 equivalence:   max|diff| = {d1:.3e}  "
              f"{'PASS' if t1 else 'FAIL'} (tol 1e-6)")

        # T2: state invariance at alpha=0
        s_pert = s + torch.randn_like(s) * 5.0
        d2 = float((v04.embed_set(p, s_pert) - v04_set).abs().max())
        t2 = d2 < 1e-6
        ok &= t2
        print(f"T2 state invariance @a=0: max|diff| = {d2:.3e}  "
              f"{'PASS' if t2 else 'FAIL'} (tol 1e-6)")

        # T3: predictor read-only isolation (with gates OPEN, the harder case)
        v04.alpha.data.fill_(0.5)
        ctx = v04._context(p, s)
        q_out = v04._cond_queries(s)                              # eval: no drop/noise
        p_q = v04.pred_q.unsqueeze(0) + (
            v04.pred_state(s) + v04.pred_act(
                torch.zeros(len(s), 60, device=dev))).unsqueeze(1)
        alone = v04._run_fuse(q_out, ctx, v04._mask_rows(8, dev))
        joint = v04._run_fuse(torch.cat([q_out, p_q], 1), ctx,
                              v04._mask_rows(8 + 16, dev))[:, :8]
        d3 = float((alone - joint).abs().max())
        t3 = d3 < 1e-5
        ok &= t3
        print(f"T3 predictor isolation:   max|diff| = {d3:.3e}  "
              f"{'PASS' if t3 else 'FAIL'} (tol 1e-5)")

        # T4: gate opening moves ONLY the conditioned 4 queries; 4..7 exactly v0.3
        open_set = v04.embed_set(p, s)
        d_cond = float((open_set[:, :4] - ref_set[:, :4]).abs().max())
        d_unc = float((open_set[:, 4:] - ref_set[:, 4:]).abs().max())
        open_pert = v04.embed_set(p, s_pert)
        s_cond = float((open_pert[:, :4] - open_set[:, :4]).abs().max())
        s_unc = float((open_pert[:, 4:] - open_set[:, 4:]).abs().max())
        t4 = d_cond > 1e-3 and d_unc < 1e-6 and s_cond > 1e-3 and s_unc < 1e-6
        ok &= t4
        print(f"T4 4/4 split @a=0.5:      cond-shift {d_cond:.3e} (>0 expected), "
              f"uncond-shift {d_unc:.3e} (=0), state-sens cond {s_cond:.3e} (>0), "
              f"uncond {s_unc:.3e} (=0)  {'PASS' if t4 else 'FAIL'}")

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
