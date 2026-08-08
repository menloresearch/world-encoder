"""Plannability diagnostic for PushT rung-0 (2026-08-04, supplements pt. 18).

Separates "dynamics wrong off-manifold" from "planner/cost wrong": for a few
real env states, plan with CEM, then (a) roll the DYNAMICS under the planned
actions (imagined trajectory, probe readout per step) and (b) execute the SAME
actions open-loop in the REAL env. Reports per-step block-pose divergence
(imagined vs real, px) and imagined vs realized cost. A planner that exploits
model error shows imagined cost falling while realized cost does not.

Symmetric across arms — run with --arm z8 and --arm dense at matched knobs.

  PYTHONPATH=/home/menlo/brain/ishneet/diffusion_policy CUDA_VISIBLE_DEVICES=4 \
    python -m world_tokenizer.pusht_plan_diag --arm z8
"""
import argparse
import json

import numpy as np
import torch

from world_tokenizer.pusht_cem import build, GOAL_XY, GOAL_TH


def pose_cost(xy, th):
    exy = np.linalg.norm(np.asarray(xy, np.float64) - np.array(GOAL_XY))
    eth = abs((th - GOAL_TH + np.pi) % (2 * np.pi) - np.pi)
    return exy + 60.0 * eth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=3)   # plan->execute rounds per seed
    ap.add_argument("--out",
                    default="/home/menlo/brain/ishneet/world-encoder/results/pusht")
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    np.random.seed(0)

    from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
    enc, planner, cfg = build(args.arm, dev)
    H = planner.H

    rows = []
    for si in range(args.n_seeds):
        env = PushTImageEnv(legacy=True, render_size=96)
        env.seed(100000 + si)
        obs = env.reset()
        planner.reset()
        for rnd in range(args.rounds):
            img = torch.from_numpy(obs["image"]).float().unsqueeze(0)
            patch = enc.patches(img)
            z = enc.latent(patch)
            planner.plan(z)                      # fills planner.prev with the full mean
            plan = planner.prev[0].clone()       # [H,4] normalized

            # imagined: dynamics rollout + probe readout
            imag = []
            zi = z.clone()
            with torch.no_grad():
                for h in range(H):
                    zi = zi + planner.dyn(zi, plan[h:h + 1]).float()
                    y = (zi.flatten(1) - planner.xm) @ planner.W + planner.ym
                    bx, by = float(y[0, 2]), float(y[0, 3])
                    th = float(torch.atan2(y[0, 5], y[0, 4]))
                    imag.append((bx, by, th))

            # real: execute the same actions open-loop
            real = []
            for h in range(H):
                a2 = ((plan[h].view(2, 2) + 1.0) * 256.0).clamp(0, 512).cpu().numpy()
                for a in a2:
                    obs, _, _, _ = env.step(a)
                real.append((env.block.position[0], env.block.position[1],
                             env.block.angle))

            div = [float(np.linalg.norm(np.array(i[:2]) - np.array(r[:2])))
                   for i, r in zip(imag, real)]
            rows.append({
                "seed": si, "round": rnd,
                "block_div_px_per_step": [round(x, 1) for x in div],
                "imag_cost_final": round(pose_cost(imag[-1][:2], imag[-1][2]), 1),
                "real_cost_final": round(pose_cost(real[-1][:2], real[-1][2]), 1)})

    div_all = np.array([r["block_div_px_per_step"] for r in rows])   # [n,H]
    imag_f = np.array([r["imag_cost_final"] for r in rows])
    real_f = np.array([r["real_cost_final"] for r in rows])
    summary = {
        "arm": args.arm, "n": len(rows),
        "block_div_px_median_per_step": np.median(div_all, 0).round(1).tolist(),
        "imag_cost_final_median": float(np.median(imag_f)),
        "real_cost_final_median": float(np.median(real_f)),
        "exploit_gap_median": float(np.median(real_f - imag_f)),
    }
    print(json.dumps(summary, indent=1), flush=True)
    with open(f"{args.out}/plan_diag_{args.arm}.json", "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
