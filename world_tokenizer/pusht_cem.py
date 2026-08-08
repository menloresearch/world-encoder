"""PushT rung-0 CEM planner + closed-loop eval (claim C, 2026-08-04, PROGRESS pt. 18).

Latent MPC through the stock PushTImageRunner protocol (n_test 50, seed 100000,
max_steps 300, legacy_test): per replan, encode the latest frame -> arm latent,
CEM over 8 dyn-steps (pop 256, elite 32, 4 iters, sigma 100->floor 10 px, mean
init = shifted previous plan), cost = mean over last 2 horizon steps of
||probe_xy - (256,256)||_2 + 60*|wrap(probe_theta - pi/4)| (block pose only),
execute the first dyn-step (= 2 raw 10 Hz actions), replan.

Per-planner-call latency/memory = single-call microbenchmark (B=1, 5 warmup +
20 timed replans): latent encode + full CEM loop; the shared ViT patch pass is
excluded (identical for both arms) and reported separately.

  PYTHONPATH=/home/menlo/brain/ishneet/diffusion_policy CUDA_VISIBLE_DEVICES=1 \
    python -m world_tokenizer.pusht_cem --arm z8
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from world_tokenizer.model import load_vitv2
from world_tokenizer.pusht_dynamics import LatentDynamics, ENC_DIR, CKPT_ROOT

NORM_M = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
NORM_S = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
GOAL_XY, GOAL_TH = (256.0, 256.0), np.pi / 4


class ArmEncoder(nn.Module):
    """image [B,3,96,96] float01 -> arm latent [B,n_tok,in_dim], standardized."""

    def __init__(self, arm, dyn_ckpt, dev):
        super().__init__()
        self.arm, self.dev = arm, dev
        self.vit = load_vitv2(pretrained=True).to(dev).eval()
        self.mu = torch.from_numpy(dyn_ckpt["mu"]).to(dev)
        self.sd = torch.from_numpy(dyn_ckpt["sd"]).to(dev)
        if arm != "dense":
            from world_tokenizer.pusht_dynamics import enc_class
            enc_dir = os.path.join(CKPT_ROOT, ENC_DIR[arm])
            nq = json.load(open(os.path.join(enc_dir, "results.json")))["args"]["queries"]
            self.enc = enc_class(arm)(n_cams=1, d=256, state_dim=5, n_queries=nq,
                                      lamb_patch=2.0, mask_ratio=0.25).to(dev).eval()
            self.enc.load_state_dict(torch.load(os.path.join(enc_dir, "seed0.pt"),
                                                map_location=dev))

    @torch.no_grad()
    def patches(self, img):                                  # the shared ViT pass
        x = torch.nn.functional.interpolate(img.to(self.dev), size=(224, 224),
                                            mode="bilinear", align_corners=False)
        x = (x - NORM_M.to(self.dev)) / NORM_S.to(self.dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.vit(x)["patch_latent"].float()       # [B,196,768]

    @torch.no_grad()
    def latent(self, patch):                                 # arm-specific, timed per call
        if self.arm == "dense":
            z = patch
        else:
            B = patch.shape[0]
            s = torch.zeros(B, 5, device=self.dev)           # state token is masked anyway
            ctx = self.enc._context(patch.unsqueeze(1), s)
            z = self.enc.fuse(ctx, self.enc._mask(True, self.dev), pool=False)
            if self.arm == "z8fsq":
                z = self.enc.fsq(z)
        return (z - self.mu) / self.sd


class CEMPlanner:
    def __init__(self, dyn, probe, dev, pop=256, elite=32, iters=4, horizon=8,
                 sigma0=100 / 256, sigma_floor=10 / 256, chunk=1024, seed=0,
                 h2=False):
        self.dyns = list(dyn) if isinstance(dyn, (list, tuple)) else [dyn]
        self.dyn, self.dev, self.h2 = self.dyns[0], dev, h2
        self.W = probe["W"].to(dev)
        self.xm = probe["x_mean"].to(dev)
        self.ym = probe["y_mean"].to(dev)
        self.pop, self.elite, self.iters, self.H = pop, elite, iters, horizon
        self.s0, self.sfloor, self.chunk = sigma0, sigma_floor, chunk
        self.gen = torch.Generator(device=dev).manual_seed(seed)
        self.prev = None                                     # [B,H,4] normalized

    def reset(self):
        self.prev = None

    @torch.no_grad()
    def _cost(self, z):                                      # z [M,n_tok,d] -> [M]
        y = (z.flatten(1) - self.xm) @ self.W + self.ym      # [M,6]
        exy = (y[:, 2:4] - torch.tensor(GOAL_XY, device=self.dev)).norm(dim=1)
        th = torch.atan2(y[:, 5], y[:, 4])
        eth = (th - GOAL_TH + np.pi) % (2 * np.pi) - np.pi
        return exy + 60.0 * eth.abs()

    @torch.no_grad()
    def _rollout_cost(self, z0, acts):
        # z0 [B,n_tok,d] (h2: tuple (z_prev, z_cur)); acts [B,P,H,4] -> cost [B,P]
        B, P = acts.shape[:2]
        cost = torch.zeros(B * P, device=self.dev)
        for i in range(0, B * P, self.chunk):
            sl = slice(i, min(i + self.chunk, B * P))
            a = acts.reshape(B * P, self.H, 4)[sl]
            if self.h2:
                zp = z0[0].repeat_interleave(P, 0)[sl]
                z = z0[1].repeat_interleave(P, 0)[sl]
            else:
                z = z0.repeat_interleave(P, 0)[sl]
            c = torch.zeros(z.shape[0], device=self.dev)
            if len(self.dyns) > 1:                   # E0: independent per-member rollouts
                zs = [z.clone() for _ in self.dyns]
                for h in range(self.H):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        zs = [zm + d(zm, a[:, h]).float()
                              for zm, d in zip(zs, self.dyns)]
                    if h >= self.H - 2:
                        c = c + sum(self._cost(zm) for zm in zs) / (2 * len(self.dyns))
            else:
                for h in range(self.H):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        if self.h2:
                            r = self.dyn(zp, z, a[:, h]).float()
                            zp, z = z, z + r
                        else:
                            z = z + self.dyn(z, a[:, h]).float()
                    if h >= self.H - 2:
                        c = c + self._cost(z) / 2
            cost[sl] = c
        return cost.view(B, P)

    @torch.no_grad()
    def plan(self, z0):
        # z0 [B,n_tok,d] standardized (h2: tuple) -> first-step chunk [B,2,2] env units
        B = (z0[1] if self.h2 else z0).shape[0]
        if self.prev is None or self.prev.shape[0] != B:
            mean = torch.zeros(B, self.H, 4, device=self.dev)
        else:                                                # receding-horizon shift
            mean = torch.cat([self.prev[:, 1:], self.prev[:, -1:]], 1)
        sigma = torch.full_like(mean, self.s0)
        for _ in range(self.iters):
            eps = torch.randn(B, self.pop, self.H, 4, device=self.dev,
                              generator=self.gen)
            acts = (mean.unsqueeze(1) + sigma.unsqueeze(1) * eps).clamp(-1, 1)
            cost = self._rollout_cost(z0, acts)              # [B,P]
            idx = cost.topk(self.elite, dim=1, largest=False).indices
            el = torch.gather(acts, 1, idx[..., None, None].expand(-1, -1, self.H, 4))
            mean = el.mean(1)
            sigma = el.std(1).clamp_min(self.sfloor)
        self.prev = mean
        a_env = (mean[:, 0].view(B, 2, 2) + 1.0) * 256.0     # first dyn-step, 2 raw actions
        return a_env.clamp(0, 512)


def build(arm, dev, seed=0, horizon=8, pop=256, history=1, ensemble=1):
    dyn_dir = f"pusht_dyn2_{arm}" if history == 2 else f"pusht_dyn_{arm}"
    ck = torch.load(os.path.join(CKPT_ROOT, dyn_dir, "seed0.pt"),
                    map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    if history == 2:
        from world_tokenizer.pusht_dynamics_h2 import LatentDynamicsH2
        dyn = LatentDynamicsH2(cfg["n_tokens"], cfg["in_dim"]).to(dev).eval()
    else:
        dyn = LatentDynamics(cfg["n_tokens"], cfg["in_dim"]).to(dev).eval()
    dyn.load_state_dict({k: v.to(dev) for k, v in ck["model"].items()})
    if ensemble > 1:                      # E0: seed ensemble, h1 only (pt. 22)
        assert history == 1
        dyn = [dyn]
        for s in range(1, ensemble):
            cks = torch.load(os.path.join(CKPT_ROOT, dyn_dir, f"seed{s}.pt"),
                             map_location="cpu", weights_only=False)
            m = LatentDynamics(cfg["n_tokens"], cfg["in_dim"]).to(dev).eval()
            m.load_state_dict({k: v.to(dev) for k, v in cks["model"].items()})
            dyn.append(m)
    enc = ArmEncoder(arm, ck, dev)
    planner = CEMPlanner(dyn, ck["probe"], dev, seed=seed,
                         pop=pop, horizon=horizon, h2=(history == 2),
                         chunk=(1024 if arm == "dense" else 8192))
    return enc, planner, cfg


class PushTCEMPolicy(nn.Module):
    """BaseImagePolicy-shaped adapter: predict_action = encode latest frame + CEM."""

    def __init__(self, enc, planner, dev):
        super().__init__()
        self.enc, self.planner, self._dev = enc, planner, dev
        self.call_times = []
        self._zprev = None      # h2: previous call's latent = exactly 1 dyn-step back

    @property
    def device(self):
        return torch.device(self._dev)

    @property
    def dtype(self):
        return torch.float32

    def reset(self):
        self.planner.reset()
        self._zprev = None

    @torch.no_grad()
    def predict_action(self, obs_dict):
        img = obs_dict["image"][:, -1]                       # [B,3,96,96] float01
        patch = self.enc.patches(img)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        z0 = self.enc.latent(patch)
        if self.planner.h2:
            zp = (self._zprev if self._zprev is not None
                  and self._zprev.shape[0] == z0.shape[0] else z0)
            act = self.planner.plan((zp, z0))                # [B,2,2]
            self._zprev = z0
        else:
            act = self.planner.plan(z0)                      # [B,2,2]
        torch.cuda.synchronize()
        self.call_times.append(time.perf_counter() - t0)
        return {"action": act.cpu()}


def microbench(enc, planner, dev, n_warm=5, n_timed=20):
    img = torch.rand(1, 3, 96, 96, device=dev)
    patch = enc.patches(img)
    planner.reset()

    def _plan():
        z = enc.latent(patch)
        planner.plan((z, z) if planner.h2 else z)

    for _ in range(n_warm):
        _plan()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(n_timed):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        _plan()
        torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
    peak = torch.cuda.max_memory_allocated() / 2**20
    tv = []
    for _ in range(n_timed):                                 # shared ViT pass, for context
        torch.cuda.synchronize(); t0 = time.perf_counter()
        enc.patches(img)
        torch.cuda.synchronize(); tv.append((time.perf_counter() - t0) * 1e3)
    planner.reset()
    return {"call_ms_median": float(np.median(ts)), "call_ms_p95": float(np.percentile(ts, 95)),
            "peak_mem_mib": float(peak), "vit_ms_median": float(np.median(tv))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ENC_DIR) + ["dense"])
    ap.add_argument("--n-test", type=int, default=50)
    ap.add_argument("--n-envs", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out",
                    default="/home/menlo/brain/ishneet/world-encoder/results/pusht")
    ap.add_argument("--bench-only", action="store_true")
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--pop", type=int, default=256)
    ap.add_argument("--history", type=int, default=1, choices=[1, 2])
    ap.add_argument("--ensemble", type=int, default=1)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    enc, planner, cfg = build(args.arm, dev, seed=args.seed,
                              horizon=args.horizon, pop=args.pop,
                              history=args.history, ensemble=args.ensemble)
    bench = microbench(enc, planner, dev)
    print(f"[{args.arm}] microbench: {json.dumps(bench)}", flush=True)
    res = {"arm": args.arm, "dyn_params": cfg["params"], "bench": bench,
           "horizon": args.horizon, "pop": args.pop, "history": args.history,
           "ensemble": args.ensemble}
    if not args.bench_only:
        from diffusion_policy.env_runner.pusht_image_runner import PushTImageRunner
        out_dir = os.path.join("/home/menlo/brain/ishneet/diffusion_policy/data/outputs",
                               "pusht_rung0", args.arm + args.tag)
        os.makedirs(out_dir, exist_ok=True)
        runner = PushTImageRunner(
            out_dir, n_train=0, n_test=args.n_test, max_steps=args.max_steps,
            n_obs_steps=2, n_action_steps=2, fps=10, legacy_test=True,
            test_start_seed=100000, n_envs=args.n_envs)
        policy = PushTCEMPolicy(enc, planner, dev)
        t0 = time.perf_counter()
        log = runner.run(policy)
        wall = time.perf_counter() - t0
        score = float(log["test/mean_score"])
        res.update({"test_mean_score": score, "wall_sec": wall,
                    "n_planner_calls": len(policy.call_times),
                    "closedloop_call_ms_median":
                        float(np.median(policy.call_times) * 1e3)})
        print(f"[{args.arm}] test/mean_score {score:.4f} wall {wall:.0f}s", flush=True)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"rung0_{args.arm}{args.tag}.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"SAVED {os.path.join(args.out, f'rung0_{args.arm}{args.tag}.json')}", flush=True)


if __name__ == "__main__":
    main()
