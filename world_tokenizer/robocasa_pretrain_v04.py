"""v0.4 "StateAdd" (V0.4.md §2 axis A rank-1 + research/v04/01-03): state-privileged encoder.

Extends the v0.3 harness (robocasa_pretrain_v03.MMPerceiverMCPatch) with:

  A. StateAdd query conditioning — state -> MLP(d_s->256->8*256) -> reshape [8,256],
     added to the 8 learned queries through PER-QUERY tanh(alpha) gates, alpha zero-init.
     At alpha=0 the policy-facing unpooled latent is EXACTLY the v0.3 encoder
     (v04_equiv_test.py is the gate; the v0.3 ckpt loads with strict=False).
  B. 4/4 split (02_shortcut_mitigation §3) — only queries 0..3 receive the injection
     (cond_qmask buffer); queries 4..7 never see state.
  C. P=16 dedicated predictor queries (03_objectives §4b) — attend the SAME vision K/V
     (state column masked: state stays out of the value path, 01 §6.2), receive state +
     action-chunk embeddings by broadcast-add. PerceiverFuse has no query self-attention,
     so the 8 output queries structurally cannot attend to them (Octo readout isolation
     for free). Aux loss lives ONLY on predictor outputs, read at the final block.
  D. Paired objective (03 §5 #1) — action-conditioned future-vision-latent prediction:
     target = RAW frozen ViT-B/16 patch-pooled features per cam at t+h (hard-frozen,
     straight from the cache; stop-grad by construction), cosine loss, weight lamb_aux.
     h is given in control ticks (20 Hz); the cache is stride-5, so h snaps to
     h_eff = round(h/stride)*stride (h=6 -> 5 ticks = 0.25 s; h=16 -> 15 ticks = 0.75 s).
     The action chunk covers exactly t..t+h_eff so the residual the loss chases is
     scene content, not arm motion.
  E. Anti-shortcut stack (02 §5, config-gated, defaults ON) — per-dim state dropout
     p=0.2 (DiT-Block recipe); Gaussian state noise sigma = 10^(-SNR/20) on the
     normalized state (RDT SNR form, default 20 dB -> 0.1; NADA sigma* needs rollouts);
     CFG-style whole-condition dropout, p cosine-annealed 0.5 -> 0.1 over the first half
     of training then constant, null = learned token at the injection level; the
     zero-init gates from A. Regularization touches ONLY the new state paths (query
     conditioning + predictor conditioning); the v0.3 loss paths see clean state.

The v0.3 losses (inv + lamb*sig + lamb_patch*patch) are kept verbatim; total loss adds
+ lamb_aux * aux. NOTE (recorded risk): with state conditioning the queries, the v0.3
pred_s(z_from_v) term becomes partially copyable once gates open — condition dropout
keeps it non-copyable on dropped samples and v04_detector.py D2 is the watchdog.

    cd /home/menlo/brain/ishneet/world-encoder && \
    CUDA_VISIBLE_DEVICES=0 /home/menlo/brain/ishneet/robocasa/.venv/bin/python3 \
      -m world_tokenizer.robocasa_pretrain_v04 --h 6 --lamb-aux 0.05 \
      --dataset /mnt/nas/data/robocasa/datasets/v1.0/target/atomic/PickPlaceCounterToCabinet/20250811/lerobot
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_tokenizer.mm_perceiver import _mlp  # noqa: E402
from world_tokenizer.robocasa_pretrain_v03 import MMPerceiverMCPatch, gates  # noqa: E402


class MMPerceiverMCStateAdd(MMPerceiverMCPatch):
    """v0.3 encoder + StateAdd gated query conditioning + dedicated future predictor."""

    def __init__(self, n_cams=3, act_len=5, act_dim=12, n_pred=16, lamb_aux=0.05,
                 state_drop=0.2, state_noise=0.1, cond_drop_min=0.1, cond_drop_max=0.5,
                 **kw):
        super().__init__(n_cams=n_cams, **kw)
        d, M = self.proj_v.out_features, self.n_queries
        d_s, d_v = self.proj_s.in_features, self.proj_v.in_features
        # A: state -> MLP(d_s->256->M*d) -> [M,d]; per-query tanh(alpha) gates, alpha=0
        self.state_mlp = _mlp(d_s, M * d, 256)
        self.alpha = nn.Parameter(torch.zeros(M))
        # B: 4/4 split — only the first M//2 queries ever receive the injection
        qm = torch.zeros(M)
        qm[:M // 2] = 1.0
        self.register_buffer("cond_qmask", qm)
        # E3: learned null token (CFG condition dropout), at the injection level
        self.null_u = nn.Parameter(torch.randn(M, d) * 0.02)
        # C: dedicated predictor queries + their conditioning (state + action chunk)
        self.pred_q = nn.Parameter(torch.randn(n_pred, d) * 0.02)
        self.pred_state = _mlp(d_s, d, 256)
        self.pred_act = _mlp(act_len * act_dim, d, 256)
        # D: FRAPPE-style projector head: pooled predictor outputs -> K per-cam targets
        self.pred_head = _mlp(d, n_cams * d_v, 1024)
        self.n_pred, self.lamb_aux = n_pred, lamb_aux
        self.state_drop, self.state_noise = state_drop, state_noise
        self.cond_drop_min, self.cond_drop_max = cond_drop_min, cond_drop_max

    # ---- anti-shortcut stack (train-time only) ----
    def cond_drop_p(self, progress):
        """Cosine anneal cond_drop_max -> cond_drop_min over first half, then constant."""
        if not self.training or self.cond_drop_max <= 0:
            return 0.0
        lo, hi = self.cond_drop_min, self.cond_drop_max
        if progress >= 0.5:
            return lo
        return lo + (hi - lo) * 0.5 * (1.0 + math.cos(math.pi * progress / 0.5))

    def _reg_state(self, s):
        if self.training and self.state_drop > 0:
            s = F.dropout(s, p=self.state_drop, training=True)   # per-dim, 1/(1-p) scaled
        if self.training and self.state_noise > 0:
            s = s + self.state_noise * torch.randn_like(s)
        return s

    # ---- query construction ----
    def _cond_queries(self, s_reg, p_drop=0.0):
        """[B,M,d] output queries: learned q + qmask * tanh(alpha) * (state MLP | null)."""
        B = s_reg.shape[0]
        u = self.state_mlp(s_reg).view(B, self.n_queries, -1)
        if p_drop > 0:
            drop = (torch.rand(B, device=s_reg.device) < p_drop)[:, None, None]
            u = torch.where(drop, self.null_u.unsqueeze(0), u)
        gate = (self.cond_qmask * torch.tanh(self.alpha))[None, :, None]
        return self.fuse.q.unsqueeze(0) + gate * u

    def _run_fuse(self, q, ctx, attn_mask):
        """PerceiverFuse blocks on EXTERNAL queries (same op order as fuse.forward)."""
        x = q
        f = self.fuse
        for ca, n1, ffn, n2 in zip(f.ca, f.n1, f.ffn, f.n2):
            x = x + ca(n1(x), ctx, attn_mask=attn_mask)
            x = x + ffn(n2(x))
        return x

    def _mask_rows(self, n_rows, device):
        """[n_rows, K*196+1]: block the state K/V column for every query row."""
        n_vis = self.n_cams * self.n_patch
        m = torch.zeros(n_rows, n_vis + 1, dtype=torch.bool, device=device)
        m[:, n_vis:] = True
        return m

    # ---- training forward ----
    def forward(self, patch, state, act, fut_v, progress=1.0):
        # patch [B,K,196,768]; state [B,d_s] (normalized); act [B,act_len*act_dim];
        # fut_v [B,K,768] = frozen per-cam pooled ViT features at t+h_eff (stop-grad data)
        B = patch.shape[0]
        ctx = self._context(patch, state)                        # clean state, as v0.3
        with torch.no_grad():
            tv_pp = self.tgt_v(patch)                            # [B,K,196,d]
            tv = tv_pp.mean((1, 2))
            ts = self.tgt_s(state)

        s_reg = self._reg_state(state)                           # E1+E2, new paths only
        q_out = self._cond_queries(s_reg, self.cond_drop_p(progress))       # [B,M,d]
        p_q = self.pred_q.unsqueeze(0) \
            + (self.pred_state(s_reg) + self.pred_act(act)).unsqueeze(1)    # [B,P,d]
        x = self._run_fuse(torch.cat([q_out, p_q], 1), ctx,
                           self._mask_rows(self.n_queries + self.n_pred, patch.device))
        zv_set, zp = x[:, :self.n_queries], x[:, self.n_queries:]           # final block

        # ---- v0.3 losses, verbatim structure ----
        z_from_v = zv_set.mean(1)
        z_from_s = self.fuse(ctx, self._mask(False, patch.device))
        inv = (self.pred_s(z_from_v) - ts).square().mean() \
            + (self.pred_v(z_from_s) - tv).square().mean()

        n_tot = self.n_cams * self.n_patch
        n_sel = max(1, int(self.mask_ratio * n_tot))
        sel = torch.randperm(n_tot, device=patch.device)[:n_sel]
        ic, ip = sel // self.n_patch, sel % self.n_patch
        q = self.dec_pos[ic, ip].unsqueeze(0).expand(B, -1, -1)
        pred_pp = self.dec(q, zv_set)
        patch_loss = (pred_pp - tv_pp[:, ic, ip]).square().mean()

        ev = self.proj_v(patch).mean((1, 2))
        es = self.proj_s(state)
        z_full = self.fuse(ctx)
        sig = self.sigreg(ev) + self.sigreg(es) + self.sigreg(z_full)

        # ---- v0.4 aux: cosine to frozen future per-cam pooled features ----
        pred = self.pred_head(zp.mean(1)).view(B, self.n_cams, -1)
        aux = (1.0 - F.cosine_similarity(pred, fut_v, dim=-1)).mean()

        loss = inv + self.lamb * sig + self.lamb_patch * patch_loss + self.lamb_aux * aux
        return {"loss": loss, "inv": inv.detach(), "sig": sig.detach(),
                "patch": patch_loss.detach(), "aux": aux.detach(),
                "z": z_full.detach()}

    # ---- eval / policy-facing paths (no reg, no cond-drop) ----
    @torch.no_grad()
    def embed_set(self, patch, state, null_cond=False):
        """Unpooled policy-facing latent [B,M,d]. At alpha=0 this is EXACTLY v0.3's
        fuse(ctx, mask_block_state, pool=False). null_cond swaps in the learned null."""
        B = patch.shape[0]
        ctx = self._context(patch, state)
        if null_cond:
            u = self.null_u.unsqueeze(0).expand(B, -1, -1)
        else:
            u = self.state_mlp(state).view(B, self.n_queries, -1)
        gate = (self.cond_qmask * torch.tanh(self.alpha))[None, :, None]
        q = self.fuse.q.unsqueeze(0) + gate * u
        return self._run_fuse(q, ctx, self._mask_rows(self.n_queries, patch.device))

    @torch.no_grad()
    def embed_vision(self, patch, state):
        """Pooled conditioned latent — keeps the v0.3 gates() battery reusable."""
        return self.embed_set(patch, state).mean(1)

    @torch.no_grad()
    def predict_future(self, patch, state, act):
        """Aux prediction [B,K,768] (detector aux-sweep path)."""
        B = patch.shape[0]
        ctx = self._context(patch, state)
        p_q = self.pred_q.unsqueeze(0) \
            + (self.pred_state(state) + self.pred_act(act)).unsqueeze(1)
        zp = self._run_fuse(p_q, ctx, self._mask_rows(self.n_pred, patch.device))
        return self.pred_head(zp.mean(1)).view(B, self.n_cams, -1)


# ---------------- data helpers (shared with v04_equiv_test / v04_detector) ----------------

def infer_stride(ep, t):
    same = ep[1:] == ep[:-1]
    return int(np.min((t[1:] - t[:-1])[same]))


def load_actions(dataset):
    """dict episode -> [T, act_dim] float32 commanded actions (per frame, 20 Hz)."""
    import pandas as pd
    acts = {}
    for pq in sorted(glob.glob(os.path.join(dataset, "data", "chunk-*", "*.parquet"))):
        df = pd.read_parquet(pq, columns=["action", "episode_index"])
        acts[int(df["episode_index"].iloc[0])] = np.stack(
            df["action"].to_numpy()).astype(np.float32)
    return acts


def build_future_and_actions(ep, t, actions, h_eff):
    """Per cache row i: fut_row[i] = row of (ep_i, t_i + h_eff), clamped to the episode's
    last cached frame (E4 min() pattern); act[i] = actions[t_i : t_i+h_eff] flattened,
    padded past episode end by repeating the last action (action-hold)."""
    row = {(int(e), int(tt)): i for i, (e, tt) in enumerate(zip(ep, t))}
    max_t = {}
    for e, tt in zip(ep, t):
        max_t[int(e)] = max(max_t.get(int(e), -1), int(tt))
    act_dim = next(iter(actions.values())).shape[1]
    fut = np.empty(len(ep), np.int64)
    acts = np.empty((len(ep), h_eff * act_dim), np.float32)
    n_clamp = 0
    for i, (e, tt) in enumerate(zip(ep, t)):
        e, tt = int(e), int(tt)
        ft = min(tt + h_eff, max_t[e])
        n_clamp += int(ft != tt + h_eff)
        fut[i] = row[(e, ft)]
        a = actions[e]
        idx = np.clip(np.arange(tt, tt + h_eff), 0, len(a) - 1)
        acts[i] = a[idx].reshape(-1)
    return fut, acts, n_clamp, act_dim


def pool_patches(patch, bs=2048):
    """[N,K,196,768] fp16 -> per-cam patch-pooled [N,K,768] fp32 (the frozen aux target)."""
    out = np.empty((len(patch), patch.shape[1], patch.shape[3]), np.float32)
    for i in range(0, len(patch), bs):
        out[i:i + bs] = patch[i:i + bs].astype(np.float32).mean(2)
    return out


def _grad_norm(module):
    tot = 0.0
    for p in module.parameters():
        if p.grad is not None:
            tot += float(p.grad.norm()) ** 2
    return tot ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cache",
                    default="/mnt/nas/data/robocasa/kepler_cache/pnp_patches.npz")
    ap.add_argument("--out-base",
                    default="/mnt/nas/data/robocasa/kepler_ckpt/pnp_mc3_v04sa")
    ap.add_argument("--h", type=int, default=6,
                    help="prediction horizon in control ticks (20 Hz); snaps to stride")
    ap.add_argument("--lamb-aux", type=float, default=0.05)
    ap.add_argument("--lamb-patch", type=float, default=2.0)
    ap.add_argument("--n-pred", type=int, default=16)
    ap.add_argument("--mask-ratio", type=float, default=0.25)
    ap.add_argument("--state-drop", type=float, default=0.2)
    ap.add_argument("--state-noise-snr", type=float, default=20.0,
                    help="dB on normalized state; <=0 disables")
    ap.add_argument("--cond-drop-max", type=float, default=0.5)
    ap.add_argument("--cond-drop-min", type=float, default=0.1)
    ap.add_argument("--init-ckpt", default=None,
                    help="optional v0.3 ckpt to warm-start (strict=False)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--queries", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="smoke mode: stop after N optimizer steps, log every 25")
    ap.add_argument("--no-gates", action="store_true")
    args = ap.parse_args()

    d = np.load(args.cache)
    patch, state_raw, ep, t = d["patch"], d["state"].astype(np.float32), d["ep"], d["t"]
    mu, sd = state_raw.mean(0), state_raw.std(0) + 1e-6
    state = (state_raw - mu) / sd
    is_test = (ep % 10 == 0)
    K = patch.shape[1]
    stride = infer_stride(ep, t)
    h_eff = max(1, round(args.h / stride)) * stride
    print(f"{len(ep)} samples, K={K}, test {is_test.sum()}, stride {stride}, "
          f"h {args.h} -> h_eff {h_eff} ticks ({h_eff / 20:.2f}s)", flush=True)

    actions = load_actions(args.dataset)
    fut_row, act_chunk, n_clamp, act_dim = build_future_and_actions(ep, t, actions, h_eff)
    pooled = pool_patches(patch)                                 # frozen aux targets
    print(f"actions {act_dim}-D, chunk {h_eff}x{act_dim}, "
          f"{n_clamp} end-of-episode clamped targets", flush=True)

    sigma = 10 ** (-args.state_noise_snr / 20) if args.state_noise_snr > 0 else 0.0
    dev = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = MMPerceiverMCStateAdd(
        n_cams=K, d=args.d, state_dim=state.shape[1], n_queries=args.queries,
        lamb_patch=args.lamb_patch, mask_ratio=args.mask_ratio,
        act_len=h_eff, act_dim=act_dim, n_pred=args.n_pred, lamb_aux=args.lamb_aux,
        state_drop=args.state_drop, state_noise=sigma,
        cond_drop_min=args.cond_drop_min, cond_drop_max=args.cond_drop_max).to(dev)
    if args.init_ckpt:
        sd_ck = torch.load(args.init_ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(sd_ck, strict=False)
        assert not unexpected, unexpected
        print(f"warm-start {args.init_ckpt}: {len(missing)} new-param keys "
              f"(v0.4 additions), 0 unexpected", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    tr_idx = np.flatnonzero(~is_test)
    steps_per_epoch = math.ceil(len(tr_idx) / args.batch)
    total_steps = args.max_steps or args.epochs * steps_per_epoch
    step = 0
    for epoch in range(args.epochs):
        model.train()
        np.random.shuffle(tr_idx)
        tot = np.zeros(4)
        gnorm = np.zeros(3)
        nb = 0
        for i in range(0, len(tr_idx), args.batch):
            sl = tr_idx[i:i + args.batch]
            p = torch.from_numpy(patch[sl].astype(np.float32)).to(dev)
            s = torch.from_numpy(state[sl]).to(dev)
            a = torch.from_numpy(act_chunk[sl]).to(dev)
            fv = torch.from_numpy(pooled[fut_row[sl]]).to(dev)
            step += 1
            out = model(p, s, a, fv, progress=step / total_steps)
            opt.zero_grad()
            out["loss"].backward()
            ga = float(model.alpha.grad.norm())
            gs = _grad_norm(model.state_mlp)
            gv = _grad_norm(model.proj_v)
            opt.step()
            model.update_target()
            tot += [float(out["loss"]), float(out["inv"]),
                    float(out["patch"]), float(out["aux"])]
            gnorm += [ga, gs, gv]
            nb += 1
            if args.max_steps and step % 25 == 0:
                g = torch.tanh(model.alpha.detach())[:args.queries // 2]
                print(f"step {step}: loss {tot[0]/nb:.4f} aux {tot[3]/nb:.4f} "
                      f"p_drop {model.cond_drop_p(step/total_steps):.3f} "
                      f"|tanh a|cond {float(g.abs().mean()):.5f} "
                      f"grad[alpha {gnorm[0]/nb:.2e} s_mlp {gnorm[1]/nb:.2e} "
                      f"proj_v {gnorm[2]/nb:.2e}]", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        g = torch.tanh(model.alpha.detach())
        print(f"ep {epoch}: loss {tot[0]/nb:.4f} inv {tot[1]/nb:.4f} "
              f"patch {tot[2]/nb:.4f} aux {tot[3]/nb:.4f} | "
              f"p_drop {model.cond_drop_p(step/total_steps):.3f} "
              f"|tanh a|cond {float(g[:args.queries//2].abs().mean()):.5f} | "
              f"grad[alpha {gnorm[0]/nb:.2e} s_mlp {gnorm[1]/nb:.2e} "
              f"proj_v {gnorm[2]/nb:.2e}]", flush=True)
        if args.max_steps and step >= args.max_steps:
            break

    model.eval()
    res = {}
    if not args.no_gates:
        # v0.2 z reference straight from the policy z cache (same frames), as v0.3
        from world_tokenizer.seese3_probe import Z_CACHE, FRAME_IDX
        fc = json.load(open(FRAME_IDX))
        zc_meta = json.load(open(f"{Z_CACHE}/meta.json"))
        zmm = np.memmap(f"{Z_CACHE}/z.f32", dtype=np.float32, mode="r",
                        shape=(fc["total_frames"], zc_meta["d"]))
        rows = np.array([fc["episodes"][str(e)]["start"] for e in ep]) + t
        z_v02 = np.array(zmm[rows])
        dd = {"patch": patch, "state": state_raw, "ep": ep, "t": t}
        res = gates(model, dd, z_v02, args.dataset, dev)
        print("GATE:", json.dumps(res, default=float)[:800], flush=True)

    out_dir = f"{args.out_base}_h{args.h}_l{args.lamb_aux}"
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, f"seed{args.seed}.pt"))
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"args": vars(args), "class": "MMPerceiverMCStateAdd",
                   "h_eff": h_eff, "act_dim": act_dim, "sigma": sigma,
                   "alpha_final": model.alpha.detach().cpu().tolist(),
                   "gate": res, "state_mu": mu.tolist(), "state_sd": sd.tolist()},
                  f, indent=1, default=float)
    print(f"SAVED {out_dir}", flush=True)


if __name__ == "__main__":
    main()
