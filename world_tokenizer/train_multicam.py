"""Build-1 trainer (V0.2.md): multi-camera MMPerceiverChunks on the K-cam shard caches.

Same recipe as train_chunks.py (losses, opt, epochs) with mm_perceiver3's K-camera
context + a per-serial camera-id embedding; still single-timestep. The eval IS the
Build-1 gate — on the SAME held-out chunks it probes, apples-to-apples:
  mc4    z_v of this model, all K cameras          (the candidate)
  mc1    z_v of this model, camera slot 0 only     (isolates the extra views' worth)
  v01    z_v of the frozen single-view v0.1 ckpt on camera slot 0 (the bar to hold:
         slot 0 = sorted-first external serial = the view the single-view cache used)
  raw4/raw1  pooled raw ViT features (all-cams / cam0)
Gate reads: mc4 ee-R2 >= v01 ee-R2 (~0.25 on kuka) with RankMe not collapsed (v0.1
~130) proves the bottleneck compresses 4 views without losing the force signal; mc4 >
mc1 says the extra views actually help.

    python -m world_tokenizer.train_multicam --train-cfgs 6 7 --tag kuka_mc4 \
        --out-dir /mnt/nas/data/RH20T/checkpoints/multicam
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_tokenizer.dataloader import load_split, make_loader  # noqa: E402
from world_tokenizer.mm_perceiver2 import MMPerceiverChunks as MMPerceiverV01  # noqa: E402
from world_tokenizer.mm_perceiver3 import MMPerceiverChunks, masked_mean  # noqa: E402
from world_tokenizer.train_chunks import probe_r2, rankme  # noqa: E402


def rand_keep(cam_ids):
    """Per-sample camera dropout (V0.2.md JQ follow-up #2): keep a uniform-size
    (1..K), uniform-composition subset of views. [B,K] bool, True=keep."""
    B, K = cam_ids.shape
    r = torch.randint(1, K + 1, (B, 1), device=cam_ids.device)
    return torch.rand(B, K, device=cam_ids.device).argsort(1) < r


def unpack_mc(batch, dev):
    """Multicam packet -> (cams list of K [B,196,768], motor, m_mask, ee, e_mask, cam_ids)."""
    rgb = batch["rgb"].to(dev, non_blocking=True)                    # [B,K,196,768]
    return (list(rgb.unbind(1)),
            batch["motor"].squeeze(1).to(dev, non_blocking=True),
            batch["motor_mask"].to(dev, non_blocking=True),
            batch["ee"].to(dev, non_blocking=True),
            batch["ee_mask"].to(dev, non_blocking=True),
            batch["cam_ids"].to(dev, non_blocking=True))             # [B,K]


@torch.no_grad()
def encode_variants(model, v01, ds, dev, bs=256):
    """z_v under each eval arm for every sample of the multicam ChunkDataset."""
    d, n = ds._d, len(ds)
    out = {k: [] for k in ("mc4", "mc1", "v01", "raw4", "raw1")}
    model.eval(), v01.eval()
    for i in tqdm(range(0, n, bs), desc="encode", mininterval=10, leave=False):
        sl = slice(i, min(i + bs, n))
        rgb = torch.from_numpy(d["patch"][sl].astype(np.float32)).to(dev)  # [b,K,196,768]
        cams = list(rgb.unbind(1))
        cam_ids = torch.from_numpy(ds._cam_ids[sl]).to(dev)
        motor = torch.from_numpy(d["motor"][sl]).squeeze(1).to(dev)
        m_mask = torch.from_numpy(d["motor_mask"][sl]).to(dev)
        ee = torch.from_numpy(d["ee"][sl]).to(dev)
        e_mask = torch.from_numpy(d["ee_mask"][sl]).to(dev)
        out["mc4"].append(model.embed_vision(cams, motor, m_mask, ee, e_mask,
                                             cam_ids).cpu().numpy())
        out["mc1"].append(model.embed_vision(cams[:1], motor, m_mask, ee, e_mask,
                                             cam_ids[:, :1]).cpu().numpy())
        out["v01"].append(v01.embed_vision(cams[0], motor, m_mask, ee,
                                           e_mask).cpu().numpy())
        out["raw4"].append(rgb.mean((1, 2)).cpu().numpy())
        out["raw1"].append(cams[0].mean(1).cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def gate_eval(model, v01, ds, split, dev):
    """Probe every arm on the held-out groups; probes fit on train groups only."""
    is_test = np.array([split[ds.groups[g]] == "test" for g in ds._group_idx])
    zs = encode_variants(model, v01, ds, dev)

    d = ds._d
    motor = d["motor"].reshape(len(ds), -1)
    mvalid = d["motor_mask"].reshape(len(ds), -1).all(0)
    y_motor = motor[:, mvalid]
    e_any = d["ee_mask"].any(1)
    y_ee = masked_mean(torch.from_numpy(d["ee"]),
                       torch.from_numpy(d["ee_mask"])).numpy()

    tr, te = ~is_test, is_test
    res = {"n_test": int(te.sum())}
    for name, z in zs.items():
        res[f"{name}_rankme"] = rankme(z[te])
        res[f"{name}_r2_motor"] = probe_r2(z[tr], y_motor[tr], z[te], y_motor[te])
        etr, ete = tr & e_any, te & e_any
        if etr.sum() > 50 and ete.sum() > 50:
            res[f"{name}_r2_ee"] = probe_r2(z[etr], y_ee[etr], z[ete], y_ee[ete])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/mnt/nas/data/RH20T/caches")
    ap.add_argument("--train-cfgs", type=int, nargs="+", default=[6, 7])
    ap.add_argument("--tag", default="kuka_mc4")
    ap.add_argument("--out-dir", default="/mnt/nas/data/RH20T/checkpoints/multicam")
    ap.add_argument("--baseline-ckpt",
                    default="/mnt/nas/data/RH20T/checkpoints/phase1/kuka/seed0.pt",
                    help="frozen single-view v0.1 ckpt = the gate baseline")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--queries", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cam-dropout", action="store_true",
                    help="per-sample random 1..K view subset on the context (targets full)")
    args = ap.parse_args()

    for _ in range(15):                                   # VM CUDA-init is flaky under load
        if torch.cuda.is_available():
            break
        time.sleep(10)
    if not torch.cuda.is_available():
        print("!! CUDA unavailable after retries — refusing to train on CPU", flush=True)
        sys.exit(2)
    dev = "cuda"
    run_dir = os.path.join(args.out_dir, args.tag)
    os.makedirs(run_dir, exist_ok=True)
    split = load_split()

    train_loader, _, ds = make_loader(args.cache_dir, tuple(args.train_cfgs),
                                      batch=args.batch, num_workers=args.workers,
                                      multicam=True)
    K, n_ids = ds._cam_ids.shape[1], len(ds.cam_vocab)
    print(f"[{args.tag}] {len(ds)} chunks, K={K} cams, {n_ids} serials "
          f"{ds.cam_vocab} | cfgs {args.train_cfgs} | dev {dev}", flush=True)

    v01 = MMPerceiverV01(d=args.d, n_queries=args.queries).to(dev)
    v01.load_state_dict(torch.load(args.baseline_ckpt, map_location=dev))

    results = {"args": vars(args), "cam_vocab": ds.cam_vocab, "seeds": {}}
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = MMPerceiverChunks(d=args.d, n_queries=args.queries,
                                  n_cam_ids=n_ids).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        t0 = time.time()
        for ep in tqdm(range(args.epochs), desc=f"seed {seed} train", mininterval=5):
            model.train()
            ep_loss, ep_inv, nb = 0.0, 0.0, 0
            for batch in train_loader:
                cams, motor, m_mask, ee, e_mask, cam_ids = unpack_mc(batch, dev)
                out = model(cams, motor, m_mask, ee, e_mask, cam_ids,
                            cam_keep=rand_keep(cam_ids) if args.cam_dropout else None)
                opt.zero_grad()
                out["loss"].backward()
                opt.step()
                model.update_target()
                ep_loss += float(out["loss"]); ep_inv += float(out["inv"]); nb += 1
            if ep % 10 == 0 or ep == args.epochs - 1:
                print(f"  seed {seed} ep {ep}: loss {ep_loss / max(nb, 1):.4f} "
                      f"inv {ep_inv / max(nb, 1):.4f}", flush=True)

        torch.save(model.state_dict(), os.path.join(run_dir, f"seed{seed}.pt"))
        res = gate_eval(model, v01, ds, split, dev)
        res["train_minutes"] = (time.time() - t0) / 60
        print(f"  seed {seed} " + " ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in res.items()), flush=True)
        results["seeds"][seed] = res
        with open(os.path.join(run_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
