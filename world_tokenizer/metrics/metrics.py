"""Encoder-only metrics — judge the encoder with no predictor in the loop.

Predictor R^2 (T-1 -> T) can't rank encoders alone: a collapsed encoder emits constant
latents that the predictor fits perfectly. These metrics read the latent geometry directly.
Design discussion + the triplet-selection TODO live in METRICS.md next to this file.

All functions take precomputed latents as (N, D) numpy arrays (torch: .cpu().numpy() first);
triplet *selection* is deliberately out of scope here.

Smoke test:  python -m world_tokenizer.metrics.metrics
"""
import numpy as np

DISTANCES = ("mse", "cosine")


def _dist(a, b, kind):
    a = a.reshape(len(a), -1).astype(np.float64)
    b = b.reshape(len(b), -1).astype(np.float64)
    if kind == "mse":
        return ((a - b) ** 2).mean(axis=1)
    if kind == "cosine":
        an = a / np.linalg.norm(a, axis=1, keepdims=True)
        bn = b / np.linalg.norm(b, axis=1, keepdims=True)
        return 1.0 - (an * bn).sum(axis=1)
    raise ValueError(f"distance must be one of {DISTANCES}, got {kind!r}")


def triplet_accuracy(z_anchor, z_pos, z_neg, tiers=None, distance="mse", n_boot=1000, seed=0):
    """Fraction of triplets with d(anchor, pos) < d(anchor, neg), per negative tier.

    z_anchor/z_pos/z_neg: (N, ...) latents, row i is one triplet (extra dims are flattened).
    tiers: optional length-N labels (e.g. "same_episode", "modality_mismatch"); None -> one
    tier "all". Returns {tier: {n, acc, ci95, margin_mean, margin_p50}} where ci95 is a
    bootstrap interval on acc and margin = d_neg - d_pos (>0 means correctly ordered).
    Accuracy saturating at ~1.0 on easy tiers is expected and uninformative — read the
    hard tiers and the margins (see METRICS.md).
    """
    n = len(z_anchor)
    assert len(z_pos) == n and len(z_neg) == n, "triplet arrays must align row-wise"
    d_pos = _dist(z_anchor, z_pos, distance)
    d_neg = _dist(z_anchor, z_neg, distance)
    margin = d_neg - d_pos
    tiers = np.asarray(["all"] * n if tiers is None else tiers)

    rng = np.random.default_rng(seed)
    out = {}
    for tier in dict.fromkeys(tiers.tolist()):  # first-seen order
        m = margin[tiers == tier]
        correct = m > 0
        boot = rng.choice(correct, size=(n_boot, len(m)), replace=True).mean(axis=1)
        out[tier] = {
            "n": len(m),
            "acc": float(correct.mean()),
            "ci95": (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))),
            "margin_mean": float(m.mean()),
            "margin_p50": float(np.median(m)),
        }
    return out


def format_triplet_report(results_by_distance):
    """Pretty table for {distance: triplet_accuracy(...) result}."""
    lines = [f"{'tier':>20} | {'dist':>6} | {'n':>5} | {'acc':>5} | {'ci95':>14} | "
             f"{'margin p50':>10}"]
    for dist, res in results_by_distance.items():
        for tier, r in res.items():
            lo, hi = r["ci95"]
            lines.append(f"{tier:>20} | {dist:>6} | {r['n']:>5} | {r['acc']:>5.3f} | "
                         f"[{lo:.3f},{hi:.3f}] | {r['margin_p50']:>10.4f}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Synthetic smoke test: easy negatives ~1.0, hard-as-positive negatives ~0.5.
    rng = np.random.default_rng(0)
    za = rng.normal(size=(4000, 64))
    zp = za + 0.1 * rng.normal(size=za.shape)
    zn = np.concatenate([3 * rng.normal(size=(2000, 64)),                  # easy: far cluster
                         za[2000:] + 0.1 * rng.normal(size=(2000, 64))])   # impossible: ~ own positive
    tiers = ["easy"] * 2000 + ["impossible"] * 2000
    res = {d: triplet_accuracy(za, zp, zn, tiers=tiers, distance=d) for d in DISTANCES}
    print(format_triplet_report(res))
    assert res["mse"]["easy"]["acc"] > 0.99, res["mse"]["easy"]
    assert 0.4 < res["mse"]["impossible"]["acc"] < 0.6, res["mse"]["impossible"]
    print("smoke test OK")
