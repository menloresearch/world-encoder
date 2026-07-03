"""Encoder-only metrics — judge the encoder with no predictor in the loop.

Predictor R^2 (T-1 -> T) can't rank encoders alone: a collapsed encoder emits constant
latents that the predictor fits perfectly. These metrics read the latent geometry directly.
Design discussion + the triplet-selection TODO live in METRICS.md next to this file.

All functions take precomputed latents as (N, D) numpy arrays (torch: .cpu().numpy() first);
triplet *selection* is deliberately out of scope here.

Smoke test:  python -m world_tokenizer.metrics.metrics
"""
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

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


def effective_rank(z):
    """RankMe (Garrido et al. 2023) — label-free dimensional-collapse detector.

    Same formula as eval_lejepa.rankme / stable_pretraining's rankme callback:
    s = svdvals(Z); p = s/sum(s) + 1e-5; RankMe = exp(-sum(p log p)).
    Returns {rankme, dim, rank_frac (rankme / max attainable), dead_dim_frac}.
    This is the predictor-R^2 blind spot: constant latents give R^2 ~ 1 but rankme ~ 1.
    """
    z = z.reshape(len(z), -1).astype(np.float64)
    s = np.linalg.svd(z, compute_uv=False)
    p = s / s.sum() + 1e-5
    var = z.var(axis=0)
    return {
        "rankme": float(np.exp(-(p * np.log(p)).sum())),
        "dim": z.shape[1],
        "rank_frac": float(np.exp(-(p * np.log(p)).sum()) / min(z.shape)),
        "dead_dim_frac": float((var < 1e-6 * var.max()).mean()) if var.max() > 0 else 1.0,
    }


def linear_probe_r2(z, targets, groups=None, n_seeds=5, test_frac=0.3, alpha=10.0):
    """Ridge probe from frozen latents to continuous targets (e.g. robot state).

    groups: length-N ids (scene/episode) — a group is never split across train/test,
    matching the scene-held-out convention in robust_robot_eval; None -> random split.
    Returns {r2_mean, r2_std, per_seed} (r2 uniform-averaged over target dims).
    """
    z = z.reshape(len(z), -1)
    targets = np.asarray(targets)
    groups = np.asarray(groups if groups is not None else np.arange(len(z)))
    uniq = np.array(sorted(set(groups.tolist())))
    per_seed = []
    for seed in range(n_seeds):
        test_groups = set(np.random.RandomState(seed).permutation(uniq)[: max(1, round(len(uniq) * test_frac))].tolist())
        te = np.array([g in test_groups for g in groups])
        probe = Ridge(alpha=alpha).fit(z[~te], targets[~te])
        per_seed.append(r2_score(targets[te], probe.predict(z[te])))
    return {"r2_mean": float(np.mean(per_seed)), "r2_std": float(np.std(per_seed)),
            "per_seed": [float(v) for v in per_seed]}


def distance_correlation(z, s, n_pairs=20000, distance="mse", seed=0):
    """Spearman rank correlation between latent distances and ground-truth distances.

    s: reference coordinates per frame (robot state, or timestamps within an episode);
    Euclidean distance is used on s — preprocess it (symlog/angles) before calling.
    Pairs are sampled uniformly; pair *selection* (within-episode etc.) is the caller's
    job. Returns {spearman, pvalue, n_pairs}.
    """
    rng = np.random.default_rng(seed)
    i, j = rng.integers(0, len(z), size=(2, n_pairs))
    keep = i != j
    i, j = i[keep], j[keep]
    d_z = _dist(z[i], z[j], distance)
    d_s = np.linalg.norm(np.asarray(s, dtype=np.float64).reshape(len(s), -1)[i]
                         - np.asarray(s, dtype=np.float64).reshape(len(s), -1)[j], axis=1)
    rho, pval = spearmanr(d_z, d_s)
    return {"spearman": float(rho), "pvalue": float(pval), "n_pairs": int(len(i))}


def alignment_uniformity(z_pos_a, z_pos_b, z_all=None, t=2.0, max_n=2048, seed=0):
    """Wang & Isola (2020), computed on L2-normalized latents.

    alignment = E ||za - zb||^2 over positive pairs (lower = more invariant; 0..4).
    uniformity = log E exp(-t ||zi - zj||^2) over all pairs of z_all (default z_pos_a),
    subsampled to max_n rows (lower = better spread; -> 0 when collapsed).
    The two numbers separate what the triplet ratio confounds: invariance vs anti-collapse.
    """
    norm = lambda z: (lambda f: f / np.linalg.norm(f, axis=1, keepdims=True))(
        z.reshape(len(z), -1).astype(np.float64))
    a, b = norm(z_pos_a), norm(z_pos_b)
    align = float(((a - b) ** 2).sum(axis=1).mean())
    zu = norm(z_all if z_all is not None else z_pos_a)
    if len(zu) > max_n:
        zu = zu[np.random.default_rng(seed).choice(len(zu), max_n, replace=False)]
    unif = float(np.log(np.exp(-t * pdist(zu, "sqeuclidean")).mean()))
    return {"alignment": align, "uniformity": unif}


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

    # effective_rank: full-rank gaussian high, rank-3 latents low
    z_collapsed = rng.normal(size=(4000, 3)) @ rng.normal(size=(3, 64))
    er_full, er_low = effective_rank(za), effective_rank(z_collapsed)
    print(f"rankme: full={er_full['rankme']:.1f}/64  collapsed={er_low['rankme']:.1f}/64")
    assert er_full["rankme"] > 50 and er_low["rankme"] < 5

    # linear_probe_r2: linear-in-latent targets recoverable, shuffled targets not
    w = rng.normal(size=(64, 7))
    state = za @ w + 0.1 * rng.normal(size=(4000, 7))
    groups = np.repeat(np.arange(40), 100)  # 40 fake scenes
    pr_good = linear_probe_r2(za, state, groups)
    pr_bad = linear_probe_r2(za, rng.permutation(state), groups)
    print(f"probe R2: linear-target={pr_good['r2_mean']:.3f}±{pr_good['r2_std']:.3f}  "
          f"shuffled={pr_bad['r2_mean']:.3f}")
    assert pr_good["r2_mean"] > 0.9 and pr_bad["r2_mean"] < 0.1

    # distance_correlation: state = rotation of latent -> high; random state -> ~0
    dc_good = distance_correlation(za, za @ np.linalg.qr(rng.normal(size=(64, 64)))[0])
    dc_bad = distance_correlation(za, rng.normal(size=(4000, 7)))
    print(f"dist-corr: rotated={dc_good['spearman']:.3f}  random={dc_bad['spearman']:.3f}")
    assert dc_good["spearman"] > 0.9 and abs(dc_bad["spearman"]) < 0.1

    # alignment_uniformity: tight positives -> low alignment; collapsed -> uniformity ~ 0
    au = alignment_uniformity(za, zp)
    au_collapsed = alignment_uniformity(z_collapsed, z_collapsed)
    print(f"align/unif: spread=({au['alignment']:.3f}, {au['uniformity']:.3f})  "
          f"collapsed unif={au_collapsed['uniformity']:.3f}")
    assert au["alignment"] < 0.1 and au["uniformity"] < au_collapsed["uniformity"]
    print("smoke test OK")
