"""Fair-test arithmetic in one command (see FAIR_TESTS.md rules 6 and 8).

Compares two arms as counts-out-of-n, each against its OWN clean score, and prints the things we
kept getting wrong on 2026-08-05: relative change (not just absolute points), Wilson intervals, an
exact test, and — the one that matters before launching — the MINIMUM DETECTABLE EFFECT at this n.

  # two arms, pooled counts, with their own clean references
  python scripts/fair_test.py --a 148 200 0.86 --b 168 200 0.90 --labels sb hybrid_base

  # per-cell spread as well (rule 5: never read a single cell on a noisy axis)
  python scripts/fair_test.py --a 148 200 0.86 --b 168 200 0.90 \
      --a-cells 0.68 0.74 0.80 0.74 --b-cells 0.88 0.82 0.84 0.82

  # power only: what can n=50 even see?
  python scripts/fair_test.py --power 50 --baseline 0.04
"""
import argparse
from math import comb, sqrt


def fisher2(a, b, c, d):
    n = a + b + c + d

    def p(x):
        return comb(a + b, x) * comb(c + d, a + c - x) / comb(n, a + c)

    lo, hi = max(0, a + c - (c + d)), min(a + b, a + c)
    obs = p(a)
    return sum(p(x) for x in range(lo, hi + 1) if p(x) <= obs + 1e-12)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * ((ph * (1 - ph) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, c - h), min(1.0, c + h)


def n_for(p1, p2, power=0.8):
    """Rollouts per arm to detect p1 vs p2 at alpha=.05 two-sided."""
    za, zb = 1.96, {0.8: 0.8416, 0.9: 1.2816}.get(power, 0.8416)
    pbar = (p1 + p2) / 2
    if p2 == p1:
        return float("inf")
    num = (za * sqrt(2 * pbar * (1 - pbar)) + zb * sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return num / (p2 - p1) ** 2


def mde(n, baseline, power=0.8):
    """Smallest p2 > baseline detectable with n per arm — the pre-launch sanity number."""
    p2 = baseline
    while p2 < 0.999:
        p2 += 0.005
        if n_for(baseline, p2, power) <= n:
            return p2
    return float("nan")


def spread(cells, clean):
    rel = [(c / clean - 1) * 100 for c in cells]
    m = sum(rel) / len(rel)
    if len(rel) < 2:
        return m, 0.0, min(rel), max(rel)
    sd = (sum((r - m) ** 2 for r in rel) / (len(rel) - 1)) ** 0.5
    return m, sd, min(rel), max(rel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs=3, type=float, metavar=("K", "N", "CLEAN"))
    ap.add_argument("--b", nargs=3, type=float, metavar=("K", "N", "CLEAN"))
    ap.add_argument("--labels", nargs=2, default=["arm_a", "arm_b"])
    ap.add_argument("--a-cells", nargs="*", type=float, default=[])
    ap.add_argument("--b-cells", nargs="*", type=float, default=[])
    ap.add_argument("--power", type=float, help="n per arm, for a power-only report")
    ap.add_argument("--baseline", type=float, default=0.04)
    args = ap.parse_args()

    if args.power:
        n = args.power
        b = args.baseline
        print(f"POWER at n={n:.0f} per arm, baseline {b:.2f} (alpha .05, 80% power)")
        m = mde(n, b)
        print(f"  minimum detectable success rate : {m:.3f}   (i.e. +{(m-b)*100:.1f}pp)")
        print("  rollouts needed for common effects:")
        for tgt in (b + 0.04, b + 0.08, 0.28, 0.5, 0.72):
            if tgt <= b:
                continue
            print(f"    {b:.2f} -> {tgt:.2f} : n = {n_for(b, tgt):7.0f}"
                  f"{'   <-- OK at this n' if n_for(b, tgt) <= n else '   UNDERPOWERED'}")
        if not args.a:
            return
        print()

    if not (args.a and args.b):
        ap.error("need --a and --b (or --power)")

    (ka, na, cla), (kb, nb, clb) = args.a, args.b
    ka, na, kb, nb = int(ka), int(na), int(kb), int(nb)
    la, lb = args.labels
    pa, pb = ka / na, kb / nb
    ra, rb = (pa / cla - 1) * 100, (pb / clb - 1) * 100

    print(f"{'arm':14} {'perturbed':>12} {'clean':>7} {'rel change':>11} {'95% CI':>16}")
    for lbl, k, n, p, cl, r in ((la, ka, na, pa, cla, ra), (lb, kb, nb, pb, clb, rb)):
        lo, hi = wilson(k, n)
        print(f"{lbl:14} {k:5}/{n:<5} {p:.3f} {cl:>7.2f} {r:>+10.1f}% {f'[{lo:.3f},{hi:.3f}]':>16}")

    print(f"\n  absolute delta   {(pa-pb)*100:+.1f}pp")
    print(f"  RELATIVE gap     {ra-rb:+.1f}pp   <- quote this one (rule 8)")
    print(f"  Fisher exact p   {fisher2(ka, na-ka, kb, nb-kb):.4f}")
    print(f"  MDE at n={na}    {mde(na, pb):.3f} vs comparator {pb:.3f} (rule 6)")

    if args.a_cells and args.b_cells:
        ma, sda, loa, hia = spread(args.a_cells, cla)
        mb, sdb, lob, hib = spread(args.b_cells, clb)
        gaps = [(a / cla - 1) * 100 - (b / clb - 1) * 100
                for a, b in zip(args.a_cells, args.b_cells)]
        mg = sum(gaps) / len(gaps)
        sdg = (sum((g - mg) ** 2 for g in gaps) / (len(gaps) - 1)) ** 0.5 if len(gaps) > 1 else 0.0
        print(f"\n  per-cell relative gaps ({len(gaps)} cells): "
              f"{' '.join(f'{g:+.1f}' for g in gaps)}")
        print(f"  mean {mg:+.1f}pp  sd {sdg:.1f}  range {min(gaps):+.1f} to {max(gaps):+.1f}")
        if sdg > abs(mg):
            print("  ⚠ sd EXCEEDS the mean — per-cell spread dominates. Do NOT quote the mean alone")
            print("    (rule 5); report the spread, and prefer a deterministic perturbation basis.")


if __name__ == "__main__":
    main()
