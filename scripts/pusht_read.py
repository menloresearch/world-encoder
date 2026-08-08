"""Read PushT planning outcomes the RIGHT way (PROGRESS pt. 33).

`Success rate:` in a plan log is INSTANTANEOUS and is printed both by the outer MPC loop
(planning/mpc.py) and from inside CEM every eval_every optimisation steps (planning/cem.py) —
so quoting it conflates "the optimiser talking to itself mid-solve" with "the planner solved
the task". The cumulative outcome is `self.is_success`, a sticky OR mask that only rises.

This prints, per log: cumulative success from the sticky mask, how many outer MPC iterations
produced it (both matter — 0.90 in 2 replans is not 0.90 in 127), and the within-solve CEM
trace for the current solve so exploitation collapse is visible.

  python scripts/pusht_read.py results/pusht/logs/*.log
"""
import re
import sys
from pathlib import Path


def read(path):
    txt = Path(path).read_text(errors="ignore")
    blocks = re.findall(r"self\.is_success:\s*\[(.*?)\]", txt, re.S)
    inst = [float(x) for x in re.findall(r"Success rate:\s+([0-9.]+)", txt)]
    cum, per_iter = None, []
    for b in blocks:
        toks = re.findall(r"True|False", b)
        if toks:
            per_iter.append(sum(t == "True" for t in toks) / len(toks))
    if blocks:
        toks = re.findall(r"True|False", blocks[-1])
        if toks:
            cum = (sum(t == "True" for t in toks), len(toks))
    return cum, len(blocks), inst, per_iter


def main():
    paths = sys.argv[1:]
    if not paths:
        print(__doc__)
        return
    print(f"{'arm':24} {'per-MPC-iteration cumulative (matched comparison)':50}")
    for p in paths:
        cum, iters, inst, per_iter = read(p)
        name = Path(p).stem
        if cum is None:
            # no outer read yet: the arm is still inside its first CEM solve
            tail = " ".join(f"{v:g}" for v in inst[-12:])
            print(f"{name:24} NO OUTER READ YET (still solving)  "
                  f"cem-evals={len(inst)}  recent-within-solve: {tail}")
            continue
        s, n = cum
        peak = max(inst) if inst else float("nan")
        last = inst[-1] if inst else float("nan")
        flag = "  <-- within-solve COLLAPSE" if inst and peak >= 0.1 and last == 0.0 else ""
        traj = " ".join(f"{v:.2f}" for v in per_iter[:8])
        print(f"{name:24} {traj:50} | final {s}/{n} = {s/n:.2f} @{iters}it "
              f"(within-solve peak {peak:g}, last {last:g}){flag}")


if __name__ == "__main__":
    main()
