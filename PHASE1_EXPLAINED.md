# Phase 1 (single-timestep), explained simply

Plain-language summary of the Phase-1 transfer-matrix run. Technical detail lives in the
training code (`world_tokenizer/train_chunks.py`, `mm_perceiver2.py`) and the per-run logs
(`logs/train_*.log`). **Numbers below are PRELIMINARY** — the generalist ("ALL") run has 1
of 5 seeds done as of writing (2026-07-07); the specialists have 5 (franka, ur5, kuka) or 2
(flexiv). Wait for the full 5-seed means before quoting figures.

## What we trained

A small "fusion brain" that takes three things a robot senses at one instant —
**what its camera sees, where its joints are, and how hard it's pushing (force/torque)** —
and squeezes them into one compact summary (a *latent*).

It learns by a fill-in-the-blank game: **hide one sense, guess it from the other two.**
To guess force from vision + joints, the model is forced to notice force-related clues in
the image — so the vision summary quietly starts carrying force/joint information it
otherwise wouldn't. The camera backbone (a frozen ViT) is never trained; only the small
~2M-parameter fusion head (a Perceiver) learns.

We trained **5 versions** to answer one question — *does one encoder work for every robot,
or do you need one per robot?*

- **4 specialists** — one per robot: flexiv (cfg1+2), ur5 (cfg3+4), franka (cfg5), kuka (cfg6+7).
- **1 generalist ("ALL")** — all 7 configs / all 4 robots together.

Each is trained 5 times (seeds) for stability, then probed on **every** robot → a 5×4
transfer matrix.

## How we tested it

The key move: after training, **feed the model ONLY the camera image** (hide joints and
force) and ask — *from just this vision summary, can a simple linear readout recover the
robot's joints and forces?* If yes, training-alongside-state taught vision to carry state.

- **Score:** R² (higher = better; 1.0 perfect, 0 = useless).
- **Baselines to beat:**
  - **raw camera features** (plain frozen ViT, no fusion) — beating it means fusion added something.
  - **PCA** (just shrinking the camera features) — beating it proves the gain is real
    *cross-sensor learning*, not just compression.
- **RankMe** — a safety check that the summary didn't secretly collapse into garbage (a dead
  latent can fake a good probe score).
- **Held-out** robots/tasks the model never trained on (split by group, so it can't cheat).

## What we found (preliminary)

**1. The generalist works — "one encoder for all robots" holds.**
The ALL encoder is **as good as or better than each specialist on that specialist's own
robot**, and it **crushes** the specialists when they're asked to work on a *different*
robot (a specialist's latent doesn't transfer better than raw vision; the generalist's
does). So: you don't need four encoders — one covers all four robots.

**2. Fusion pays off exactly where vision is blind — force.**
The vision-only summary predicts **force/contact much better than raw camera on every
robot** (a solid, consistent win, ~+0.08–0.09 R²). For **joint angles** the gain is small —
the camera can already *see* roughly where the arm is, so fusion adds little there.

**3. No collapse** — RankMe stays healthy (~150–190) across the board.

**The honest one-liner:** *cross-sensor fusion helps where the camera can't see
(force/contact), not where it already can (joint pose) — and a single encoder covers all
four robots.*

### Preliminary diagonal (each encoder on its OWN robot), vision-only z_v vs raw camera

| robot | z_v joints | raw joints | z_v **force** | raw force |
|---|---|---|---|---|
| flexiv | 0.27 | 0.23 | **0.30** | 0.21 |
| ur5 | 0.32 | 0.32 | **0.16** | 0.10 |
| kuka | 0.36 | 0.39 | **0.44** | 0.35 |
| franka | −0.31 *(hard, tiny config)* | −6.71 | — (no F/T sensor) | — |

## Caveats
- ALL run only 1/5 seeds done — means will shift; not figure-ready.
- One odd cell: **kuka joints**, where plain vision still beats fusion.
- This is **single-timestep** (one instant, no time axis). Motion/dynamics — and likely a
  bigger joint/action gain — are the **temporal** next phase.

## Next steps
1. Finish the 5-seed run → full 5×4 mean±std table.
2. **Vision-only-trained ablation** (train the same head with no state token) — makes the
   "fusion helped" claim defensible; highest priority before publishing.
3. Add triplet-accuracy (geometry) to the eval.
4. Investigate the kuka-joints cell.
5. **Temporal phase** (200 ms chunks + 1D-CNN proprioception) — where the joint/action
   payoff should show up.
