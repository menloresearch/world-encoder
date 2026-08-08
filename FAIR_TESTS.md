# Fair-test checklist

Read this before launching an arm and before writing a number into a doc. Every rule below is here
because it cost us a wrong conclusion on 2026-08-05 — the example under each rule is the actual failure,
not a hypothetical. Nine of the ten fired in a single day.

Two commands do most of the work:

    python scripts/pusht_read.py <plan logs...>      # correct PushT metric + iteration counts
    python scripts/fair_test.py --a 148 200 0.86 --b 168 200 0.90   # deltas, CIs, p, power

---

## 1. Record the experimental CELL next to every external number

Not the paper — the **cell**: planner × control mode × task × budget.

> **Cost:** we justified a whole 3-arm GPU programme with "adversarial world modelling takes CEM 78→94
> on this harness". That is the **open-loop** row. The **MPC** row — our configuration — is **92 → 92,
> zero gain**. Both rows are in the same table. Three arms and ~40 GPU-minutes aimed at a saturated cell.

Adversarial-vote verification does **not** protect against this: the claim was verified 3-0 as a
statement about the table, and still applied to the wrong row.

## 2. Know what your metric actually measures before you quote it

> **Cost:** `Success rate:` in DINO-WM plan logs is **instantaneous**, and is printed both by the outer
> MPC loop *and* from inside CEM every optimisation step. Most lines are the optimiser talking to itself
> mid-solve. We read them as cumulative for a full day. The real metric is the sticky `is_success` mask.
> Corrected board: dense 0.90 in **2** iterations vs ensemble 0.28 in **127** — the iteration count was
> invisible and it doubled the apparent gap.

Rule: `pusht_read.py` for outcomes; always report the iteration count alongside.

## 3. Every intervention needs a matched null-intervention arm

The comparator is *your own arm with the intervention removed* — never a historical number.

> **Cost:** three adversarial arms got 8 extra epochs of finetuning that the incumbent never had. The
> mechanism result ("realised cost stops worsening") looked like a pass. The clean-finetune control —
> same script, same epochs, `--adv-weight 0` — showed the effect **more strongly**. It was the extra
> epochs. Retracted.

If you cannot name the null-intervention arm, you do not have an experiment.

## 4. Held-out splits, always. In-sample statistics at d ≳ n are meaningless

> **Cost:** probe R² of **exactly 1.000** on a 75,264-dim dense feature grid against ~1,400 samples —
> pure interpolation, reported as decodability. Held-out R² was 0.970, so the conclusion survived, but it
> was luck. `pusht_cost_rowspace.py` now defaults to a 70/30 split and prints in-sample alongside.

## 5. No single-cell reads on a high-variance axis

> **Cost:** camera severity 0, seed 1234 → **−0.3pp (parity)**. Reported as "the answer". Seed 7 →
> **−18.7pp (full loss)**. Four seeds gave −7.3pp ± 7.9. Two opposite conclusions from one axis.

Camera-type axes draw a random *direction*, so per-seed spread is 10-70× the photometric axes. Minimum
3 cells, report the spread, and prefer a **deterministic direction basis** (`camera_dir`) so the variance
becomes structure instead of noise.

## 6. Power-check before launching, not after

> **Cost:** a whole 2×2 grid at n=50 in the 0.0-0.3 band. Nothing was significant; the clean-finetune
> control (no fix at all) tied the best intervention arm. Detecting 0.04 → 0.12 needs **179** rollouts;
> a 2× win at our scale needs **258**. n=50 is fine for the extremes we registered (7 rollouts detect
> the 0.72 gate) and useless for the middle.

`fair_test.py` prints the minimum detectable effect for your n. If your expected effect is below it,
either raise n or don't run the arm.

## 7. Arms must share the training regime — log effective batch and LR

> **Cost:** our dense control ran under `accelerate --num_processes 4`, where `dataloader.batch_size` is
> **per-process** → 4× effective batch at unchanged LR. Our compact arm ran single-GPU. We compared them
> and concluded "our recipe is 30pp worse". The comparison spanned two optimiser regimes.

Record launcher, process count, effective batch and LR in the run note. Differences here invalidate
everything downstream.

## 8. Report percentage change from each arm's own clean score, and print the clean score

> **Cost:** "+15.5pp on lighting" was measured where the baseline had already collapsed to 0.14-0.22,
> while "−18.7pp on camera" was measured where it was still 0.74. Absolute points across cells at
> opposite ends of the range are indefensible. In relative terms the real result is *stronger*
> (baseline loses 84% of its performance, ours 49%) — the bad framing was understating us.

## 9. Calibrate perturbation magnitude before comparing axes

> **Cost:** our headline compared **lighting s2 (SSIM 0.886)** against **camera s1 (SSIM 0.484)** — a
> 4.5× stronger perturbation. Worse, the families did not overlap at all: photometric maxed at 0.886
> while camera started at 0.484, so no severity pair in the runner was comparable. Building a matched
> cell (`camera s0`, SSIM 0.858) moved the geometric penalty from −17.9pp to **−7.3pp**. About 60% of a
> number we had quoted for two days was a severity artefact.

## 10. Before claiming a mechanism, name the number that would kill it — and check whether you already have it

This is the one that cost the most, and it is free to apply.

> **Cost:** we built a full day's narrative on "compressed latents are vulnerable to planner exploitation
> where dense features resist" — cost-geometry enrichment, the support-walk mechanism, six eliminated
> causes. The killing number was **DINO-WM Table 2, already written in our own memory notes**: DINO
> **CLS**, a single 384-d global vector — 5× more compact than ours — scores **0.44** where we score
> **0.06-0.08**, same harness, same recipe, frozen. Compactness does not explain our failure. We quoted
> that table all day as a *target* and never once turned it around to falsify our own story.

Ask it out loud before the claim goes in a doc: *"what result would make this false, and is it already
in our notes or in the baseline paper's tables?"*

---

## Pre-registration stub — paste into PROGRESS.md before launching

```
ARM: <name>
QUESTION: <the one thing this decides>
COMPARATOR: <your own arm minus the intervention; NOT a historical number>
METRIC: <exact quantity + how read; iteration/step count reported>
REGIME: launcher / processes / effective batch / LR  (must match comparator)
N + POWER: n=<>, minimum detectable effect=<from fair_test.py>; expected effect=<>
READ RULE: pass=<>  iterate=<>  kill=<>
PREDICTION: <state it so it can be wrong>
KILL-MY-CLAIM: <the number that would falsify this, and whether we already have it>
```
