# Encoder-only metrics

How good is the encoder, *without* the predictor?

The pipeline metric so far is the predictor's R² (latent at T from T-1). That number cannot stand
alone as an encoder metric: a collapsed encoder emits (near-)constant latents, which the predictor
fits perfectly — R² ≈ 1, encoder useless. It also scores the encoder and predictor jointly, so a
regression in one can hide behind an improvement in the other. The metrics here read the latent
geometry directly, with the encoder frozen and no predictor in the loop.

Planned suite (all in `metrics.py`):

All computations are implemented in `metrics.py` (data-agnostic: they take precomputed latent
arrays; frame/pair/triplet *selection* is the open TODO below). Smoke test:
`python -m world_tokenizer.metrics.metrics`.

| metric | what it measures | notes |
|---|---|---|
| `triplet_accuracy` | latent geometry: invariant to nuisance, sensitive to world state | per-tier acc + bootstrap CI + margins |
| `effective_rank` | collapse (the predictor-R² blind spot) | RankMe, same formula as `eval_lejepa.py` / stable_pretraining |
| `linear_probe_r2` | information content (Stage-2 gate already does a version) | Ridge, group-held-out splits, mean±std over seeds |
| `distance_correlation` | continuous geometry (Spearman: latent vs state distance over frame pairs) | caller preprocesses state and picks the pair population |
| `alignment_uniformity` | invariance and spread as two separate numbers (Wang & Isola 2020) | on L2-normalized latents |

Minimal headline trio: effective rank (collapse) + probe R² (information) + triplet accuracy
(geometry).

## Triplet accuracy — decided design

**Definition.** Anchor `(v_t, s_t)` from a held-out episode, a positive, and a negative; encode all
three, score `d(z_a, z_pos) < d(z_a, z_neg)`. Report per negative tier: accuracy (with bootstrap
CI) and the margin distribution `d_neg − d_pos` — accuracy alone hides barely-passing encoders.
Sample ~5k triplets per tier.

**Why not "latent distances should correlate with pixel MSE"** (the first idea, rejected):

- It rewards preserving pixel geometry — the identity function scores perfectly — while the point
  of a JEPA encoder is to *discard* pixel nuisance and keep abstract structure.
- It penalizes exactly the desirable behaviors: invariance to noise/lighting (large pixel MSE,
  ~zero latent MSE should be *good*) and sensitivity to small-but-meaningful changes (gripper moves
  5 cm ≈ small pixel MSE, should be *large* latent MSE).
- Pixel MSE is a poor ground truth anyway: lighting changes swamp it, and a semantically unrelated
  frame can sit at modest pixel distance.

**Positives** (scored separately):

1. temporal neighbor `t+k`, small `k` (~0.2 s) — the natural JEPA positive, no artificial
   corruption needed; require robot-state distance below a threshold to avoid fast-motion frames.
2. nuisance-corrupted anchor (image noise / jitter / blur, state untouched) — measures invariance.

**Negative ladder**, hardest first. One easy tier saturates at ~100 % from epoch zero (any encoder
retaining crude appearance statistics passes); the informative signal is in the hard tiers, and the
per-tier accuracy *profile* is the metric.

| tier | negative | measures |
|---|---|---|
| 1 | same episode, temporally distant (`|Δt|` > threshold AND state distance > percentile, to filter revisited poses) | world-state sensitivity — same scene/objects/background, only the state differs |
| 2 | same task, different episode | layout/phase sensitivity |
| 3 | same config, different task | activity sensitivity (should be ≈100 %) |
| 4 | different config | sanity floor only — the easiest tier, never the headline. (Training targets the full cfg1–7 dataset; cfg3-only is just the current POC, where this tier is additionally OOD and even less informative) |
| 5 | modality mismatch: anchor's own image with a distant frame's state `(v_t, s_t′)`, and symmetrically `(v_t′, s_t)` | fusion — does the latent actually use both inputs? |

**On circularity.** Selecting negatives from metadata (task/episode/config IDs, timestamps) and raw
sensor state is *not* circular — those exist independent of any learned model. Circular would be
selecting negatives with the encoder itself or its training loss. The real trap in the Stage-2
setup is different: robot state is an encoder *input*, so purely state-selected negatives can be
passed by copying the state input into the latent and ignoring vision. Tier 5 exists to close that
shortcut (it cannot be solved from one modality), and tier 1's time-based selection doesn't reduce
to state distance alone.

**Scoring decisions.** Distance = MSE on the fused latent (SIGReg standardizes magnitudes;
cross-check that cosine gives the same ranking — `metrics.py` computes both). Held-out episodes
only. Headline numbers: tier 1 and tier 5.

**Known blind spot.** None of the proxies see object state (cup moved, arm didn't). Tier-1
temporal negatives cover it partially (time passing usually means the scene changed). Full coverage
needs object tracking — deferred.

## TODO — triplet/data selection

The computation in `metrics.py` is data-agnostic on purpose; selection is the open half.

- Pick thresholds empirically by eyeballing sampled triplets: positive `k`, tier-1 minimum `Δt`,
  state-distance percentile for false-negative filtering.
- **Literature review** on triplet-style encoder evals. Known precedents to start from:
  - FaceNet-style *verification accuracy* and deep-metric-learning *Recall@K* (CUB-200, Cars196,
    SOP) — class labels as ground truth. Confirms the pattern: categorical labels → triplets. Our
    analogue of "class" is the (task, episode, time) hierarchy.
  - SSL evals use kNN accuracy on labels the same way.
  - Time-Contrastive Networks (Sermanet et al., 2017) — closest precedent: robot video, *time* as
    the supervisory signal for positives/negatives, no class labels. Essentially tier 1.
- **Bigger vision encoder (e.g. DINOv2) or VLM descriptions to pick negatives.** Precedent: LPIPS
  uses a trained network as a distance judge. Sensible as a *mining* tool (find hard negatives
  inside a tier, filter false negatives); risky as *ground truth* — the metric would then measure
  agreement with the teacher's vision-only geometry, which is blind to robot state and imports the
  teacher's biases. Keep truth = metadata + time + state; let the big model rank within tiers.
