# World Encoder — the project, explained from first principles

A self-contained explainer for presenting the project (brain team, blog, new collaborators).
Technical results live in [EXPERIMENTS.md](EXPERIMENTS.md); current execution plan in
[PLAN.md](PLAN.md); full project state in [HANDOFF.md](HANDOFF.md).

## The problem

A robot is not a camera. It's a camera **plus** joint encoders, force/torque sensors, a
gripper — later audio and IMU — all streaming at different rates (camera ~10 Hz and jittery,
force/torque 100–125 Hz). Yet almost every robot learning system today (the OpenVLA
paradigm) uses a **vision-only** encoder and bolts the robot state on as an afterthought.

Our bet: robots need their own encoder — one model that turns *all* the sensor streams into
a single compact latent. Downstream models (VLAs, world models, planners) consume that
latent instead of raw pixels, and the same latent doubles as compression for shipping
sensor data edge→cloud.

## Why encoder-first

**Everything downstream can only use information that survives the encoder.** If the
representation doesn't contain force, no world model built on top can ever reason about
contact — and we proved this matters: a single video frame simply doesn't contain force
(Stage 1: force R² ≈ 0 from vision, no matter how we finetuned).

So the encoder is the load-bearing component. It's also:

- **the reusable asset** — train it once with self-supervision; every downstream task gets
  it for free;
- **the fastest thing to iterate on** — evaluated with cheap probes instead of training a
  full robot policy each time.

## Why self-supervised, why JEPA, why LeJEPA

Robot data has no labels at scale, so training must be self-supervised. Within SSL there
are three families; we rejected two for principled reasons:

| family | why not / why |
|---|---|
| **Reconstruction** (autoencoders / MAE) | Forces the latent to memorize pixel-level junk — textures, backgrounds — irrelevant to the robot. We want *world state*, not appearance. |
| **Contrastive / "equate"** | Forces two views to have *equal* embeddings. But different sensors genuinely contain **different information** (the wrist camera can't see what the side camera sees). Forcing equality collapses the latent to the *intersection* of what all sensors know. |
| **JEPA — "predict, don't equate"** ✅ | Mask one part, predict its *latent* (not its pixels) from the rest. Prediction respects information asymmetry — the model is allowed to be uncertain about what it can't see — and it's robust to missing sensors by construction (a masked modality at training time = a dead sensor at deployment time). |

JEPA's classic disease is **collapse**: a constant latent predicts itself perfectly.
**LeJEPA** is why we chose this specific lineage: its SIGReg loss (push embeddings toward
an isotropic Gaussian, tested on random 1-D slices) *provably* prevents collapse with
almost no hyperparameter tuning — plus the lab ships a pretrained ViT checkpoint (`e0`)
we warm-start from, and the `stable-pretraining` framework we build on.

### The four losses — each prevents a specific failure

| # | loss | role / failure it prevents |
|---|---|---|
| 1 | Masked latent prediction over (modality × time) tokens | The learning signal. Predict-don't-equate respects info asymmetry → no intersection-collapse; robust to missing modalities by default. |
| 2 | Per-modality SIGReg | Anti-collapse + magnitude standardizer — makes modalities commensurate before fusion. |
| 3 | Joint SIGReg on the fused latent | Keeps the world-model latent expressive (high-rank), not collapsed. |
| 4 | Action-conditioned forward prediction *(later)* | The causal engine — same-time alignment is only correlational; a robotics model eventually needs causality. |

## Our encoder, concretely

Every sensor reading becomes a token of **(value, what, where, when)**:

- **value** — per-modality tokenizer. Video → patch tokens from the **frozen** LeJEPA ViT
  (frozen because Stage 1 showed finetuning on robot video degrades the encoder and adds
  nothing). State → scale-respecting encodings: symlog for unbounded quantities, sin/cos
  for angles, 6D for rotations.
- **what** — a learned modality embedding (vision / joints / force / …).
- **where** — spatial position, for image patches.
- **when** — the *real timestamp*, via a continuous-time embedding (mTAN / Time2Vec
  style). **This is the core architectural bet:** it lets streams at different native
  rates fuse *without resampling and without padding*. Nobody in robot SSL does this
  (MSDP, the closest work, explicitly assumes synchronized observations) — this is our
  white space.
- **fuser** — a **Perceiver**: a small set of learned queries cross-attends over the whole
  heterogeneous token pile → fixed-size latent. Linear cost in token count, indifferent to
  how many sensors or which robot. With masks, one architecture spans a 6-DOF UR5 and a
  7-DOF KUKA — that's how the joint-dim problem (6 / 14 / 21 across configs) was solved:
  keep the joints in a masked 8×3 grid rather than dropping them.

Validated so far: **modality fusion at a single timestep** (Stage 2). The time axis is
Phase 2 of the current plan.

## How we differ from other approaches

Three paradigms dominate robot representation today. We're none of them, and the
differences are principled, not cosmetic.

**1. Vision-only VLAs (the OpenVLA paradigm).** A vision(-language) encoder produces the
representation; robot state is bolted on downstream as raw numbers. The encoder never sees
force, joints, or contact — so nothing built on top can recover them. We measured exactly
this: **force R² ≈ 0 from vision alone** (Stage 1). Our encoder ingests every stream as a
first-class token, so state survives *into* the latent.

**2. Late fusion — "encode each modality separately, then concatenate."** The common
multimodal recipe: a vision encoder, a proprioception encoder, a force encoder — each
trained on its own — and you `concat` their outputs into one vector. This looks like fusion
but isn't:

- **No modality learns from another.** The vision encoder is optimized without ever being
  asked about force, so its features stay force-blind. Concatenation just staples
  independent representations together; the *cross-modal* structure — the whole point — is
  never learned, only deferred to whatever consumes the concat.
- **Brittle to missing sensors.** Drop a sensor and you have a hole in a fixed-width vector;
  the downstream model faces a distribution it never trained on.
- **No shared geometry.** Separately-trained latents live in unrelated spaces at
  incommensurate scales.

We keep the per-modality *tokenizers* (symlog, sin/cos, 6D, frozen ViT) — that part is
right — but instead of concatenating we **fuse by cross-modal prediction**: mask one
modality, predict its latent from the rest. That forces vision to encode force-relevant
structure *because it is trained to predict force*. The proof is the headline number: at
eval we feed **vision only** and it predicts robot state at **R² 0.65 vs 0.52 raw / 0.42
PCA** (cfg3+4). Late fusion cannot produce that — its vision features were never shaped by
the other senses. (Per-modality SIGReg also gives every stream a shared, commensurate
geometry *before* it fuses — the standardization concatenation skips.)

**3. Generative world models (e.g. DreamZero).** Autoregressive video-diffusion models that
predict the future *as pixels* + actions, then act. They are powerful **policies / dynamics
models** — but their "representation of the world" is video itself, so they spend enormous
capacity modeling appearance (the reconstruction trap we deliberately avoid) and are
vision-centric: video can't recover the 100 Hz force stream that never entered it.

| | generative world model (DreamZero-style) | us |
|---|---|---|
| **what it is** | a policy / dynamics model — outputs behavior | an encoder — outputs a reusable latent |
| **predicts in** | pixel/video space (diffusion, autoregressive) | latent space (JEPA — predict, don't reconstruct) |
| **sensors** | video + actions (vision-centric) | multi-rate fusion: F/T @100 Hz, joints, gripper, vision @10 Hz |
| **role** | *consumes* representations | *produces* the representation others consume |

**Complementary, not competing.** Autoregression isn't a weakness — it's how DreamZero
models time and causality, and on the *temporal* axis it's genuinely ahead of us (time is
our Phase 2). The honest, strongest framing: **a generative world model like DreamZero is
exactly what should consume our latent instead of raw video** — that's how it would finally
get contact and force into its world state, which pixels alone can't give it.

## The evidence that it works

Stage 2, the headline experiment: train the Perceiver on vision + state with masked
cross-modal prediction; at eval feed it **vision only** and ask a linear probe to predict
robot state (scene/group-held-out, 5 seeds):

| latent (→ predict robot state, R²) | cfg3 (24k frames) | cfg3+4 (86k frames) |
|---|---|---|
| **cross-modal Perceiver `z_v` (256-d)** | **0.551 ±0.018** | **0.653 ±0.008** |
| raw vision (768-d, frozen ViT) | 0.257 ±0.075 | 0.516 ±0.010 |
| PCA-256 of vision (compression control) | 0.134 ±0.047 | 0.418 ±0.015 |

Two things make this meaningful:

1. **Beating the PCA control proves the gain is cross-modal learning, not compression** —
   a latent 3× smaller than raw vision predicts the robot far better.
2. **Only vision enters at eval** — so the encoder *learned to read robot-relevant
   structure out of pixels* because it was trained alongside the state stream.

RankMe stays ~211 throughout (no collapse). Gains hold on every seed.

## How we evaluate — and why each piece

Standing rule, learned the hard way in Stage 1: **one metric will lie to you.** (Our
task-id probe looked like a regression; it was actually saturated and measuring ImageNet
appearance.) So every eval pairs:

- **RankMe (effective rank)** — label-free collapse detector. Probe R² alone can be gamed
  by a collapsed latent.
- **Linear probe R²** on robot state / force / contact — is the information *in* there?
- **Triplet accuracy** — is the geometry right: nearby world states nearby in latent,
  invariant to nuisance (camera view), sensitive to what actually changed?
- **Controls** — raw vision; PCA-256 (compression control); vision-only-*trained*
  ablation (protects the headline from "you just trained an in-domain encoder").
- **Split hygiene** — held out by (cfg, task, user) *group*, frozen in a committed CSV.
  One user repeating one task ten times produces near-duplicate scenes; frame-level
  splits leak so badly the probe hits 1.0.
- **The transfer matrix** (Phase 1) — per-embodiment encoders vs one joint encoder, each
  evaluated on every robot: the direct empirical test of the "one encoder across robots"
  thesis, and the honest price tag on cross-embodiment transfer.

## The decoder — what it's for

The decoder is **not** part of the training signal — training on reconstruction would drag
the latent right back to memorizing pixels (the autoencoder trap; it's why this is a
world-*encoder*). The decoder is our **superpowered linear probe**: a linear probe tells
you information *exists* in the latent; a generative decoder *shows you what the latent
knows*.

- Stage 3: a cheap **robot_state decoder** on frozen latents — quantifies latent content.
- Later: **PixNeRD → latent diffusion** pixel decoder. Diffusion rather than MSE because
  the latent intentionally discards pixel detail, so decoding is one-to-many — a
  generative decoder samples plausible detail instead of averaging into blur.
- The decoder is also the product half of the compression pitch: transmit the tiny
  latent, decode at the cloud.

## Design decisions we made along the way (rapid fire)

- **cfg3 first** — smallest slice, fastest iteration; scale only what's validated.
- **Freeze vision** — Stage-1 evidence, not ideology.
- **Robot-agnostic state via masking, not truncation** — keep joints, mask unused rows.
- **Native rates, no interpolation** — tick-anchored chunks with real timestamps; this
  quietly pre-builds the temporal phase.
- **One fixed external camera for v1** — multi-cam was adding noise; wrist cams are
  identifiable per config, so exclude them deterministically. Multi-view learning comes
  later as cross-view *prediction*, never latent equality (the dual-arm / wrist-cam
  information-asymmetry objection is correct, and predict-don't-equate already resolves it).
- **cfg5's force stays but masked** — Franka has no physical F/T sensor; conveniently
  that makes it a built-in missing-modality robustness test.
- **Audio, actions, temporal, decoder each gated behind their own stage** — the first
  public run rests only on what's proven.

**In one paragraph:** a provably-non-collapsing, prediction-based, multi-rate sensor
encoder for robots — validated one claim at a time, with an eval suite designed not to
fool ourselves. Next: the Phase-1 matrix run (one encoder vs per-robot encoders, full
RH20T), then time.

---

## Appendix — saying it in one line

Candidate tagline: *"sensory compression and entanglement is intelligence."* Catchy
direction; two words need fixing before it goes public:

1. **"Entanglement" is self-sabotaging.** In representation learning, *entangled* is the
   pejorative — and our own roadmap lists **disentanglement** as a latent-structure goal.
   It also reads as a quantum buzzword. The concept actually meant — unifying multiple
   sensory streams into one percept — has a proper name from neuroscience: **binding**
   (the binding problem). "Sensory binding" says the right thing and earns a legitimate
   reference instead of a wince.
2. **"Compression … is intelligence" invites our own ablation as a rebuttal.** The
   PCA-256 control *is* pure compression, and it lost (0.42 vs our 0.65). The Stage-2
   result is precisely that the gain is **cross-modal prediction, not compression**. The
   defensible claim is *predictive* compression: compress in a way that lets each sense
   predict the others. ("Is intelligence" also overclaims — same-time alignment is only
   correlational; causality via actions is still ahead of us. "Is the seed of / begins
   with" keeps the punch without the exposure.)

Variants that survive scrutiny:

- **"Predictive compression across the senses is the beginning of intelligence."** —
  closest to the original, fixed.
- **"Intelligence starts where the senses compress into one predictive picture of the
  world."** — blog-friendly.
- **"Compress the senses, bind them by prediction."** — terse, method-shaped, almost a
  design principle.

The original works as a slide *provocation* — but be ready for "then why did your PCA
baseline lose?" The second variant is the one to put in writing.
