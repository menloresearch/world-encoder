# World Autoencoder — v0 Model Architecture (Encoder)

*Scope: the **encoder** that realizes the transformation below. Loss functions, quantization
details, and the decoder are out of scope for this v0 spec. Companion to
`world-autoencoder-thesis-summary.md`. Target dataset: `USC-PSI-Lab/humanoid-everyday`.*

---

## 0. The transformation

Naive form (no compression — a latent per timestep ignores temporal redundancy):

```
S(t × m × s)  ->  L(t × d)
```

Corrected — **chunk first, then compress**:

```
S(t × m × s)  ->  S(n × c × m × s)  ->  L(n × d)
```

- `t` = time (samples), `m` = modality index, `s` = per-modality feature dim (ragged — each
  modality has its own `s`, and at native rates its own `t`).
- `n` = number of temporal chunks, `c` = chunk length (frames/samples per chunk). `t = n · c`.
- `d` = latent width. **One latent token per temporal chunk** → time compression ratio = `c`.

> Temporal redundancy is the primary compression target. Spatial + cross-modal compression
> happens *inside* a chunk via the Perceiver bottleneck; temporal compression happens *across*
> the `c` frames collapsed into one chunk latent.

---

## 1. Target dataset (`humanoid-everyday`)

| Group | Fields | Note |
|---|---|---|
| Vision | `observation.rgb.egocentric`, `observation.depth.egocentric` | per-frame image grids |
| Spatial | `observation.lidar` | point list |
| Proprio | `observation.arm_joints`, `leg_joints`, `hand_joints` | low-dim vectors |
| Inertial | `observation.imu.{quaternion(4), accelerometer(3), gyroscope(3), rpy}` | low-dim |
| Odometry | position / velocity / orientation (rpy + quat) | low-dim |
| Tactile | `observation.tactile.values` (+ sensor ids) | sensor grid |
| Action | `action` | input (optional) / IDM target |
| Language | task index → task description (260 tasks, 7 categories) | episode-level |

**Caveat:** all streams are resampled to **30 Hz**, which flattens the cross-rate imbalance
that constitutes the thesis's headroom signal. The pipeline should be built **rate-agnostic**
so native rates (the v0.5 test) can be re-injected without touching the model.

---

## 2. Architecture, layer by layer (each choice justified)

```
 raw streams            tokens (width D)         chunk latents        codec output
 S(t×m×s)   ──L1──▶  per-modality tokens  ──L3──▶  n × {q latents} ──L4──▶  L(n×d)
   │                      ▲                          (Perceiver/ACA           (causal
   │                      │ L2: robot-state           bottleneck)             latent
   └─ native rates ───────┘  FiLM / AdaLN                                     transformer)
      = variable token
      count per modality
```

> **Lineage / what makes this a codec, not HPT.** This stem→trunk shape mirrors HPT
> (`2409.20537`): per-modality stems → a shared trunk → latent tokens. The differences are the
> whole contribution — HPT has **no decoder** (it's for policy learning, not reconstruction), is
> **vision + proprioception only**, and is **not causal/streaming**. Our L2 reconstruction-forcing
> conditioning, the L3 rate-distortion bottleneck, the full sensor suite (tactile/force/IMU at
> native rates), and L4 streaming are exactly where we diverge.

### L0 — Stream loaders (native-rate aware)
Each modality keeps its own clock. Over one chunk of wall-clock `Δt = c/30 s`, native rates
give a **different sample count per modality** (few RGB frames, many IMU samples). Don't
resample — let counts vary. The set-based fusion in L3 absorbs this for free. (On the shipped
30 Hz data every count equals `c`; native-rate is the v0.5 upgrade that makes the headroom
claim testable.)

### L1 — Per-modality tokenizers → common width `D`
Map each modality's raw `s`-dim signal to tokens of width `D`:
- **RGB / depth** — per-frame patch encoder. *Either* a from-scratch CNN/ViT patch-embed,
  *or* a **pretrained video-AE encoder** — treating the system as a robot-state-enriched
  **video** autoencoder is the fastest path to a working model. Candidate backbones: **RVM**
  (Recurrent Video Masked AE, `2512.13684`) — recurrent, causal, linear-in-horizon, trained by
  predicting masked *future* frames (a built-in predictive-reconstruction objective);
  **Adaptive 1D Video Diffusion AE** (`2602.04220`) — video → variable-length 1D discrete token
  sequence + diffusion decoder; or a Cosmos/MagViT-2 tokenizer. Yields tokens per frame/clip.
- **LiDAR / point cloud** — PointNet-lite → tokens.
- **Proprio / odometry / IMU** — per-*chunk* DCT / frequency-domain tokenizer (FAST-style,
  `2501.09747`): map the chunk's smooth low-dim signal to the frequency domain, drop
  high-frequency coefficients, quantize → tokens. Frequency-domain chunking exploits the
  temporal redundancy of smooth streams far better than a per-sample MLP (the fallback). IMU
  jitter handling à la Mojito `2502.16175`.
- **Tactile** — small MLP / 1D conv over the sensor array → token(s).
- **Action** — same FAST-style DCT tokenizer (`2501.09747`), or a VQ action tokenizer
  (VQ-VLA `2507.01016`); optional input, or hold out as the IDM target.
- **Language/task** — text embedding → 1 episode-level conditioning token.

Each token also carries: a **time-within-chunk positional encoding**, a **learned
modality-type embedding** (so fusion knows IMU≠vision), and a sensor-id embedding where
relevant (tactile pads). *Native rates fall out here:* variable token counts per modality are
fine because L3 is set-based.

### L2 — Robot-state → vision conditioning (FiLM / AdaLN)
Robot state is injected into the video encoder via a small conditioning head: it maps the
chunk's low-dim state (proprio + IMU + odom) → modulation parameters, applied to the
vision/depth tokens:
- **FiLM** (`1709.07871`) — per-channel affine `γ ⊙ feat + β`; use if the vision branch is a CNN.
- **AdaLN** — the transformer counterpart (as in DiT) — state generates scale/shift/gate at
  each vision transformer block's LayerNorm; use if the branch is a ViT.

Why this matters: it makes vision encoding *conditioned on the body* ("I am gripping" reshapes
features near the hand) rather than running two independent encoders side by side. It is also a
**cheap first instance of the cross-modal term MJEPA says we need** — robot state actively
informs the vision latent.

### L3 — Temporal chunking + per-chunk Perceiver compression (the bottleneck)
Window tokens into `n` chunks. For each chunk, gather **all** its multimodal tokens (variable
count) into one set. A bank of **learned latent queries** (`q` ≈ 16–64, width `D`) does
**asymmetric cross-attention** over the set — the Perceiver IO encoder pattern — followed by a
few latent self-attention blocks. Project/pool the bank → **one chunk
latent `d`** (or keep the small bank; design choice). Output: `n × d`.

Why the Perceiver pattern here specifically:
1. **Modality-agnostic** — one block ingests vision, IMU, tactile with no per-modality plumbing.
2. **Input-length-invariant** — exactly what native, async sampling rates demand.
3. The latent bottleneck **is** the rate-distortion knob (`q`, `d` set the bitrate). This is the
   layer that earns the word "codec." *Option:* make `q` **adaptive** per chunk — more tokens for
   complex/high-motion chunks — as in the Adaptive 1D Video Diffusion AE (`2602.04220`).

### L4 — Cross-chunk causal latent transformer (streaming)
The `n` chunk latents pass through a **causal** transformer so each is contextualized by the
past → temporal coherence + a *streaming* codec (the Mimi/Moshi pattern). Output: **`L(n × d)`**
— the deliverable, and the state sequence a world model (C2) rolls out on. *Alternative:* carry a
**recurrent latent state** across chunks (RVM, `2512.13684`) — naturally causal and **linear** in
horizon rather than quadratic, which suits long-horizon streaming better than a full causal
transformer.

Optional (deferred to v0.5): **RVQ** with the split-RVQ trick — early codes carry
semantic/predictive signal, residual codes carry reconstruction detail. v0 can stay continuous.

---

## 3. Decoder + the joint-vs-independent experiment (sketch; out of scope for v0)

**Decoder.** Mirrors the encoder: `L(n×d)` → decode heads → reconstruction. The decoder is
**polymorphic** — the same latent feeds different heads by use case: *raw-signal* decode for
transport/teleop, *action* decode for control, and later *text* decode for cloud monitoring and
*3DGS/geometry* decode for spatial reasoning. Reconstruction is the training signal that forces
information retention, not something run in the control loop. (A diffusion decoder — e.g. Adaptive
1D Video Diffusion AE `2602.04220` — is a high-quality *offline* option but carries the GAIA-2
latency caveat, so it is unsuitable for the reactive path.)

**The experiment.** The thesis success criterion is the full joint-vs-independent test:

> At equal total bitrate, the **joint** codec over all modalities reconstructs every modality
> better than **N independent** per-modality codecs — and the gap **grows during cross-modal
> events** (e.g. contact). Build the N-independent baseline from day one; it *is* the experiment.

The **video-anchored** comparison (video+state with L2 on, vs video-only with L2 off) is the
**first milestone** toward that — it isolates "does robot state help the video codec" — but it is
*not* the full claim; don't let the POC collapse to it (keep ≥1 non-vision modality reconstructed
and first-class).

Run **both** objective variants — pure-reconstruction joint codec vs joint codec **+** an explicit
**cross-modal predictive term** (mask one modality's tokens at L3, predict its decode from the
rest — the MJEPA / `2009.01791` APD mechanism, general form in `2411.00522`). Otherwise the
joint-vs-independent test can come out negative for the wrong reason — a bad objective, not absent
headroom.

---

## 4. Open decisions to settle

1. **Sampling rate** — 30 Hz (fast POC, but flattens the headroom signal) vs native rates
   (harder plumbing, but the only way L3 demonstrates the claim). → Build rate-agnostic now,
   prototype on 30 Hz, re-inject native rates as the first real test. (v0 / v0.5 split.)
2. **Scope guardrail** — the "robot-state-enriched video AE" framing is pragmatic *and* consistent with the
   experiment (video = anchor, robot state = added modality). But don't let it quietly collapse to
   "vision + a conditioning head": keep ≥1 non-vision modality (tactile/force/IMU) **reconstructed
   and first-class**, or the moat evaporates.
3. **Pretrained vs scratch vision** — a pretrained video tokenizer is the fastest route to a working baseline. Watch the
   **freeze-ceiling** (thesis §6): a frozen latent may lack reconstruction detail → plan a small
   learned residual encoder.
4. **Objective** — out of scope for this v0 spec. `2411.00522` (Langer, info-theoretic multimodal VAE) is the
   *general* version of MJEPA's "no cross-modal term → degrades below unimodal" finding; it tells us
   *when* joint coding helps. Architecture already supports the term (see §3).

---

## 5. Layer → reference map

| Layer | Reference | Link |
|---|---|---|
| L1 tokenizers | DCT/freq-domain action+proprio FAST `2501.09747`, VQ-VLA `2507.01016`; IMU Mojito `2502.16175`; tactile Sparsh `2410.24090` | https://arxiv.org/abs/2501.09747 |
| L1 vision branch | RVM recurrent video MAE `2512.13684`; Adaptive 1D Video Diffusion AE `2602.04220`; Cosmos/MagViT-2 `2310.05737` | https://arxiv.org/abs/2512.13684 |
| L2 conditioning | FiLM `1709.07871`; AdaLN (DiT-style) | https://arxiv.org/abs/1709.07871 |
| L3 fusion/compress | Perceiver IO `2107.14795` (asymmetric cross-attention) | https://arxiv.org/abs/2107.14795 |
| L4 streaming | Mimi/Moshi `2410.00037`; RVQ `2203.01941` | — |
| architecture lineage | HPT stem/trunk `2409.20537` — no decoder / vision+proprio / not causal; our codec diverges | https://arxiv.org/abs/2409.20537 |
| prior art: joint compression | Lu et al. CVPR22 multi-modality image/video compression; Neural Codecs as Biosignal Tokenizers `2510.09095` | https://arxiv.org/abs/2510.09095 |
| objective (v0.5) | Langer `2411.00522`; Zambelli `1910.03854`; MJEPA `2606.25225`; APD `2009.01791` | — |
