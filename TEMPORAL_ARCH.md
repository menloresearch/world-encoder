# Kepler Temporal (Phase 2) — Architecture Design

Design doc for the temporal extension of Kepler-Encoder-v0.1. Written to (a) pin the
architecture precisely so it can be implemented without re-litigating, and (b) align a
second reader on *why* it's shaped this way. Section numbers in **[PLAN 2.x]** refer to
`PLAN.md` § "Phase 2 — Temporal".

**Reading guide (this doc grew — which section is what):**
- **§5** = the canonical *as-designed* flat architecture (v1). Start here for the arch.
- **§16** = the *as-built* plan of record (what actually got implemented/trained).
- **§0b** = the same v1 in plain English (skim if §5 is enough); **§15** = a *future* generic-v2 redesign (roadmap, not built).
- **§18.1–18.6** = the eval protocol; **§18.7** = condensed results. Full debugging journal → `TEMPORAL_JOURNAL.md`; numbers + saved JSONs → `results/temporal/RESULTS.md`.
- **§20/§20.1** = external work (LeWM, RoboTTT) motivating the next redesign.

---

## 0. TL;DR — the one decision

**Question raised:** do we fuse *space first then time*, or *time first then space*? Do we
need a spatial Perceiver feeding a temporal Perceiver?

**Answer: neither. One flat multi-rate Perceiver.** We tokenize each sensor stream at its
own native rate, tag *every* token with `(modality, position, time)` embeddings, and pour
all of them — all modalities, all timesteps — into a **single** Perceiver that fuses across
modality and time jointly. There is no spatial-vs-temporal ordering because there is no
factorization to order.

Three independent reasons this is right:
1. It's exactly how the **original Perceiver / Perceiver IO** handle video and video+audio
   (flatten space+time, inject per-axis position features, one latent array).
2. It's what **PLAN §arch (a)** already commits to ("dump a whole time-window of
   timestamp-embedded tokens into ONE Perceiver").
3. It's the only design that honestly expresses our **multi-rate, no-resample** thesis (see
   §5.3).

The two-level "stack a Perceiver on top" idea and the "freeze v0.1, add a temporal head"
idea are **not discarded** — they become **ablation baselines**, and crucially they run from
the *same* configurable block (§6), so "flat vs stacked" is a config flag, not a rewrite.

---

## 0b. v1 in full (the Phase-2 build target, in plain English)

The plan-of-record model (§16), start to finish. v1 = **v0.1 with a time axis added the cleanest
way**: same encoder, same loss, same eval philosophy, but it ingests a window of `C` ticks, tags
every token with *when*, and fuses them in one flat Perceiver.

**Pipeline** (shapes: `d=256`, `N`≈64–256 latents, `M` = total context tokens):
1. **Inputs** — a window of `C` contiguous ticks (one segment, §7.1): vision `[B,C,196,768]`,
   motor `[B,C,8,3]`+mask, ee `[B,C,13,15]`+mask, each with timestamps.
2. **Tokenize** — one `Conv1d(C_in→d, k, s)` per stream: vision `k=1` (≡ linear `proj_v`), motor
   `k=1` (≡ `proj_m`), ee `k=16,s=4` over time → three lanes of width-`d` tokens.
3. **Inject structure** — each token `+= modality + position + continuous-time` embedding (mTAN,
   log-spaced ms→s). This is where time enters the token, not a grid.
4. **Flatten + concat** → one flat context `[B, M, d]`, `M = C·196 + C·8 + n_ee`. Structure
   dissolves into a self-describing bag of tokens.
5. **Flat Perceiver** (`PerceiverBlock`) — `N` learned latents cross-attend all `M` (`O(N·M)`,
   linear), then latent self-attention (`O(N²)`) → belief `[B, N, d]` (not pooled).
6. **Query-based predictor** — output queries carrying `(modality, time)` read the belief →
   predicted embeddings `[B, O, d]`. (The "IO predictor", §5.5 — outputs latents, discarded after
   training, NOT PixNerd.)
7. **Loss** (= Phase 1 + time adaptations, §7.4) — `inv` = MSE(preds, EMA-target of held-out
   `(modality,time)`) + `λ`·per-timestep SIGReg (per-modal + joint). `λ≈0.02`, EMA≈0.99.

**Masking (training):** hide a subset of `(modality × time)` tokens — a modality (incl. vision,
whole-frame) and/or the future (`t'>t`) — predict their representations from the rest (§7.1).
**Eval:** hide motor + ee, keep vision → `z_v`; what's readable from `z_v` is state vision learned to
carry (the thesis, over a window).

**Warm-started from v0.1.** Deliberately **not** in v1 (deferred, §15/§16): typed/any-variate tokens
and dual-path fusion. v1 is the minimal, attributable temporal step; the gate (§10) decides what's next.

---

## 1. The problem shape (why temporal is hard here)

Our data is **multi-modal, multi-rate, and irregular in time**:

| stream | rate | per-tick shape (today, v0.1) | notes |
|---|---|---|---|
| vision (1 external cam) | ~10 Hz (one frame/tick) | `196 × 768` | frozen ViT-B/16 patch tokens |
| motor (joints + gripper) | ~10 Hz (one grid/tick) | `8 × 3` | 7 joint rows + 1 gripper row |
| ee (F/T + TCP pose) | **~100–125 Hz** (dense) | `13 × 15` | native high-freq samples inside the tick |

- **Ticks are irregular: ~6.7–14.7 Hz**, and scenes contain multi-minute recording gaps
  (`GAP_MS = 500` ms splits ticks into contiguous segments).
- **Nothing shares a clock.** ee runs ~10× faster than vision; tick spacing wobbles.
- **The bet ("white space"):** do *not* resample everything onto a common grid. Instead put
  the real timestamp *inside* each token and let attention handle the irregularity. Phase 2
  exists to prove this beats the classical "resample-to-a-grid" approach (see §10).

---

## 2. What the two Perceiver papers give us

### 2.1 Perceiver — arXiv:2103.03206
- A small **latent array of `N` tokens** cross-attends to a large input of `M` tokens →
  then runs `L` **latent self-attention** blocks (a small transformer *on the latents*).
- **`N` (the latent "length") is decoupled from `M` (the input size).**
- **Cross-attention is `O(N·M)` — linear in input size.** This is the whole point: you can
  grow the input (our window length `C`) without the `O(M²)` blowup a plain transformer has.
- Latent self-attention is `O(N²)`, and `N` is tiny → depth is cheap.
- Repeated cross+self blocks can **share weights** (like an unrolled RNN) → more depth, same
  parameter count.

### 2.2 Perceiver IO — arXiv:2107.14795
- Adds an **output query array of `O` tokens** that cross-attends the final latents to emit
  outputs of *any* shape/length.
- So **input size `M`, latent size `N`, and output size `O` are three independent knobs.**
- Used for language, optical flow, StarCraft, and video+audio+label autoencoding.

### 2.3 The catch both papers hammer (this is the crux for us)
Attention is **permutation-invariant** — it has no built-in notion of space or time. So
**all structure must be injected as embeddings on the tokens**: which modality, which
patch/row, and — the new axis for us — **when**. This is *exactly* why our continuous-time
embedding **[PLAN 2.1]** goes *inside the token*, not into a grid.

### 2.4 How they actually handled video (the friend's real question)
For video and video+audio, Perceiver / Perceiver IO **flatten the entire spatio-temporal
array into one input set**, attach **Fourier position features with a separate frequency
band per axis** (x, y, t), and let **one latent array** attend to all of it. **No spatial
network, no temporal network, no ordering.**

Why does the rest of the field factorize (ViViT, TimeSformer: "spatial attention then
temporal attention")? Only to dodge the **`O((C·P)²)`** cost of full joint space-time
self-attention in a plain ViT (`P` = patches/frame, `C` = frames). **That cost is a
transformer problem, and the Perceiver bottleneck already removes it** (cross-attn is linear
in `M`). So the reason to factorize does not apply to us. Same-family data point: **V-JEPA
does joint spatio-temporal attention** over tubelet tokens — it just caps token count via 3D
patching + masking; we cap it via the latent bottleneck. The field-wide signal is *joint,
not factorized* — factorization is a compute workaround we don't need.

---

## 3. The data & tokenization (exact numbers, from `world_tokenizer/chunk_state.py`)

Constants: `N_PATCH=196`, `N_MOTOR=8`, `MOTOR_CH=3`, `EE_T=13`, `EE_DIM=15`, latent width
`d=256`, `vis_dim=768`.

### 3.1 Vision — sparse, ~10 Hz, native tokens (unchanged from v0.1)
- Frozen ViT-B/16 → **196 patch tokens × 768** per frame, **one frame per tick**.
- Over a window of `C` ticks → **`C × 196` vision tokens**.
- Spatial "where" is **already baked into the ViT's 768-dim features** (its own position
  embeddings). We only *add* the modality embedding + the new **time** embedding.
- Tokenizer: `Conv1d(768 → d, kernel=1)` — a `k=1` conv *is* a per-token linear layer (this is
  v0.1's `proj_v`; unified with the ee CNN in §3.5 / §5.1).

### 3.2 Motor — sparse, ~10 Hz, native tokens (unchanged from v0.1)
- Shape `(8, 3)` per tick. **8 rows** = 7 joint rows (row 6 is masked for 6-DOF robots) +
  1 gripper-width row. **3 channels** = `[sin q, cos q, symlog dq]` (gripper row uses ch0 =
  symlog width; KUKA joint torque is ignored).
- A per-value validity mask `motor_mask (8, 3)` (True = valid) rides along; the loader
  packs values+mask into **6 channels** (`motor * mask` ⊕ `mask` = `[B, 8, 6]`).
- Tokenizer: `Conv1d(6 → d, kernel=1)` (≡ v0.1's linear `proj_m`). Over `C` ticks → **`C × 8` motor tokens**.
- A kernel-16 CNN over time is **nonsensical at ~10 Hz** (16 samples = ~1.6 s), so motor
  stays as native tokens **[PLAN 2.2]**.

### 3.3 ee (force/torque + TCP pose) — DENSE, ~100–125 Hz — the stream that changes
- Per tick, `(13, 15)`: up to **`EE_T = 13`** high-freq samples that fall in
  `[tick_k, tick_{k+1})`, each **`EE_DIM = 15`**-dim:
  - `[0:6]` = symlog zeroed **force/torque** (6)
  - `[6:9]` = symlog **TCP xyz** (3)
  - `[9:15]` = **TCP orientation as 6D rotation** (6)  *(quaternion → 6D via `quat_to_6d`)*
- Validity: `ee_mask (13,)` — **one bool per sample** (True = valid). It is **all-False for
  all of cfg5 and ~⅓ of cfg3 scenes** (no high-freq data). Fully-masked samples are
  *excluded* from the ee loss, never averaged over zero valid elements (NaN trap).
- **The 13 slots are already the native ~100 Hz samples inside one tick interval.** So over
  `C` ticks they concatenate into one near-continuous ~100 Hz series of length **`C × 13`**.
  (Sanity check: a tick interval is ~68–150 ms at 6.7–14.7 Hz; at 100–125 Hz that's ~7–19
  samples, capped/padded to 13. Consistent.)

**Shape reframing (this is the correct mental model, confirmed against the loader):**

```
                 v0.1 (single tick)      Phase-2 window (C ticks)
  vision         [B,        196, 768]     [B,  C,     196, 768]   →  C·196 tokens
  motor          [B,          8,   3]     [B,  C,       8,   3]   →  C·8   tokens
  ee (dense)     [B,         13,  15]     [B,  C·13,       15]    →  1-D CNN over time
```

Note the ee row flattens `(C, 13)` → **`C·13`**: that single long time axis is exactly what
the 1-D CNN convolves over. Vision/motor keep a plain `C` because they're one sample/tick;
ee gets `C·13` because it's dense. Using `C·13` for ee (and plain `C` for the others) is the
right way to express the rate difference.

### 3.4 The 1-D CNN tokenizer for ee **[PLAN 2.2]**
- A small 1-D convolution slides a **kernel of width ~16** along the `C·13` time axis
  (start **kernel-16 → 128** feature maps) → compresses the flood of raw samples into a
  handful of **ee tokens**, each summarizing the local *shape* of the force over a short
  span (a spike, a ramp, a contact onset).
- Analogy: the ViT patchifier is a **spatial** tokenizer for vision; the 1-D CNN is a
  **temporal** tokenizer for the dense ee stream. Both are per-stream front-ends; **neither
  does fusion.** (This is why running a CNN "over time" for ee is *not* the same as doing
  "temporal-first" fusion — see §5.4.)
- Validity mask rides in as **extra channel(s)**: input = 15 values + per-sample validity.
  Justified because ee is **locally near-regular within a chunk** (conv assumes roughly even
  spacing, true locally even though the window as a whole is irregular).
- Each output ee token is tagged with the **timestamp of its receptive-field center** +
  modality embedding, then enters the flat context (§5).

> ⚠️ **Doc bug to fix in PLAN 2.2 before implementing.** PLAN 2.2 currently says the dense
> CNN input is *"flatten the 8×3 grid … 24 vals + 24 mask = 48 in-ch … variable-DOF
> masking."* Those numbers are the **motor** packet (`8×3 = 24`, per-value DOF masking), not
> ee. The stream that gets the CNN is **ee, which is 15-dim with a per-*sample* validity
> mask** → input is **~16 channels** (15 values + 1 validity), *not* 48, and there is no 8×3
> grid or variable-DOF here. Build the CNN on `(B, ~16, C·13)`. Also: motor does **not** get
> a CNN at all (it's the sparse stream). Pin this before wiring the tokenizer.

### 3.5 One tokenizer, one knob (the unified view — folded in from v2 §15)
All three tokenizers above are the **same class**: `Conv1d(C_in → d, kernel=k, stride=s)` over the
stream's sequence axis. The only per-stream difference is `(k, s)`:
- **vision** `Conv1d(768→d, k=1, s=1)` over patches — `k=1` *is* a per-token linear (≡ `proj_v`).
- **motor** `Conv1d(6→d, k=1, s=1)` over rows — `k=1` (≡ `proj_m`).
- **ee** `Conv1d(16→d, k=16, s=4)` over time — `k>1` = temporal patching.

This is the *one* v2 idea (§15) we adopt for the Phase-2 build: it collapses the "linear-vs-CNN"
special-casing into a single code path where `k=1` reproduces v0.1 exactly and `k>1` is the ee
patcher. It is a refactor for configurability (kernel becomes a knob), **not** a modeling change —
`k=1` and a linear layer are identical. The other two v2 ideas (typed tokens, dual-path fusion)
are **not** in the build target — see §16.

---

## 4. The continuous-time embedding — the "when" slot **[PLAN 2.1]**

- **Target = mTAN, but only mTAN's *continuous-time embedding*, not the whole mTAN
  network.** Why: the Perceiver already *is* mTAN's other half — mTAN embeds time then
  attends reference points to observations; our **learned queries = reference points** and
  **cross-attention = attention-to-observations**. Running full mTAN would duplicate the
  Perceiver.
- **Sequence:** start with **fixed Fourier / Time2Vec, log-spaced frequencies**; upgrade to
  **learned** frequencies (= mTAN proper) only if the numbers demand it — same interface, no
  rework.
- **Make-or-break knob:** the frequency bank must **span ~ms → seconds** (ee at ~100 Hz vs
  multi-second episodes). This matters *more* than Fourier-vs-mTAN.
- Follow the papers: use a **separate frequency band per axis** — the time band is its own,
  distinct from any spatial position encoding. Minor knob: **concatenate vs add** the time
  embedding (paper found concat slightly better; v0.1 adds — cheap to try both).
- **Continuity note:** v0.1 already reserved this slot. In `chunk_state.py`/`mm_perceiver3`
  the ee `pos_e (13, d)` is commented *"ee slot (time placeholder)"*, and every token has
  carried its `ts` since Phase 1 **[PLAN 1.2]**. Phase 2 just **fills the placeholder with a
  real continuous-time embedding** — no data or interface rework.
- RoPE / VideoRoPE only later, for latent self-attention, and the **continuous (real-Δt)**
  variant — not integer-index.

---

## 5. The final architecture (flat)

### 5.1 Pipeline (shapes tracked end-to-end)

Symbols: `B`=batch · `C`=window length in ticks · `d`=256 (latent width) · `N`=`n_latents`
(latent array length) · `O`=output-query count · `M`=total context tokens · `n_ee`=ee tokens
after the CNN.

```
════════════ 1. RAW WINDOWED INPUTS (per sample = C contiguous ticks, one segment) ════════════

 VISION                         MOTOR                          EE (dense F/T + TCP)
 frozen ViT-B/16 (precomputed)  joint/gripper grid             ~100–125 Hz samples
 [B, C, 196, 768]               [B, C, 8, 3]                   [B, C, 13, 15]
 (always present)               + motor_mask [B, C, 8, 3]      + ee_mask [B, C, 13] (per-sample)
 per-tick time  t_1..t_C        per-tick time t_1..t_C         per-sample times (denser than ticks)

════════════ 2. PER-STREAM TOKENIZERS  (ONE Conv1d(C_in→d, k, s) class;  k=1 ≡ linear; no resample) ════════════

 VISION lane                    MOTOR lane                     EE lane
 Conv1d(768→d, k=1, s=1)        pack vals⊕mask → [B,C,8,6]     flatten time: [B, C·13, 15]
 over the 196-patch axis        Conv1d(6→d, k=1, s=1)          +validity → [B, 16, C·13]
 → [B, C, 196, d]               over the 8-row axis            Conv1d(16→d, k=16, s=4) over time
   (k=1  ≡ v0.1 proj_v)         → [B, C, 8, d]                 → [B, n_ee, d]  n_ee ≈ C·13/s
                                  (k=1  ≡ v0.1 proj_m)           (k>1 = temporal patching)

════════════ 3. INJECT STRUCTURE  (+= modality, position, TIME embeddings, each [d]) ════════════

 + mod_emb[vision]              + mod_emb[motor]               + mod_emb[ee]
 (patch pos already in ViT)     + pos_m[8]  (which row)        (time replaces old pos_e slot)
 + time_emb(t_c) per tick       + time_emb(t_c) per tick       + time_emb(t_j) per ee token
 → [B, C, 196, d]               → [B, C, 8, d]                 → [B, n_ee, d]

════════════ 4. FLATTEN each lane to a token list, then CONCAT ════════════

 [B, C·196, d]         ⊕        [B, C·8, d]          ⊕         [B, n_ee, d]
                                     │
                                     ▼
                         CONTEXT   [B, M, d]        M = C·196 + C·8 + n_ee
                         (one flat set: all modalities, all times)

════════════ 5. FLAT PERCEIVER  (PerceiverBlock) ════════════

     latents  self.q [N, d] ──expand──► [B, N, d]
        ┌────────────────────────────┴───────────────────────────┐
        │  n_cross × CROSS-ATTENTION                                │
        │    Q=[B,N,d]  attends  KV=context[B,M,d]                  │  cost O(N·M) ← linear in M
        │    attn_mask [B,N,M] (blocks masked (modality×time) +     │
        │                       invalid tokens; causal optional)    │
        │  → [B, N, d]                                              │
        │  n_self × LATENT SELF-ATTENTION  ([B,N,d]→[B,N,d])        │  cost O(N²)  ← N tiny
        └────────────────────────────┬───────────────────────────┘
                                     ▼
              LATENT ARRAY   [B, N, d]    (spatio-temporal belief; NOT pooled)

════════════ 6. PERCEIVER-IO PREDICTOR  (read out held-out / future target latents — NOT PixNerd; §5.5) ════════════

     output queries [B, O, d]  each carries a target (modality, TIME) coord
                                     │
              CROSS-ATTN:  Q=queries[B,O,d]  attends  KV=latents[B,N,d]
                                     ▼
              PREDICTIONS   [B, O, d]

════════════ 7. LOSS ════════════

     targets = EMA-target embeddings of held-out (modality,time) tokens  [B, O, d]
     inv   = MSE( predictions[B,O,d] , targets[B,O,d] )        ← predict-don't-equate
     sig   = per-timestep SIGReg on marginal embeddings        ← anti-collapse, NOT time-pooled
     loss  = inv + λ·sig     (λ≈0.02, EMA≈0.99)

════════════ EVAL ════════════
     z_v : run steps 2–5 with motor+ee masked → vision-only latent array (probe / downstream)
```

**Concrete instantiation (C=16, d=256, N=128, CNN stride ≈ 4):**

| stage | vision | motor | ee | context `M` |
|---|---|---|---|---|
| raw | `[B,16,196,768]` | `[B,16,8,3]` | `[B,16,13,15]` | — |
| tokenized + `d` | `[B,16,196,256]` | `[B,16,8,256]` | CNN → `[B,~52,256]` | — |
| flattened | `[B, 3136, 256]` | `[B, 128, 256]` | `[B, ~52, 256]` | **`[B, 3316, 256]`** |
| after Perceiver | — | — | — | latents `[B, 128, 256]` |
| after decoder | — | — | — | preds `[B, O, 256]` |

The only place `C` and the per-stream rates "disappear" is the flatten in step 4 — after that
the model sees a flat token set whose *only* record of time is the embedding added in step 3.
That is the no-resample bet made concrete.

### 5.2 The three stages, precisely
1. **Per-stream tokenizers — one `Conv1d(C_in→d, k, s)` class, per-stream `(k, s)`.** `k=1` is
   exactly a per-token linear layer, so vision (`Conv1d(768→d, k=1)`) and motor (`Conv1d(6→d, k=1)`
   over 6 = 3 values + 3 mask) reduce to v0.1's `proj_v`/`proj_m`; the dense ee stream uses `k>1`
   (`Conv1d(16→d, k=16, s=4)`) to patch over time → `~O(C)` tokens. One code path; kernel is the
   only per-stream difference (see §3.5). Every token gets modality emb + within-modality position
   + **continuous-time embedding**.
2. **One flat Perceiver.** All tokens (all modalities, all times) form one context of size
   `M`. `N` learned latents cross-attend → `L` latent self-attention blocks → **latent array
   `[B, N, d]`** (not pooled — the decoder needs the array). Bidirectional by default; causal
   via a mask on the time tag when needed (§7.2).
3. **Perceiver-IO predictor** (the JEPA predictor head — see §5.5, *not* a pixel decoder). Output
   queries carrying a target **`(modality, time)`** coordinate cross-attend the latent array →
   predict that token's embedding, matched (MSE) against the **EMA-target** embedding of the
   held-out token. Output length `O` is independent of `C` and `N`, so "predict ee at `t+Δ` for 5
   values of Δ" = 5 output queries.

### 5.3 Why flat wins *for us* (not just "the paper did it")
- **It's the only design that honestly expresses the multi-rate no-resample bet.** Time is a
  continuous per-token tag, so a 100 Hz ee token and a 10 Hz vision token simply have
  different `t` and coexist — no common grid. A spatial-then-temporal hierarchy would pool
  per tick → impose a tick grid → **resampling by the back door**, fighting the exact thing
  we're testing.
- **It preserves the fine interaction we're chasing.** A vision patch at `t=2` can attend
  *directly* to an ee sample at `t=7`. That subtle, time-integrated visual→force signal is
  the scientific point. Pooling each tick to one `z_t` first bottlenecks precisely that.
- **Compute doesn't force a hierarchy** (see §8).

### 5.4 Heading off the obvious objection
"But you run a CNN over time for ee — isn't that temporal-first?" No. The CNN is
**tokenization**, exactly the role the ViT patchifier plays for vision. It produces ee tokens;
it does **not** fuse ee with other modalities or do global temporal reasoning. All fusion —
across modality *and* time — happens in the one flat Perceiver. Tokenizers are per-stream;
fusion is joint. No inconsistency.

### 5.5 The "decoder" is the JEPA predictor — NOT PixNerd
Stage 6 is called a "decoder" only because Perceiver-IO names its readout head that way. In our
setting its role is the **JEPA predictor**:
- It outputs **latent embeddings** (`[B, O, d]`), never pixels.
- It exists solely to compute the masked-prediction loss (§7.4) and is **discarded after training** —
  the product is the encoder.
- It is **not PixNerd.** PixNerd (Phase 3) is a *separate* pixel decoder — a downstream probe / viz
  tool trained afterward on the *frozen* encoder's latents, not part of this training loop.

It's a query-based head (not v0.1's MLP predictor) only because temporal prediction must specify
*which time* to read out, and a query can carry a `(modality, time)` coordinate while an MLP head
cannot. Read "decoder" as "predictor" everywhere in §5–§7.

---

## 6. The code primitive: a configurable `PerceiverBlock`

### 6.1 Where the code is today
`PerceiverFuse` (`world_tokenizer/mm_perceiver.py:23`) is a *partial* Perceiver:
- ✅ learned query array + cross-attention (`depth` cross-attn + FFN layers)
- ❌ **no latent self-attention** — missing the paper's "latent transformer" (cheap depth)
- ❌ **mean-pools the queries** at the end (`x.mean(1)` → `[B, d]`) — can't stack/decode on a
  pooled vector
- ❌ **always uses its own `self.q`** — can't be seeded with an external latent array

Three surgical changes fix all of it.

### 6.2 The block
```python
PerceiverBlock(
    dim        = 256,      # D  — latent width (v0.1 used 256)
    n_latents  = 8,        # N  — latent array "length"   ← bump to 64–256 for a window (v0.1=8 was for 217 tokens)
    n_cross    = 1,        # cross-attend layers (>1 = re-read the input)
    n_self     = 6,        # latent self-attention blocks  ← cheap depth, O(N²)  (v0.1 had 0)
    n_heads    = 8,
    share_weights = True,  # tie repeated cross+self (Perceiver's RNN trick; more depth, same params)
    pool       = None,     # None → [B, N, D]  (needed to stack/decode);  "mean" → [B, D]  (probe/eval only)
    ff_mult    = 4,        # FFN hidden = 4·D
)
# forward(context, attn_mask=None, latents=None)
#   latents=None      → use self.q       (fresh window)
#   latents=[B,N,D]   → seed from outside (stacking / recurrent belief state)
```

### 6.3 Flat vs stacked = same block, different config
Because the block takes an optional external latent array and can skip pooling:
- **Flat (default):** one `PerceiverBlock` over the full time-tagged token set.
- **Two-level stack (ablation):** a per-tick block (`pool=None`) whose `[B,N,d]` outputs feed
  a second block over time.
- **Recurrent (future):** one block with `latents=` set to the previous window's output.

So "which architecture" is a **config flag, not a rewrite** — and the same code runs the flat
model *and* the ablations that the flat conclusion calls for (§10).

---

## 7. Masking, causality, regularization, loss

### 7.1 Masked prediction over (modality × time) **[PLAN 2.3]**
Generalizes v0.1's "hide one modality" to "hide a subset of `(modality, time)` tokens." In the
flat set that's just an attention mask blocking those context columns. Targets are the
**EMA-target** embeddings of the held-out tokens, read out by output queries at those
`(modality, time)` coordinates. Special case = **future prediction**: block all tokens with
`t' > t`, predict the future ee/contact latent.

**Vision IS masked during training — at whole-frame granularity.** The hide-one scheme (from v0.1's
`hide=("v",)` pass) applies to vision too: vision is *not* an always-on input — it's masked in some
passes and predicted from the other modalities/times. Masking granularity is **`(modality × time)`,
i.e. per-frame**: hiding vision blocks *all 196 patches of a frame together* (all cameras together),
and the vision target is the mean-pooled patch embedding, not per-patch. We do **not** do
MAE/V-JEPA-style *within-frame patch (tubelet) masking*. That finer video masking is a **deferred
lever** (trigger: flat plateaus and we want a stronger vision pathway) — not in the v1 build target.
At **eval** the masking is reversed: hide motor + ee, keep vision only → `z_v`.

**Windowing constraint (from the tick segmentation, §1).** A `C`-tick window must be drawn from a
**single contiguous segment** — it must never straddle a `> GAP_MS` (500 ms) recording gap.
`chunk_state.py` already splits ticks into gap-free segments and drops each segment's last tick;
Phase-2 window selection must stay inside one segment, or the continuous-time embedding would encode
a Δt of minutes between adjacent tokens and poison the temporal attention.

### 7.2 Causal vs bidirectional **[PLAN 2.3]**
Causality is an **attention-mask property on the time tag**, not baked into the conv:
- **Prediction head / deployment:** causal (a query at `t` sees only `t' ≤ t` — no peeking at
  what it predicts).
- **Within-window representation objective:** test **both**; V-JEPA-style bidirectional often
  wins on probe quality. This is an experiment, not a fixed choice.

### 7.3 SIGReg under time **[PLAN 2.3b]**
SIGReg only regularizes the **marginal** embedding distribution (anti-collapse); it does *not*
learn dynamics — the prediction objective does. So its Epps–Pulley statistic must **not pool
over time**: apply **per-timestep** (each instant's marginal), never on a time-pooled latent.
Placement/strength (per-modal vs joint; on/off on the temporal latent) is an ablation.
Pre-empts the "LeJEPA is bad at time variance due to global pooling" critique.

#### Open debate — should SIGReg touch informative (non-first) frames at all? (Dhanoosh, 2026-07-16)
**Argument (Dhanoosh):** SIGReg is a **shape** regularizer — it forces the marginal toward an
isotropic Gaussian. Gaussian is the ideal shape *only* for an **information-free** latent (a
max-entropy prior). Temporal frames after the first carry information from context, so their latent
**shouldn't** be Gaussian — and because it's a *shape* constraint, **lowering λ ("softening") does
not fix it**; even weakly it pulls toward the wrong shape. Proposed fix: apply the Gaussian
constraint only where there is no prior info — the **first / context-free latent**, or the
**new-information / residual** component.

**Counter-nuance:** SIGReg constrains the **marginal** (over sequences at a fixed t), not the
**conditional** `p(z_t | history)`. A Gaussian marginal is compatible with sharp, informative
conditionals (aggregate-posterior / mixture argument), so per-timestep SIGReg is **not proven** to
fail — and empirically the current per-timestep recipe **trains healthily and gives a real
cross-modal signal**. LeJEPA's isotropic-Gaussian optimality is also derived for the i.i.d. setting,
so whether it transfers to correlated sequences is genuinely open. ⇒ This is an **optimality**
question, not "broken" — **empirical and cheap to test.**

**Resolution (2026-07-16):** Dhanoosh **conceded** the counter — *"you can have the shape of a Gaussian
but have the latent be informative."* So the objection drops from "per-timestep SIGReg destroys info"
(a **shape** impossibility) to "it may not be the **optimal** placement" (empirical). Not a blocker.
Corroborated by our diagnostics: SIGReg was never the disease (§18.11 — RankMe ~8–10 meant it was being
*overpowered*, not over-regularizing; the invariance objective failing to learn was the disease). And
the §18.12 fix *adds* SIGReg (**joint** SIGReg on the fused latent) — that's what stabilizes the raw
target. So SIGReg is load-bearing in the **helpful** direction here: the fused latent *needed* it (v0.1
had it; v0.2 had dropped it). His instinct that SIGReg is structurally important was right; the sign was
flipped — apply it to the fused latent, don't remove it. His residual-innovation form stays queued as
(iii) below, to run once the base model is fixed.

**Decision:** do **not** block the NH1 gate on this. Run the gate with the current per-timestep
recipe (it trains + gives signal); then A/B SIGReg placement. Dhanoosh owns the exact form + math;
we implement it as a config flag and run.

**SIGReg-placement ablation (config flag):**
- **(i) per-timestep marginal** — current.
- **(ii) first / context-free latent only** — no SIGReg on later frames (Dhanoosh's simplest form).
- **(iii) residual / innovation** — SIGReg on `r_t = z_t − f(z_{<t})` (the "no prior info" part);
  the accumulated state `z_t` left free (predictive-coding flavor; likely his intended form).

Judge on the temporal probes (§19.1 P3/P4). If (ii)/(iii) beats (i), adopt it.

### 7.4 Loss — same as Phase 1, with two time adaptations
The Phase-1 loss (in `mm_perceiver3.py`) is `loss = inv + λ·sig` (`λ ≈ 0.02`, `EMA ≈ 0.99`):
- **`inv`** — masked cross-modal latent prediction (MSE, predict-don't-equate): hide one modality,
  predict its EMA-target embedding (`tgt_v/tgt_m/tgt_e`, mean-pooled) from the fused latent of the
  others, via per-modality MLP predictor heads (`self.pred["v"/"m"/"e"]`).
- **`sig`** — SIGReg (`SlicedEppsPulley`): **per-modal** (`ev`, `em`, ee when ≥8 valid) **+ joint**
  (on `z_full`). Anti-collapse only; it does not learn dynamics.

v1 keeps this **exactly**, changing only two things for time (plus one mechanism change):
1. **Masking spans `(modality × time)`, not just modality** — the held-out set now includes future
   tokens (`t' > t`). Same MSE, larger held-out set. (§7.1)
2. **SIGReg is applied per-timestep, never on a time-pooled latent** (§7.3). Per-timestep is
   *orthogonal* to per-modal: keep the per-modal + joint SIGReg from Phase 1, just compute it on
   each instant's marginal instead of one pooled vector.
3. *(mechanism, not a loss change)* the predictor goes from per-modality **MLP heads** → a
   **query-based (time-aware) predictor** (§5.5), because prediction now has to specify *which time*.

So: **same loss family and components as Phase 1** — masked cross-modal latent prediction +
(per-modal + joint) SIGReg — generalized over `(modality × time)`.

---

## 8. Token budget & compute (why flat is affordable)

`M = C·196 (vision) + C·8 (motor) + ~O(C) (ee)`.

| window `C` | vision | motor | ee (post-CNN) | total `M` |
|---|---|---|---|---|
| 16 | 3,136 | 128 | ~tens | **~3.3k** |
| 64 | 12,544 | 512 | ~hundred | **~13k** |

The original Perceiver ran on **50k+** input elements (ImageNet = 50,176 pixels). Our
`~3–13k` is comfortable. Cross-attn cost `O(N·M)` (e.g. `N=256, M=3.3k` → ~0.85M entries/head
— trivial); latent self-attn `O(N²)` regardless of `M`. **Compute does not force a
hierarchy** at our window sizes — the two-level stack's only advantage is at very long `C`.

---

## 9. Variants (kept as ablations, not the default)

- **Freeze-v0.1 stack (ablation baseline).** Freeze the validated Phase-1 per-tick encoder,
  train only a temporal `PerceiverBlock` over the sequence of `z_t`. Cheaper, builds on
  validated Phase-1, but bottlenecks cross-time detail. Running it lets the flat model *prove*
  it beats the pooled hierarchy — evidence for the joint design.
- **Recurrent / carry-forward (future; the world-model, loss #4) [PLAN arch (b)].** Carry a
  latent belief state forward, updating it with each new window (`latents=` ← previous output).
  Naturally causal, streaming, arbitrary-length; it's the structure you roll under an action
  (`z_t, a_t → z_{t+1}`). **Prove the window model first.**

---

## 10. Eval & the decisive ablation

- **Eval [PLAN 2.4]:** future force/contact prediction at varying `Δt`, vs the single-timestep
  Phase-1 baseline.
- **Decisive ablation [PLAN 2.5]** — three legs, all from the same configurable block:
  1. **Flat continuous-time** (this doc's default) — time in the token, no resample.
  2. **Resample-everything-to-a-grid + 1-D CNN** — the classical DSP baseline the white-space
     bet must beat.
  3. **Freeze-v0.1 stack** (§9) — the hierarchy baseline.
  If flat doesn't beat (2) and (3), we adopt the simpler thing honestly.
- **Gate [PLAN]:** temporal masking beats single-timestep on future-state prediction with
  RankMe stable. Loss #4 only after this gate.

---

## 11. The knobs (all config-driven)

| knob | symbol | default | what it controls |
|---|---|---|---|
| latent length | `N` (`n_latents`) | 64–256 (v0.1: 8) | representational capacity of the window belief |
| window length | `C` | 16 → sweep | how much time is in context (cheap: cross-attn linear in `M`) |
| latent self-attn depth | `n_self` | 6 | processing depth (cheap, `O(N²)`) |
| cross-attend layers | `n_cross` | 1 | how many times the latents re-read the input |
| weight sharing | `share_weights` | True | depth without more params |
| width | `dim` (`d`) | 256 | latent width |
| heads | `n_heads` | 8 | attention heads |
| output length | `O` | task-dependent | how many `(modality,time)` targets to decode (IO decoder) |
| time freq band | — | log-spaced, ms→s | the make-or-break time-embedding range |

---

## 12. Build order

1. **`PerceiverFuse → PerceiverBlock` refactor** — add latent self-attention, optional pool,
   optional external `latents=`. (Fixes PLAN 2.2's CNN-input note first, §3.4.)
2. **`mm_perceiver_temporal.py`** — per-stream tokenizers as one unified `Conv1d(C_in→d, k, s)`
   class (reuse frozen ViT; `k=1` for vision/motor ≡ `proj_*`, `k>1` for ee; §3.5) + continuous-
   time embedding + one flat `PerceiverBlock` + Perceiver-IO predictor. **Warm-start from v0.1
   weights** (same block, more tokens).
3. **Masking + loss** over `(modality × time)`; per-timestep SIGReg.
4. **Ablation harness** — flat vs resample+CNN vs freeze-v0.1, all by config.

---

## 13. Open questions (decide by experiment, not upfront)

1. **Fourier vs learned time frequencies** — start fixed, escalate only if numbers demand.
2. **Causal vs bidirectional** for the within-window objective.
3. **SIGReg placement** — per-modal vs joint; on/off on the temporal latent.
4. **Latent length `N` and window `C`** — sweep.
5. **Time embedding concat vs add**, and the exact **frequency band**.

---

## 14. Mapping to PLAN.md

| PLAN 2.x | this doc |
|---|---|
| 2.1 continuous-time embedding | §4 |
| 2.2 per-stream tokenizers (unified conv) | §3.3–3.5 (+ the doc-bug fix) |
| 2.3 multi-tick windows, masking, causality | §7.1–7.2 |
| 2.3b SIGReg under time | §7.3 |
| 2.4 eval (future prediction) | §10 |
| 2.5 decisive ablation | §10 |
| arch (a) window Perceiver | §5 (the flat default) |
| arch (b) recurrent Perceiver | §9 (future) |

---

## 15. Generic v2 architecture — configurable tokenization + fusion bake-off

> **Status: ROADMAP / design space, NOT the Phase-2 build target.** v2 bundles three *independent*
> ideas with different triggers, not one architecture. Only one — the unified conv tokenizer — is
> in the build now (folded into §3.5 / §5). The other two are deferred: **typed tokens** (trigger:
> onboarding a robot outside the training mix) and **dual-path fusion** (trigger: flat plateaus or
> we want the last 1–2%). The plan of record is **§16**. Keep this section as the map, not the order.

§5 is the concrete v1 (hardcoded `196 / 8 / 15`, linear-vs-CNN split, flat Perceiver only).
This section generalizes it after three concerns (raised 2026-07-15) and a literature pass:
(1) unify the tokenizers so "linear vs CNN" is one configurable knob; (2) stop hardcoding the
input dims so the encoder is robot/sensor-agnostic; (3) treat factorized / dual-path fusion as
a real contender, not a throwaway ablation. **v2 recovers v1 exactly as a special case** — it's a
generalization, not a rewrite.

### What changed vs v1 (§5)

| stage | v1 (§5) | v2 (this section) |
|---|---|---|
| dims | hardcoded `196 / 8 / 15` | **config-driven stream registry** — dims are data |
| tokenizer | linear for sparse, 1-D CNN for ee | **one conv patchifier**, per-stream `(kernel, stride)`; `k=1` ≡ linear |
| token identity | modality + position + time | **+ sensor-identity** (typed, self-describing tokens) |
| variable DOF | pad-to-max + mask | pad+mask (kept) **or** any-variate typed tokens (new, for unseen robots) |
| fusion | flat Perceiver only | **flat Perceiver AND factorized/dual-path — two first-class arms** |
| ee encoder | 1-D CNN | conv **or** dual-path (SepFormer/DPRNN lineage) |

### Pipeline (shapes tracked end-to-end, same style as §5.1)

```
════════════ 0. STREAM REGISTRY (config, NOT code — no hardcoded dims) ════════════

 For each input stream s, one config entry:
   { name, channels C_s, seq_axis, kernel k_s, stride s_s, sensor_id, rate }
 Adding a robot / sensor = adding an entry.  196 / 8 / 15 become config values.

 example registry:
   vision : C=768, axis=patch, k=1,  s=1, id="cam0",   rate≈10Hz
   motor  : C=3,   axis=row,   k=1,  s=1, id="joint*",  rate≈10Hz
   ee     : C=15,  axis=time,  k=16, s=4, id="ft0",     rate≈100Hz   (encoder = conv | dualpath)

════════════ 1. RAW INPUTS  (a variable SET of streams, each [B, L_s, C_s]) ════════════

 stream 1 (vision)      stream 2 (motor)       stream 3 (ee)          ...  stream S
 [B, L_1, C_1]          [B, L_2, C_2]          [B, L_3, C_3]
 e.g.[B, C·196, 768]    e.g.[B, C·8, 3(+m)]    e.g.[B, C·13, 15(+m)]
 + per-token timestamp  + per-token timestamp  + per-token timestamp

════════════ 2. UNIVERSAL CONV PATCHIFIER  (one class, per-stream k,s;  k=1 ≡ linear) ════════════

 Conv1d(C_s → d, kernel=k_s, stride=s_s)  over the stream's seq axis  → [B, n_s, d]
   vision  k=1, s=1  → [B, C·196, d]     (k=1 recovers today's linear proj_v)
   motor   k=1, s=1  → [B, C·8,   d]     (k=1 recovers today's linear proj_m)
   ee      k=16,s=4  → [B, n_ee,  d]     (local temporal patches; n_ee ≈ L_3/s)
                        └── OPTION: swap conv → dual-path encoder for the dense stream
 [Early-Convs (arXiv:2106.14881) · PatchTST (2211.14730) · MOIRAI multi-patch-size (2402.02592)]

════════════ 3. TYPED SELF-DESCRIBING TOKENS  (+= embeddings; the robot-agnostic part) ════════════

 each token +=  modality_emb  +  SENSOR-ID emb  +  position  +  continuous-time emb(t)
                                 ▲ NEW vs v1: "I am Fx of ft0", not just "I am ee"
 ⇒ variable DOF / sensors = variable token COUNT (no fixed 8 or 15)
 fallback: pad-to-max + mask          (validated at scale by CrossFormer, arXiv:2408.11812)
 option:   any-variate attn bias      (within-variate vs cross-variate)  [MOIRAI]
 [mTAN time emb (2101.10318) · SeFT set-view (1909.12064) · Raindrop leave-sensor-out (2110.05357)]

════════════ 4. FLATTEN + CONCAT all streams → one context ════════════

 [B, n_1, d] ⊕ [B, n_2, d] ⊕ … ⊕ [B, n_S, d]  →  CONTEXT  [B, M, d]
                                                  M = Σ_s n_s   (variable, not hardcoded)

════════════ 5. FUSION — TWO first-class arms (config-selected; benchmarked head-to-head) ════════════

  ARM A — FLAT PERCEIVER (simple default)        ARM B — FACTORIZED / DUAL-PATH (contender)
  N latents cross-attend all M at once           alternate intra-axis / inter-axis attention
  → self-attn → [B, N, d]                        (within-time ↔ across-time, à la TimeSformer;
  O(N·M); joint cross-modal × time                intra-chunk ↔ inter-chunk à la SepFormer)
  [Perceiver 2103.03206 · V-JEPA joint]          → [B, N, d]   [TimeSformer 2102.05095 /
                                                  ViViT 2103.15691: divided often WINS]
                        └──────────── same output shape ────────────┘
                                     ▼
                       LATENT ARRAY  [B, N, d]   (spatio-temporal belief)

════════════ 6. QUERY / READOUT DECODER  (unchanged from v1) ════════════

 output queries [B, O, d] carry (sensor-id, TIME)  ──attend──►  latents [B, N, d]  → preds [B, O, d]
 [Perceiver-IO 2107.14795 · Octo / CrossFormer readout tokens]

════════════ 7. LOSS  (unchanged — the JEPA objective) ════════════

 inv = MSE(preds, EMA-target embeddings of held-out (sensor,time))  +  λ · per-timestep SIGReg

════════════ EVAL ════════════
 mask a subset of streams (e.g. vision only) → read out z_v  (probe / downstream)
```

### Stage-by-stage (what's new and why)

- **0. Stream registry.** The dims move from code to config. A stream is described by its channel
  count, which axis it sequences over, its patchifier `(kernel, stride)`, a sensor identity, and
  its rate. Onboarding a new robot/sensor = a new entry, not a code change. This is the
  Gato/Octo/CrossFormer "tokenize anything" stance made explicit.
- **1. Raw inputs = a variable *set* of streams.** No fixed three-lane assumption; `S` streams, each
  a `[B, L_s, C_s]` signal + timestamps.
- **2. Universal conv patchifier.** One `Conv1d(C_s→d, k_s, s_s)` per stream. `k=1` is *exactly* a
  per-token linear layer, so it reproduces v1's `proj_v`/`proj_m`; `k>1` gives temporal patches for
  dense streams. Patch-embedding-is-convolution (Early-Convs) and patching-as-tokenization
  (PatchTST) are standard; MOIRAI's multi-patch-size projection is the precedent for tying patch
  size to a stream's rate. The dense stream's patchifier is swappable for a dual-path encoder.
- **3. Typed self-describing tokens.** The one structural change that buys robot-agnosticism: each
  token carries a learned **sensor-identity** embedding, so a 6-DOF vs 7-DOF arm just emits a
  different *number* of joint tokens rather than needing a fixed padded slot. Pad-and-mask is kept
  (CrossFormer proves it scales); any-variate typed tokens (MOIRAI/Raindrop) are the path to
  *unseen*-robot transfer (Raindrop's leave-sensor-out result is the evidence).
- **4. Flatten + concat.** Same as v1, but `M = Σ_s n_s` is now variable — the model never assumed a
  fixed total.
- **5. Fusion — two arms.** Flat Perceiver (Arm A) is the simple default; factorized/dual-path
  (Arm B) is a genuine contender, because the video literature (TimeSformer/ViViT) shows *divided*
  space-time attention often beats *joint* on accuracy, not just compute. The Perceiver bottleneck
  removes the compute argument for factorizing, but not the accuracy argument — so this is an
  experiment to run, not a settled choice. Both arms emit `[B, N, d]`.
- **6–7 + Eval.** Decoder, loss, and vision-only readout are unchanged from §5.

### Honest status (what's contested / a bet)

- **Flat vs factorized fusion for *this* setting is unresolved.** Video evidence → factorized;
  Perceiver/V-JEPA → joint. Nobody has run flat-Perceiver vs factorized-Perceiver on multi-rate
  robot sensor fusion. **This is the experiment.**
- **Typed-token unseen-sensor transfer is proven for healthcare TS (Raindrop), not high-rate robot
  F/T + vision.** Right bet, still a bet.
- **No paper does exactly this stack** (JEPA-trained, multi-rate, cross-embodiment, vision + dense
  F/T). v2 composes validated pieces; the composition is the novelty and is itself unvalidated.

### References
Early-Convs (arXiv:2106.14881) · PatchTST (2211.14730) · MOIRAI (2402.02592) · DPTNet (2007.13975)
· SepFormer (2010.13154) · TimeSformer (2102.05095) · ViViT (2103.15691) · Perceiver (2103.03206) ·
Perceiver-IO (2107.14795) · Gato (2205.06175) · Octo (2405.12213) · CrossFormer (2408.11812) ·
Raindrop (2110.05357) · SeFT (1909.12064) · mTAN (2101.10318).

---

## 16. Phase-2 build target (plan of record)

The unambiguous scope for the Phase-2 run. Everything here is v1 (§5, the flat Perceiver) **plus
exactly one** v2 idea (the unified conv tokenizer, §3.5). Nothing else from §15 is in scope.

**In scope:**
1. **Flat multi-rate Perceiver** (§5) — the minimal temporal extension of v0.1: tokenize each
   stream, tag every token with continuous time, one flat `PerceiverBlock`, Perceiver-IO predictor.
2. **Unified conv tokenizer** (§3.5) — one `Conv1d(C_in→d, k, s)` class; `k=1` for vision/motor
   (≡ v0.1 `proj_*`), `k=16` for ee. Refactor for configurability; `k=1` reproduces v0.1 exactly.
3. **Pad-to-max + mask** for variable DOF (v0.1's existing scheme — validated in Phase 1).
4. Masking over `(modality × time)`, per-timestep SIGReg, warm-start from v0.1.

**Explicitly OUT of scope (deferred, with triggers):**
- **Typed / any-variate tokens** (§15 stage 3). Trigger: a robot outside the RH20T training mix.
  Reason to wait: pad-and-mask already covers the 4-robot goal; typed tokens add cost + regression
  risk for a capability (unseen-robot transfer) we're not chasing yet.
- **Dual-path / factorized fusion arm** (§15 stage 5). Trigger: flat clears the Phase-2 gate but
  plateaus, or we want to chase the accuracy delta the video literature reports. Reason to wait:
  build one fusion arm first so any gain/loss is attributable to *temporal*, not to a second
  architecture changed in parallel.

**Why this scope (the sequencing argument):** we have a validated v0.1. The lowest-regret change is
the *minimal* one that adds time, so a win/loss is attributable to temporal alone. Bundling
embodiment-agnosticism and a second fusion arm before clearing the Phase-2 gate builds
infrastructure for capabilities we haven't earned the right to need. Generality pays off once a
second/third use case pulls on it; today there's one.

---

## 17. Multi-camera (NOW THE IMMEDIATE BUILD — v0.2 §2.1; see PLAN.md §Phase 2)

**PROMOTED 2026-07-17.** After the temporal-in-encoder gate failed twice (§18.14) the roadmap was
re-scoped: temporal-in-encoder is retired and **multi-cam is now the immediate v0.2 build** — see
PLAN.md §Phase 2 (v0.2) **2.1**. This section is no longer "deferred / ready next time"; it is the
spec for the near-term work. Single-timestep, spatial-only: `[B, 1, n_cam·196, 768]`. Originally
raised 2026-07-16 (re: LeRobot 3D, github.com/SergioMOrozco/lerobot_3d).

### How we'd do it (near-free in the flat Perceiver)
Multi-camera is the *same move* as adding time or a modality: **more tokens.** Each camera `k`,
each tick `c` → 196 ViT patch tokens, all poured into the flat pile tagged
`(modality=vision, camera-id, patch-pos, time)`. Cross-attn is linear in token count, so `K`
cameras = `K·C·196` vision tokens — no architectural change. **v0.1 already supports this**:
`mm_perceiver3.py` accepts a *list* of camera tensors ("cameras = one big camera", `K×196`
concatenated); the temporal model extends identically.

The one real addition: a **camera-identity embedding** (or better, camera pose/extrinsics) so views
are distinguishable and the model can reason about geometry — this is the typed-token idea (§15)
applied to cameras. v0.1's "one big camera" omits it (views interchangeable).

Two levels (mirrors the parked "multi-view" note in PLAN.md):
1. **Multi-cam as input** (do-first, cheap): concat cameras' tokens + a camera-id embedding.
2. **Cross-view objective** (richer, parked): mask one view, predict its latent from the other
   views + state — predict-don't-equate, **never latent-equality across views** (wrist vs external
   have info asymmetry). Gate it on first measuring same-tick cross-view latent distance: if views
   already cluster, the objective isn't needed.

### The data catch
The current RH20T cache is **single-camera** — `chunk_state.py` picks one external serial and
excludes the wrist (`IN_HAND_OF_CFG`). RH20T *has* multiple cameras per rig, but we only cached one.
So multi-cam needs a **re-precompute** (extract + ViT-encode `K` cameras/tick → `K×` bigger vision
cache). Data job, not a model job.

### The geometric alternative (LeRobot 3D)
LeRobot 3D fuses cameras **geometrically**: per-camera *depth* → 3D points → one merged scene
**point cloud**, with extrinsics calibrated by ICP against the robot URDF. Explicit 3D; needs depth
cameras + calibration. Different philosophy from ours (geometric vs learned-latent). It could become
an *input stream* for us (tokenize a fused point cloud as another modality) if we ever want explicit
3D — but it is not our fusion mechanism and is not needed for v1.

**Trigger: NOW** (temporal gate failed → this is v0.2 §2.1). Start with input-concat + camera-id; keep
the cross-view objective and any explicit-3D/point-cloud input as later stretch options. **Cheap
pre-check before paying for the K-camera re-precompute:** encode each view separately with the *existing*
v0.1 encoder and measure same-tick cross-view latent distance — if views already cluster, multi-cam buys
little; if they carry complementary info (wrist vs exo — expected), the re-precompute is justified. See
§21 execution notes.

---

## 18. Hypotheses & evaluation protocol (what v1 proves, and how we package it)

Framing added 2026-07-16. Treat v1 as **hypothesis-first** (mirrors the paper). Two null
hypotheses at two stages; **reject NH1 (encoder) first** — NH2 (downstream) only matters after.
This section supersedes/expands the gate in §10.

### 18.1 The null hypotheses
- **NH1 — encoder (the v1 gate):** *the temporal latent does not learn time well or efficiently* —
  adding the time dimension gives no representational advantage over the v0.1 single-timestep encoder.
- **NH2 — predictor (downstream, later):** *the temporal state as g(·) into an action head is NOT
  better than VJEPA-AC or our own non-temporal (v0.1) checkpoint.* (FLARE-style "encoder as g(·)";
  FLARE shows g(·) quality is the deciding factor — see §"External validation lead" in PLAN.md.)

Everything in §18.2–18.3 tests **NH1**. NH2 is §18.4 (deferred).

### 18.2 NH1 eval protocol — three tests, all head-to-head vs v0.1
Fixed baselines across all three: **v0.1 single-tick z_v**, plus **naive** (last-frame
carry-forward, linear extrapolation) where applicable. "Learns time *well & efficiently*" = A + B + C.

- **(A) Future-state prediction — PRIMARY.** From a *past* window (ticks 0…t), predict force/state
  at **t+Δ** for several Δ. **Reject NH1 if** temporal z > v0.1 *and* > naive, and the **gap grows
  with Δ** (a single frame structurally can't extrapolate — this is the core discriminator).
- **(B) Dynamics probe — cleanest "time learned" signal.** Probe z for quantities a single frame
  has zero info about: joint velocity `dq`, force rate `dF/dt`, motion. **Reject NH1 if** temporal
  z ≫ v0.1 (v0.1 should be ~0 here by construction).
- **(C) Efficiency / is-time-actually-used — ablations.** (c1) **remove the continuous-time
  embedding** → performance must drop (else time isn't used). (c2) **continuous-time (no resample)
  vs resample-to-grid + 1-D-CNN** (§2.5 decisive ablation) → "efficiently" = no-resample ≥ resample.
  (c3) **window-length C sweep** → should improve with more context, then plateau.

### 18.3 The results table (what we report vs v0.1)
| metric | v0.1 (single-tick) | v1 (temporal) — reject NH1 if |
|---|---|---|
| present-time force/state R² | baseline (~0.65 ur5) | ≥ v0.1 (fusing frames shouldn't hurt) |
| **future force/state R² @ Δ** | can't / naive only | **> v0.1 & > naive; gap grows with Δ** ← headline |
| velocity / force-rate R² | ~0 (structural) | **≫ v0.1** |
| time-embedding ablation | n/a | **drops when removed** |
| continuous-time vs resample+CNN | n/a | **≥ resample** |

Report **mean ± std over ≥3 seeds**; per-embodiment breakdown on the all-cfg run (the temporal
analogue of the Phase-1 5×4 matrix). **Outcome rule:** if v1 does not beat v0.1 on future + dynamics,
**fail to reject NH1 → adopt v0.1 honestly** (this is a real, acceptable outcome).

### 18.4 NH2 — downstream / predictor (DEFERRED)
Needs three things that don't exist yet: an **action/prediction head** on the frozen encoder, a
**task**, and the baselines **VJEPA-AC** + **v0.1**. Reject NH2 if the temporal-state-fed head beats
both. This is loss-#4 / world-model territory; **VJEPA-AC needs porting (weeks)**. Gate: only after
NH1 is rejected. Ties into the FLARE / GR00T external-validation lead (encoder as g(·)).

### 18.5 Packaging (clean + reproducible — the deliverable)
- **One eval harness** (extend `eval_temporal.py`) that loads **both** the v1 and v0.1 checkpoints
  and runs tests A/B/C on the **same** windows/splits → emits `results.json` + a markdown/CSV table.
- **Matched everything:** same cfgs, same frozen `holdout_v1.csv` split, same probe (ridge,
  standardized, **fit on train only**), same Δ set, same targets. Baselines encoded on the *same*
  windowed data.
- **Multi-seed (≥3)** → mean ± std; **per-embodiment** on the all-cfg run.
- **Artifacts:** `checkpoints/temporal/<tag>/eval/{results.json, table.md}`; the future-R²-vs-Δ curve
  and the vs-v0.1 table are the blog/paper figures.
- **Log every cap/deviation** next to the numbers (subsampled cache, ee-as-snippets, single external
  cam) so results are never read as more than they are.

### 18.6 Current status vs this protocol
- **Have:** encoder trains healthily (ur5b); a **window-mean sanity probe** — which tests **none** of
  A/B/C (not future, not dynamics, smeared). It only showed "some cross-modal signal exists."
- **Missing:** the entire NH1 gate (A future, B dynamics, C ablations), v0.1-baseline loading,
  multi-seed, all-cfg.
- **⇒ NH1 harness now built** (`gate_eval.py`); first result in §18.7. NH1 still not *rejected*
  (directional-only so far).

### 18.7 Gate results & diagnostic saga (2026-07-16) — CONDENSED

The full day-by-day narrative moved to **`TEMPORAL_JOURNAL.md`**; the numbers + saved JSONs live in
**`results/temporal/RESULTS.md`**. Where it landed:

- **NH1 fails to reject.** Both the original masked-cell objective (query decoder) and the ported v0.1
  head fail the gate: the C=8 temporal latent is lower-rank (RankMe ~51 vs v0.1 ~134), at chance on
  dynamics (dq/dF/dt ≤ 0), and below single-frame v0.1 on future force at every horizon.
- **The v01-head fix (journal §18.12) buys STABILITY, not CAPABILITY.** At C=1 it faithfully recovers
  present force (0.251 ≈ v0.1 0.253; pre-fix ≈0), no collapse — but at C=8 present force halves (0.10 vs
  0.212) and the gate still fails.
- **Root cause = the temporal FUSION** (flat-Perceiver-over-ticks → mean-pool), not the head or masking:
  two independent objectives both fail while head and masking were each independently ruled out (journal
  §18.9–18.11). The mean-pooled window fuse degrades the latent (rank + present-force dilution).
- **Fix direction** = the LeWM/RoboTTT redesign (§20/§20.1): per-frame `z_t` + a separate next-embedding
  predictor on top of the v0.1 cross-modal head + joint-SIGReg. The eval protocol (§18.1–18.6) stays.

Timeline (detail in journal): 18.7 first ur5 gate (directional only) → 18.8 full matrix fails → 18.9
pooling exonerated → 18.10 masking-fix fails → 18.11 C=1 exonerates temporal, implicates the
predictor/objective → 18.12 the vise + port v0.1 head → 18.13 probe confirms (C=1 ok, C=8 halves) →
18.14 gate still fails → redesign.

---

## 19. Deeper latent evals — the physics-probe matrix (eval north star; mostly aspirational)

Framework noted 2026-07-16 (JEPA / intuitive-physics eval thinking). Three orthogonal axes → a probe
matrix; the skill is picking the few **testable, meaningful** cells. This sits **one layer above the
NH1 gate (§18)** — NH1 = "is temporal worth it at all" (cheap, now); this = "what physics does the
latent understand" (rich, blog/paper-worthy, mostly later). **Do not let it delay the gate.**

### The three axes
1. **Variance property (VICReg-style)** — maps directly onto our objective:
   - *Invariance*: same task / view / lighting → close representations. (We do NOT force this across
     modalities — "predict-don't-equate".)
   - *Variance*: latent diverges when the perceived physics changes. (= our SIGReg / anti-collapse.)
   - *Covariance*: varying mass/friction in the same scene → consistent cross-latent structure.
2. **Structural dimension** — intra-modality · inter-modality · temporal · compositional (emergent
   only when combining modalities/time) · architectural (encoder vs predictor localization, later).
3. **Physics concept** — occlusion permanence · causality · contact/friction/slippage · inertia ·
   deformability · variable weight.

Matrix = concept × variance × dimension. Most cells are meaningless; select the useful ones
(e.g. *counterfactual mass × causal × inter-modality*; *occlusion × variance × temporal*).

### The binding constraint = DATA (triage on RH20T)
RH20T = single-cam RGB + joints + F/T over time; **no physics-param labels, no counterfactuals, no
multi-cam.** So:

- **Testable now (overlap heavily with NH1):**
  - *contact/force × variance × inter-modality* — does `z_v` diverge at contact onset? Label contact
    from the F/T stream. (Our cross-modal force story, sharpened.)
  - *inertia / force-dynamics × temporal* — velocity `dq`, force-rate `dF/dt` from the latent
    (= the NH1 dynamics probe).
  - *invariance × intra-modality via augmentation* — same frame + lighting/crop → close `z`. Cheap,
    no new data; a clean variance-property check.
- **Needs data we lack (defer):**
  - *cross-view invariance* → multi-cam re-precompute (§17).
  - *counterfactual mass/friction × causal* ("same scene, 2× mass") → **needs SIM** or controlled
    paired data — the single most valuable cell (causal vs correlational) and the hardest.
  - *occlusion permanence, deformability, variable weight* → labels/collection we don't have.

### Implication
The counterfactual cell is the strongest science AND needs sim → it's a concrete argument **for the
real2sim roadmap** (sim gives free counterfactuals: same scene, vary mass/friction). The physics
matrix doesn't just sit beside real2sim — it motivates it.

**Sequencing:** run the testable subset *alongside* NH1; treat the sim/multi-cam/label-gated cells as
roadmap that the real2sim + multi-cam work unlocks.

### 19.1 Constructed probes (runnable specs) + what to actually run

| # | probe | cell (concept × variance × dim) | input → label | metric | baseline | reject "no physics" if | data |
|---|---|---|---|---|---|---|---|
| P1 | Contact detection | contact × variance × inter-modality | `z_v` → binary contact `‖F/T‖>τ` (from hidden ee) | AUROC | raw ViT; v0.1 | `z_v` ≫ raw & chance; temporal > v0.1 | ✅ now |
| P2 | Force regression | contact × variance × inter-modality | `z_v` → 6-dim F/T at a tick | R² | raw; v0.1 | `z_v` > raw; temporal ≥ v0.1 | ✅ now |
| P3 | **Dynamics** | inertia × variance × temporal | `z` → `dq` (velocity), `dF/dt` | R² | v0.1 (~0 by construction) | temporal **≫ v0.1≈0** | ✅ now (`dq` in packet) |
| P4 | **Future contact/force** | causality × variance × temporal | past window `z` → contact/F at `t+Δ` | AUROC/R² vs Δ | v0.1; naive last-frame | temporal > both, **gap ↑ with Δ** | ✅ now |
| P5 | Aug-invariance | invariance × — × intra-modality | same frame + light/crop → cos sim `z` | alignment | — | high within / low across scenes | ⚠️ needs raw frames + ViT-with-augs (cache is post-ViT) |
| P6 | Counterfactual mass | mass × covariance × inter-modality | same visual scene, 2× mass → `z` shift | Δz vs mass | correlational control | `z` tracks mass with vision fixed | ❌ needs SIM |
| P7 | Occlusion permanence | occlusion × variance × temporal | object hidden → `z` retains it | probe over Δ | — | latent persists through occlusion | ❌ needs occlusion data/sim |

**Priority — what to actually run:**
1. **P3 (dynamics) — FIRST.** Cheapest + cleanest discriminator: v0.1 ≈ 0 by construction, so temporal ≫ 0 is an unarguable "time learned."
2. **P4 (future @ Δ) — the headline** ("temporal > v0.1 & naive, gap grows with Δ").
3. **P1/P2 (present-time contact/force)** — sanity that temporal ≥ v0.1 and cross-modal holds.
4. **Time-embedding ablation** (zero the time emb) — proves time is *used*, not decorative.
- P3+P4 alone answer "does temporal help." If yes → scale (≥3 seeds, then all-cfg per-embodiment matrix). If no → stop and rethink.

**Not now (gated):** P5 (re-encode), P6 (sim / real2sim — flagship-later), P7 + cross-view (data/multi-cam), NH2/VJEPA-AC (needs action head + port, weeks).

**Key convergence:** P1–P4 *are* the NH1 gate (§18.2) — **one build, not two.** Both encoders (v1 `temporal/ur5b`, v0.1 `phase1/ur5`) are already trained, so this is a probe script + minutes to run.

---

## 20. External: LeWorldModel (arXiv 2603.19312, Maes/Le Lidec/Scieur/LeCun/Balestriero) — what it validates, what we take (2026-07-16)
Surfaced by Dhanoosh (Slack). First JEPA to train stably **end-to-end from pixels** with only **two**
losses: (1) **next-embedding prediction** `ẑ_{t+1}=pred(z_t,a_t)`, MSE `‖ẑ_{t+1}−z_{t+1}‖²` (separate
6-layer transformer predictor ~10M, autoregressive + causal mask over an N-frame history, actions via
**AdaLN**); (2) **SIGReg** (Balestriero & LeCun 2025 *LeJEPA* — **the same regularizer we use**) on the
**marginal** latent. **No EMA, no stop-grad, no pretrained encoder** (~15M total). Validates via
physical-quantity probing (location/angle) + a decoder + surprise detection = *our* eval methodology.

**The governing mismatch (why it is not a drop-in).** LeWM is **unimodal (pixels) + trains the encoder**.
Our force signal is a **cross-modal readout from a FROZEN encoder** — the vision latent is forced to
reconstruct the ee/force modality. LeWM has **no cross-modal supervision**, so its next-embedding loss
does **not** train the vision→force map. Adopting it wholesale would recreate exactly the v0.2 failure
(§18.11–12: cos θ≈0.05, force→0).

**Does it fix what we see now (§18.13)?**
- *Present-force halving* — its **objective**: no (dynamics, not present-state, not cross-modal). Its
  **architecture**: yes — LeWM keeps a clean per-frame `z_t` + a separate predictor, never dissolving
  frames into one window pool → matches the §18.13 fix directions (i)/(ii).
- *NH1 "is temporal worth it"*: yes — next-embedding is a cleaner temporal objective than our masked-cell
  hack and **is** what P4 (future) measures. But run the gate first.
- *The force-destroying objective*: LeWM doesn't fix it; **keep v0.1's cross-modal head.**

**What we take — the implied recipe (ADD, don't replace):**
1. **Objective** = cross-modal recon head (v0.1 — keeps force, **non-negotiable, no LeWM analog**) **+**
   next-embedding prediction (LeWM — adds real dynamics) **+** joint-SIGReg (shared stabilizer).
2. **Try dropping EMA** — LeWM shows SIGReg-alone suffices; consistent with our finding that SIGReg is
   the real stabilizer (was being *overpowered*, §7.3). Clean ablation.
3. **Per-frame architecture** (keep `z_t`, don't window-pool) → the §18.13 present-force fix.
4. **AdaLN action-injection** — bank for the deferred NH2/action head (cleaner than V-JEPA2-AC concat).

**Dhanoosh's "gaussianising might not be temporally optimal" — STILL OPEN, now a differentiator.** LeWM
applies SIGReg to the **marginal** (exactly the form he suspects) and never tests a residual/innovation
form `r_t=z_t−f(z_<t)` (our ablation iii, §7.3). So LeWM shows marginal-SIGReg is **sufficient**
(stability), not **optimal**. Residual-SIGReg-under-time in a temporal world model is an **un-done
ablation we could own.** (His "Gaussian shape but informative latent" concession is confirmed by LeWM:
SIGReg-Gaussianized latents that still linearly decode physical quantities.)

**Is it V-JEPA2-AC?** Same family (action-conditioned latent world model) but distinct: V-JEPA2-AC
*post-trains* a small action predictor on a **frozen, giant, EMA-pretrained V-JEPA2**; LeWM is
end-to-end-from-pixels, no-EMA, SIGReg-only. **We sit between**: frozen backbone like V-JEPA2-AC +
SIGReg like LeWM — a defensible, novel spot.

Refs: LeWM `arxiv.org/abs/2603.19312`; V-JEPA 2 `arxiv.org/abs/2506.09985`.

### 20.1 Also considered — Prannay's three (2026-07-16); triaged against §18.14
None is a drop-in fix for the gate failure (temporal fusion degrading the latent). Verdicts by track:

- **RoboTTT — *Context Scaling for Robot Policies* (NVIDIA GEAR; Jiang/Chebotar/Zheng et al.)
  `research.nvidia.com/labs/gear/robottt/`.** Scales visuomotor **context to 8K ticks** by compressing
  history into **fast weights** (gradient descent at train *and* inference; TBPTT + sequence action
  forcing) instead of a fixed-capacity attention/latent. A **VLA policy** (GR00T N1.7), explicitly **no
  latent world model / no JEPA**. **The useful one — it validates §18.14:** its thesis is that
  fixed-capacity context does NOT scale and the fix is **recurrent memory** → independent vote that our
  "flat Perceiver over ticks → mean-pool" is the wrong shape. BUT it's a **policy (deferred NH2)**, not
  the NH1 encoder we're stuck on, and TTT (grad-descent at inference) is the **opposite of the edge/int8
  feed-forward ARM pitch** ([[kepler-arm-collaboration]]). **Take the framing** (context-length as a
  scaling axis; recurrence > fixed-window fusion — echoes the RSSM/Dreamer carry-forward we'd flagged),
  **not the mechanism.** Bank for NH2: context scaling + loss-masking (human video as context w/o action
  targets).
- **MISA — *Modality-Invariant and -Specific Representations* (Hazarika 2020) `arxiv.org/abs/2005.03545`.**
  Decompose each modality into a **shared invariant subspace + a private modality-specific subspace**,
  then fuse. Supervised sentiment, not temporal/SSL. **Relevance = the rank-collapse SYMPTOM** (§18.14
  fused latent rank 51 vs v0.1 134): a shared+private split preserves capacity (private subspaces don't
  dissolve into the shared one). **Caveat:** the *invariant* half conflicts with our "predict-don't-equate"
  stance (§7) — so it's a lens on the **cross-modal head / rank**, not a temporal-dynamics fix. Worth a
  look if the redesign still shows rank collapse.
- **TIPSv2 — vision-language pretraining, dense patch-text alignment (`arxiv.org/abs/2604.12012`).** Better
  image-text encoder (iBOT++, patch distillation, EMA/caption recipes). Not a world model / not temporal /
  not robotics. **~No relevance to the gate.** Only angle: a **candidate upgraded frozen backbone**
  (spatially-aware dense patches → richer vision tokens; matters for the backbone-dominated ARM compute
  story). Park it.

**Convergence:** RoboTTT + LeWM (§20) independently say the **fixed-latent window-pool is the wrong shape**
— exactly what §18.14 localized. Strengthens the redesign (per-frame `z_t` + separate predictor +
next-embedding term), doesn't add a new direction. MISA = a rank-collapse lever for the fusion head;
TIPSv2 = orthogonal backbone.

---

## 21. v0.2 execution notes — cheap pre-checks + design guardrails (2026-07-17)

Added after the re-scope (PLAN.md §Phase 2). The *what/why* is in PLAN.md + §17/§20; this is the *how to
build it without repeating v0.2's mistakes*. Two cheap pre-checks answer "will it work" before we spend
compute, and four guardrails keep the builds from re-breaking the per-frame latent.

### 21.1 Ordering is a priority call, not a dependency
The two v0.2 builds are **independent**: the next-embedding predictor (§2.2) sits on the frozen per-frame
encoder and needs **only existing data** (`checkpoints/phase1` + `caches/cfg*.npz` + `window_loader`);
multi-cam (§2.1) needs a **new K-camera re-precompute** (§17 "data catch"). So the predictor pre-check is
runnable *today* while the multi-cam precompute runs in parallel — don't serialize 1→2 by reflex. This
mirrors what saved us at the gate: cheap probe before expensive build.

### 21.2 Pre-check A — is the future latent even predictable? (runnable NOW, no precompute)
Before building the full carry-forward belief-state: freeze v0.1 (`checkpoints/phase1`), dump per-frame
latents over the existing temporal windows (`caches/cfg*.npz`, kuka has velocity → use cfg6/7), fit a
**simple** `z_t → z_{t+Δ}` predictor (linear / small MLP), and compare to the **naive carry-forward**
baseline (`ẑ_{t+Δ}=z_t`) on held-out windows, at several Δ.
- If a simple predictor **beats naive at any Δ** → dynamics exist; the direction is validated for ~an
  afternoon; proceed to the real predictor.
- If even a strong predictor **can't beat naive at any horizon** → red flag *before* investing in the
  belief-state. Expect the win (if any) to appear at **longer Δ / contact-onset**, not short Δ (there
  future force ≈ present force, so there's almost no headroom over carry-forward — cf. §18.14 gate).

### 21.3 Pre-check B — do views carry complementary info? (before the K-camera re-precompute)
See §17 trigger: encode each view separately with existing v0.1, measure same-tick cross-view latent
distance. Complementary (wrist vs exo) → multi-cam justified; already-clustered → little gain. Cheap read
before paying for the `K×` vision cache.

### 21.4 Guardrails (so neither build recreates the §18 failure)
1. **Freeze the encoder for the predictor.** The v0.2 failure was that *training degraded the per-frame
   latent* (§18.11). Freezing the whole v0.1 encoder (Perceiver + cross-modal head) and training only the
   predictor makes that failure mode **structurally impossible**. Joint fine-tune (LeWM proper) is the
   risky variant → **defer** until the frozen version is proven.
2. **Predict the latent *set*, not a mean-pooled vector.** Mean-pooling is the specific thing that diluted
   force (§18.9/§18.13). Have the predictor target the 8/64-latent *set* (spatially/structurally resolved)
   so it can reason about *where* contact happens; a single pooled `z_t` throws that away.
3. **§2.2 is the *unconditioned* precursor to loss #4, not loss #4.** Plain next-embedding = `ẑ_{t+Δ}=pred(z_{≤t})`.
   Loss #4 = that **+ action conditioning** (`ẑ_{t+1}=pred(z_t,a_t)`). Build/prove the unconditioned one
   first (§20 recipe: v0.1 cross-modal head kept frozen + next-embedding term + joint-SIGReg, SIGReg
   per-timestep never time-pooled, §7.3); add actions only after.
4. **Multi-cam bottleneck-collapse is the real §2.1 risk.** More input tokens into the *same* fixed 8/64
   latents = more compression pressure → can rank-collapse like temporal did. It's "informative either
   way," not "low risk": if RankMe/probe R² drop under N views, the fix is a **known knob** (more latents),
   but measure it — don't assume the bottleneck absorbs N views for free.

### 21.5 FLARE de-risk framing (what §2.2 does and doesn't deliver)
The standalone predictor is **not** FLARE — FLARE's predictor lives inside the VLA using our g(·) as the
target encoder (code unreleased, [[gear-flare-collab]]). What §2.2 delivers is the **de-risk**: proof our
multimodal frozen latents have learnable dynamics + a fully-specified, reproducible predictor. FLARE's own
ablation says g(·) quality drives policy success, so even a modest future-force R² still carries
integration value — the downside is hedged.
