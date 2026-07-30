# mm_perceiver4 — design notes

A rebuild of `mm_perceiver3.py` that stops confusing the **temporal**, **spatial/sensor**,
and **physical** axes of each modality, and repurposes the Perceiver from a spatial pooler
into a **per-modality temporal resampler** so all modalities can be aligned on a common
rate *before* any cross-modal mixing.

## The problem with mm_perceiver3

Every modality was flattened into one undifferentiated token soup and pooled:

```
concat([196 vision patches, 8 motor rows, 13 ee ticks]) -> [B, K*196+21, d] -> Perceiver -> [B, d]
```

Three semantically different axes got concatenated along one token dimension:

- vision's `196` is **spatial** (patches),
- motor's `8` is a **channel/sensor** axis (7 joints + gripper),
- ee's `13` is **time** (100 Hz samples inside one tick window).

Treating ee's time axis as tokens meant `ee_mask` (a mask over *time*) was fed into the
fusion attention mask as if it masked tokens. The model never had a notion of sampling
rate, and could not align modalities that tick at different rates.

## Canonical input: `[B, T, S, P]`

mm_perceiver4 **strictly enforces** a 4-D layout per modality, no backward compatibility:

```
[B, T, S, P]   Batch, Time (chunksize*sampling_rate), Spatial/sensor, Physical
```

| modality | shape (current cache) | T | S | P |
|----------|-----------------------|---|---|---|
| `rgb`    | `[B, 1, K*196, 768]`  | 1 frame           | K*196 patches        | 768 ViT dims |
| `motor`  | `[B, 1, 8, 6]`        | 1 tick            | 8 rows (7 joints + gripper) | 6 = 3 vals + 3 mask bits |
| `ee`     | `[B, 13, 1, 15]`      | 13 hi-freq samples | 1 (no spatial extent) | 15 = F/T 6 + tcp xyz 3 + rot6 |

`unpack()` reshapes the existing cache packet into this layout; nothing downstream is
allowed to see a 3-D tensor.

## Pipeline

All stages are **per-modality** until fusion. The order is fixed **A** (spatial → temporal);
see "Ordering" below for why the alternatives don't apply here.

### 1. Stem — physical → d, *temporal* (Conv1d)

Physics is a time series, so the `P → d` channel map is a **convolution over time**, not a
per-frame linear. `Conv1dStem` runs as `[B*S, P, T] -> [B*S, d, T]` (kernel with 'same'
padding; kernel=1 degrades to a per-timestep linear, which is what a T=1 modality wants).

```
[B, T, S, P] --Conv1d over T--> [B, T, S, d]
```

### 2. Spatial compression — S → n_s (config)

`SpatialCompress`, switchable:

- `spatial="perceiver"` — `n_s` learned queries cross-attend over the sensor axis,
- `spatial="avgpool"` — plain mean over S (`n_s = 1`).

Simple is a valid choice; both are exposed as a config knob for ablation.

```
[B, T, S, d] --> [B, T, n_s, d]
```

### 3. Temporal resampling — T → R (the point of the redesign)

`TemporalResample` uses the **Perceiver as a resampler**: `R` learned queries cross-attend
over the time axis, so a modality with *any* native rate lands on a common `R`. This is
where alignment happens — per modality, on the time axis, before cross-modal mixing.
`ee_mask` is applied here as a **mask over time** (its correct home), and samples with no
valid ee at all are handled downstream (excluded from ee loss terms).

```
[B, T, n_s, d] --R queries over T--> [B, R, n_s, d]
```

**T=1 caveat.** rgb and motor are `T=1` in the current cache, so their temporal stage is a
*learned broadcast* (R queries attending to one token), not real resampling. Only `ee`
(T=13) exercises the mechanism today. T=1 deliberately runs through the **same** resampler
code as T>1 — no special-casing — so the T=1 ablation compares *architectures*, not code
paths. Real temporal resampling for rgb/motor needs `precompute_chunks.py` to emit
multi-tick chunks (a separate preprocessing change).

### 4. Fusion — flattened, MLA over time

After stage 3 all modalities share `[B, R, n_s, d]`. Stack and flatten the
modality/sensor/feature axes into one **wide** token per timestep:

```
stack 3 modalities -> [B, R, 3*n_s, d]  --flatten-->  [B, R, w],   w = 3*n_s*d
```

The cross-modal mix lives inside the transformer, not in a separate pre-projection. A
learned **positional embedding over R** is added (time order matters), then a stack of
`WideBlock`s:

- **residual stream and FFN stay wide** (`w`) — they carry the full modality/sensor/feature content,
- **attention is Multi-head Latent Attention (MLA)** — Q and K/V are compressed through
  separate low-rank latents (`q_latent`, `kv_latent`), with `W_qk` absorbing `W_q^U W_k^U`
  so scores are computed directly in the kv-latent space; the value is up-projected back to
  `w` on output. MLA is the principled low-rank factorization of "project the wide stream
  down for attention, back up on output". Attention is over the **R time axis**.

Finally pool over R and read out `w -> d`:

```
[B, R, w] --WideBlock x depth--> [B, R, w] --mean over R--> [B, w] --readout--> [B, d]
```

**Hiding a modality** (for the cross-prediction objective) is **feature-zeroing**: a
flattened fuse has no attention column to block, so the modality's `n_s*d` slice of the
wide token is zeroed before the block stack. This is the weaker (feature-level) form of the
hide-one ablation, accepted as the cost of the simpler flattened fusion.

## Training objective (unchanged from mm_perceiver3)

Cross-modal masked latent prediction ("predict, don't equate"):

- **hide-one cross-prediction** — for each modality, fuse with that modality zeroed and
  predict its EMA-target embedding from the fused latent (MSE),
- **per-modal SIGReg** + **joint SIGReg** on the fused latent (`SlicedEppsPulley`),
- **EMA-target stems**, one per modality, updated by `update_target()`.

ee can be entirely absent (all of cfg5, ~1/3 of cfg3): those samples are excluded from the
ee prediction and ee SIGReg terms rather than averaged over zero valid entries (NaN trap).
`embed_vision()` returns the vision-only fused latent (motor + ee hidden) — the eval latent.

## Ordering

With a flattened linear/MLA fuse, fusion is no longer an attention over modality tokens, so
the two mm_perceiver-style variants that fused *by attention* collapse:

- **Order C** (spatial compression conditioned on other modalities, done inside a joint
  cross-attention) is **impossible** — there is no joint attention step to carry it.
- **spatial-first (A)** vs **temporal-first (B)** would still differ, but B resamples time
  on the full uncompressed spatial axis (e.g. 196 patch streams in parallel), which is the
  expensive path. mm_perceiver4 fixes **order A** (spatial then temporal); it is the cheap,
  natural ordering and keeps the temporal resampler operating on `n_s` compressed streams.

## Config knobs

| arg | meaning | default |
|-----|---------|---------|
| `d` | model width | 256 |
| `n_s` | spatial queries (perceiver mode) | 8 |
| `R` | temporal resample rate = time query count | 8 |
| `spatial` | `"perceiver"` or `"avgpool"` | `"perceiver"` |
| `fuse_depth` | number of MLA WideBlocks | 4 |
| `fuse_heads` | MLA heads (must divide `w = 3*n_s*d`) | 8 |
| `fuse_q_latent` / `fuse_kv_latent` | MLA latent ranks | 192 / 64 |
| `stem_kernel` | Conv1d temporal kernel | 3 |
| `lamb`, `ema`, `n_slices` | SIGReg weight, target EMA, SIGReg slices | 0.02, 0.99, 512 |

## Status

- Model implemented and shape-validated (both spatial modes, no-ee case without NaN,
  configurable MLA ranks/heads, T>1 temporal resampling with mask sensitivity, T=1
  broadcast).
- **Not yet done:** a dedicated training script (strict `[B,T,S,P]`, driven by
  `dataloader.make_loader`), and multi-tick chunks in `precompute_chunks.py` needed to make
  rgb/motor temporal resampling do real work.
