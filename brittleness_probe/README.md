# Camera-vs-placement brittleness probe

A small, self-contained study asking why VLAs break when the camera moves but
tolerate objects moving: **how far does a vision encoder's token grid travel
under a camera change versus an object rearrangement?**

Write-up and discussion:
[Why VLAs are so viewpoint brittle](https://forum.menlo.ai/t/why-vlas-are-so-viewpoint-brittle/66)

Headline: camera-only moves shift the token grid **1.3-1.9× farther** than
object-only moves in all four encoders VLAs actually deploy, and the patches
that move are not the ones that lost the objects — object identity survives,
viewpoint dominates.

| encoder | objects moved | camera moved | both | camera ÷ objects |
|---|---:|---:|---:|---:|
| SigLIP So400m | 0.460 | 0.607 | 0.633 | 1.32× |
| SigLIP2 So400m | 0.397 | 0.537 | 0.565 | 1.35× |
| DINOv2 ViT-L | 0.306 | 0.566 | 0.602 | 1.85× |
| CLIP ViT-L | 0.358 | 0.519 | 0.540 | 1.45× |
| VGGT-1B (patch, single-image) | 0.016 | 0.042 | 0.045 | 2.63× |
| VGGT-Ω-1B-512 (patch, single-image) | 0.559 | 0.690 | 0.690 | 1.23× |

## The setup

18 photos of a two-arm tabletop rig: the same four objects in **3 layouts**,
each shot from **6 camera positions** (A-C overhead, D-F at table height).
Because the grid is complete, every pair of photos differs in exactly one way:

| pair group | meaning | n |
|---|---|---:|
| `placement` | same camera, objects moved | 18 |
| `camera` | same objects, camera moved | 45 |
| `both` | both changed | 90 |

The metric is the mean cosine distance between **corresponding tokens of the
patch grid** at 224px (uniform squash), i.e. the visual state the policy's
language model consumes. A 10% brightness change on one image is carried
through every table as a scale anchor.

## Data

The 18 photos are ~87 MB of camera JPEGs and are **not** in the repo. Point the
scripts at a directory containing them:

```
gridset/
  A1_IMG_4321.JPG   B1_IMG_4322.JPG   ...   F1_IMG_4326.JPG
  A2_IMG_4327.JPG   ...
  A3_IMG_4333.JPG   ...
```

Only the **first two characters** of each filename matter: camera letter `A-F`
then layout digit `1-3`. Pass `--img-dir /path/to/gridset` or set
`GRIDSET_DIR`. Contact me for the original set, or shoot your own — a phone on
a tripod is enough, the point is that the grid is complete and the objects are
untouched within a layout.

## Environments

Two, because VGGT pins its own torch:

```bash
# encoders (SigLIP / SigLIP2 / DINOv2 / CLIP) — the repo venv is enough
uv sync

# VGGT / VGGT-Omega: separate venv, older pinned torch
uv venv .vggt-venv
.vggt-venv/bin/pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121
.vggt-venv/bin/pip install git+https://github.com/facebookresearch/vggt.git
```

DINOv2 loads from a local `torch.hub` checkout if one exists, else pulls
`facebookresearch/dinov2` from GitHub. VGGT-1B weights come from HF
(`facebook/VGGT-1B`); VGGT-Ω is **gated** — you need the source checkout on
`PYTHONPATH` and a local `vggt_omega_1b_512.pt`.

## Running it

```bash
# 1. four VLA encoders: all group stats + main figures
python probe_encoders.py --img-dir /path/to/gridset

# 2. NN-affinity bars + single-token object tracking (reads step 1's JSON)
python probe_figures.py --img-dir /path/to/gridset

# 3. clear MDS panels (no GPU, reads step 1's JSON)
python probe_mds.py

# 4. geometry-pretrained encoders, single-image + all-18-view
.vggt-venv/bin/python probe_vggt.py --model vggt  --img-dir /path/to/gridset
PYTHONPATH=/path/to/vggt-omega .vggt-venv/bin/python probe_vggt.py \
    --model omega --checkpoint /path/to/vggt_omega_1b_512.pt \
    --img-dir /path/to/gridset

# 5. VGGT figures (reads step 4's JSONs)
python plot_vggt.py --model vggt
python plot_vggt.py --model omega
```

Everything writes JSON to `results/` and PNGs to `figures/`; both are
overridable with `--results` / `--figures`.

## Layout

| file | what |
|---|---|
| `probe_encoders.py` | SigLIP, SigLIP2, DINOv2, CLIP — group stats, patch-NN affinity, k-means segments, heatmap, MDS, encoder bars |
| `probe_figures.py` | affinity bars and bezel-token tracking (SigLIP only) |
| `probe_mds.py` | MDS redrawn without thumbnails, plus a third-axis check |
| `probe_vggt.py` | VGGT-1B and VGGT-Ω, single-image and all-18-view, split by frame / global stream |
| `plot_vggt.py` | the three VGGT figures per model |
| `results/` | every distance the write-up quotes, including all 153 pairwise values per encoder |
| `figures/` | the figures as published |

## What to run next

Ordered roughly by cost. The first two need nothing beyond the existing 18
images; the third is the one that could change downstream data efficiency.

- **Quantise the patch latents.** Cheapest, and it targets the mechanism
  directly: quantisation collapses near-identical features onto one code, which
  should drop fine camera detail while keeping coarse content. Measure the ratio
  after quantisation, the rate at which matched grid indices land on the same
  code across views, and sweep codebook size. Report absolute distances too, so
  a real improvement is distinguishable from both numbers collapsing.
- **Scramble the conditioning tokens.** The causal version of this probe. Fix
  camera and objects, inject noise into conditioning tokens, keep the runs that
  fail, inspect which tokens were corrupted. This probe shows the representation
  moves; that one asks whether the policy depends on the part that moves. It
  also separates two hypotheses: if calibration is memorised at specific token
  positions the damage is localised, and if the camera signal is spread across
  all 256 tokens the dependence is diffuse.
- **Latent-space canonicaliser.** A down-projection and up-projection trained
  against the latent from the correct camera angle — what RoVi-Aug and VISTA do
  in pixel space, done in latent space so nothing is re-rendered. Needs paired
  multi-view data, so it is the expensive option.
- **Spatial-question stability.** Not plain captioning: stable captions are
  consistent with everything here, since content survives. Ask spatial questions
  instead ("where is the Joy-Con relative to the shoe last?") to separate
  content from geometric grounding.
- **Measured camera angles** instead of `A`-`F` labels, which would turn the
  ratio into a dose-response curve.

## Reading the results honestly

- This is a **representation probe, not a policy evaluation**. It asks whether
  the token at a given grid index moves more under layout or camera change. It
  does not run a policy, and it does not test optimal cross-view patch matching.
- The bands **overlap at the extremes**: the smallest camera move (adjacent side
  views E→F, 0.31) scores below the largest rearrangement (0.57). The claim
  rests on the group means plus the nearest-neighbour structure, not on
  separation of every pair.
- Distance tracks how much of the *image* changed, and a camera move changes all
  of it. That is the mechanism, not a confound to explain away.
- All-18-view VGGT inference is a **reconstruction setting**, not VLA training.
  Camera identity becomes more explicit there, as expected.
- Absolute distances are not comparable across encoders (different training
  objectives and feature scales); the camera ÷ objects ratio is.
