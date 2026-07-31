# Reference implementations survey — v0.4 encoder redesign

**Date:** 2026-07-30. **Task:** for 10 methods flagged in REDESIGN_RESEARCH_20260729.md, find the official
(or best community) implementation, verify the mechanism-bearing code actually exists in the repo, and
assess portability onto our stack (fork of real-stanford/diffusion_policy + custom PerceiverFuse module,
8 learned queries d=256 over frozen ViT-B/16 patches, robomimic/RoboCasa).

**Method:** every repo below was fetched (repo page, GitHub API metadata/trees, and raw source files for
the mechanism-bearing modules). Claims from raw-file reads are marked; anything not directly fetched is
tagged **UNVERIFIED**. Fetch date 2026-07-30.

## Summary table

| # | Method (paper) | Repo | License | Code status | Mechanism in repo? | Port verdict |
|---|---|---|---|---|---|---|
| 1 | AdapTac (2505.13982) | github.com/kingchou007/adaptac-dex | MIT | Model/policy code present; dataset + some "init code" still TODO | YES — `FFG_policy.py`, `fuse_network.py` | COPY pattern (aux head + force-query attn); encoder is point-cloud, don't port it |
| 2 | FLARE (2505.15659, GR00T N1.5) | none (traces in github.com/NVIDIA/Isaac-GR00T) | Apache-2.0 (GR00T) | **NOT open-sourced** — disabled stubs only | NO — future_tokens embedding exists, loss/MLP absent | BUILD from paper; use FRAPPE code as the working reference |
| 3 | FRAPPE (2602.17259) | github.com/OpenHelix-Team/frappe | **NONE (no license file)** | Official, functional (RDT + RoboTwin) | YES — `flappe_model.py`, `mid_train_runner.py` | REIMPLEMENT (small mechanism; no-license blocks verbatim copy) |
| 4 | MCR (2410.22325) | github.com/luccachiang/robots-pretrain-robots | MIT | Official, complete + HF checkpoints | Losses YES (`mcr/trainer.py`); **metric code ABSENT** | COPY losses; BUILD the Grad-CAM/SAM-2 metric ourselves |
| 5 | SpawnNet (2307.03567) | github.com/johnrso/spawnnet | MIT | Official, complete (sim+real, ckpts) | YES — `simple_bc/encoder/spawnnet/spawnnet.py`, `vit_conv.py` | COPY adapter module; drop the IsaacGym harness |
| 6 | Patch Policy (2607.18236) | github.com/gaoyuezhou/patch_policy | none yet | **PLACEHOLDER** — "Code will be released soon" | NO | BUILD (mechanism trivial); watch repo |
| 7 | DiT-Block Policy (2410.10088) | github.com/sudeepdasari/dit-policy | MIT | Official, complete | YES — `models/diffusion.py` (adaLN-zero), `agent.py` (obs dropout) | COPY directly — cleanest lift of the ten |
| 8 | FACTR (2502.17432) | github.com/RaindragonD/factr | Apache-2.0 | Official (CMU), functional; hardware bits in companion repo | YES — `factr/agent.py` `tokenize_obs()` + `cfg/train_bc.yaml` | COPY — ~50-line curriculum, drop-in |
| 9 | Burns et al. (2312.12444) | github.com/stanford-iris-lab/segmenting_feats + github.com/kayburns/Intriguing-Properties-of-Vision-Transformers | MIT / **none** | Both exist; metric lives in the second | YES — `evaluate_segmentation.py` | RUN as-is on our ViT; adapt for Perceiver queries (no CLS) |
| 10 | Discrete-latent-target aux (2605.04678) | github.com/RUCKBReasoning/From_Pixels_to_Tokens | **NONE (no license file)** | Official (ICML'26), functional, minor TODOs | YES — `exp/train_vla.py` variants + `data_preprocess/*_lam/` | EXTRACT the VQ tokenizer + discrete head idea; harness (Qwen3-VL) irrelevant |

License red flags: FRAPPE, From_Pixels_to_Tokens, and the kayburns metric repo have **no license** —
default all-rights-reserved; reimplement rather than copy verbatim. Everything else MIT/Apache-2.0.

---

## Per-repo notes

### 1. AdapTac — `kingchou007/adaptac-dex` (IROS 2025)

- **URL:** https://github.com/kingchou007/adaptac-dex — MIT. PyTorch; depends on MinkowskiEngine +
  PyTorch3D (RISE-style sparse 3D point-cloud visual encoder), Python 3.8.
- **Mechanism files (verified by raw-file fetch):**
  - `src/adaptac/policy/FFG_policy.py` — class **`FFG`** (Force-Guided Fusion policy).
    `predict_future_force()` computes the aux: `predict_loss, predict_force = self.force_predictor.compute_loss()`,
    weighted into the total as `predict_loss * tactile_weight`. `apply_force_guided_attention()` runs
    `self.force_guided_attn(visual_feat, tactile_feat_reshaped, guide_force_reshaped)` — the *predicted*
    future force is fed back as the attention guide.
  - `src/adaptac/model/fuse_network.py` — **`ForceGuidedCrossAttention`** (predicted force expanded via
    `_expand_force_for_observations()` as the QUERY; `memory = torch.stack([visual, tactile], dim=1)` as
    K/V; implemented as a thin `nn.TransformerDecoderLayer` wrapper), plus `SelfAttentionFusion`,
    `CrossAttentionFusion`. NOTE: the per-modality scalar mixing described in the paper was NOT visible in
    this class on our read — verify before citing that detail (possible paper/code divergence).
  - Decoders: `src/adaptac/model/unet_diffusion.py` (`DiffusionUNetPolicy`, actions) and
    `src/adaptac/model/transformer_diffusion.py` (`DiffusionTransformer` — the future-force predictor is
    itself a small diffusion transformer). Tactile: `src/adaptac/model/tactile/` (MAEGAT GNN over taxels).
- **Completeness:** full file tree fetched; model/policy/dataset/training code present
  (`scripts/command_train.sh`, zarr conversion in `src/adaptac/dataset/gen_data/`). README TODO still
  lists "Release the init code" and "Release the dataset" — checkpoints lean on 3DtacDex.
- **Port assessment:** the two things we want (future-force prediction aux; force-as-query cross-attn
  producing a separate fused stream for the diffusion head) are cleanly isolated in `FFG_policy.py` +
  `fuse_network.py` and are encoder-agnostic — graft onto our diffusion_policy fork with our force window
  replacing their taxel pipeline. Do NOT port the visual side (Minkowski sparse conv over point clouds,
  heavy non-pip dependencies); our frozen-ViT+Perceiver stands in as the K/V source.

### 2. FLARE — no code release (NVIDIA GR00T family)

- **Paper:** https://arxiv.org/abs/2505.15659; project page research.nvidia.com/labs/gear/flare
  (UNVERIFIED — page itself not fetched; URL cited by arXiv/search). arXiv points to
  https://github.com/NVIDIA/Isaac-GR00T as "code".
- **Verified status:** FLARE is **not** in the open GR00T codebase. In tag `n1.5-release`,
  `gr00t/model/action_head/flow_matching_action_head.py` (raw fetch) contains class
  **`FlowmatchingActionHead`** with `self.future_tokens = nn.Embedding(config.num_target_vision_tokens, ...)`
  and the literal comment `return_all_hidden_states=False,  # NOTE (YL): not using flare now`. The
  alignment loss, future-embedding MLP, and target-encoder plumbing are absent. GitHub issues #211
  ("Is FLARE open-sourced at GR00T N1.5") and #215 (asks for release timeline; documents the missing
  pieces) have **no maintainer answer** (both fetched). Current `main` is N1.7 (`gr00t_n1d7/`), Apache-2.0,
  7.7k stars; no flare/flow_matching_action_head hits in the (truncated) main tree — the n1.5-release tag
  is the reference point.
- **Port assessment:** BUILD-not-copy, and that changes little: FLARE's mechanism (a few learnable future
  tokens + cosine alignment of their hidden states to future-frame embeddings) is exactly what FRAPPE
  ships working code for (below). Use the paper for hyperparameters, FRAPPE's repo as the code reference,
  and keep the GR00T `future_tokens` stub as evidence of how NVIDIA slots the tokens into a flow-matching
  action head.

### 3. FRAPPE — `OpenHelix-Team/frappe`

- **URL:** https://github.com/OpenHelix-Team/frappe — official ("Official implementation of FRAPPE"),
  55 stars, created 2026-02, updated 2026-07. **License: NONE** (GitHub API spdx null — treat as
  all-rights-reserved; reimplement, don't paste). Builds on RDT (Robotics Diffusion Transformer) +
  RoboTwin 2.0. 56 .py files; two-stage (mid-train, post-train) entry points
  `main_mid_train.py`/`main_post_train.py`, `finetune_mid.sh`/`finetune_post.sh`.
- **Mechanism files (verified by raw-file fetch):**
  - `models/rdt/flappe_model.py` (sic) — class **`RDT(nn.Module)`**:
    `self.learnable_tokens = nn.Parameter(torch.zeros(1, self.learnable_tokens_num, hidden_size))`
    (std=0.02 init), appended to the sequence via `torch.cat([x, expanded_learnable_tokens], dim=1)`;
    hidden states tapped at a configurable depth: `if encoder_depth is not None and (i + 1) == encoder_depth
    and num_learnable > 0`; projector MLPs (`self.projectors`) map tapped features to teacher spaces.
  - `models/mid_train_runner.py` — **`RDTRunner.compute_loss`** computes the alignment term `proj_loss`:
    L2-normalize then `1 - (z_j * z_tilde_j).sum(dim=-1)` (cosine distance), averaged over
    `len(zs) * bsz`; teacher selection by `enc_type` string matching 'clip' / 'dinov2' / 'theia' / 'vit'
    (encoders in `models/multimodal_encoder/{clip,dinov2,siglip,t5}_encoder.py`). Returned as
    `(loss, proj_loss)`; the λ combination lives in the outer training loop (`train/mid_train.py`,
    UNVERIFIED — not fetched). `models/rdt_runner.py` is the plain-RDT runner (MSE only — verified).
- **Port assessment:** this is the working implementation of our E4 redo (prefix tokens + frozen-VFM
  future-latent cosine at a mid layer). The whole mechanism is ~4 small pieces — token parameter, one-line
  sequence concat, depth tap, projector+cosine — trivially reimplementable inside a diffusion_policy
  transformer/DiT in a day; RDT specifics don't matter. The missing license is the only reason not to
  copy files wholesale.

### 4. MCR — `luccachiang/robots-pretrain-robots` (ICLR 2025)

- **URL:** https://github.com/luccachiang/robots-pretrain-robots — official, MIT, PyTorch,
  ResNet-50 backbone (`torchvision`). Checkpoints on HuggingFace `GqJiang/robots-pretrain-robots`.
- **Mechanism files (verified by raw-file fetch of `mcr/trainer.py`):** class **`Trainer.update()`**
  computes all three losses: time-contrastive (`smoothloss1/2`, InfoNCE over within-video frame triplets),
  action prediction (`bc_loss = model.module.bc_loss(pred_action, b_actions.detach())`), and the
  dynamics-alignment contrastive (`state_align_loss` via `s0loss`/`s2loss`, InfoNCE between image
  embeddings and a state-action encoder). Architecture in `mcr/models/models_mcr.py` (backbone, actor,
  projectors); config `cfgs/config_rep.yaml`; DROID preprocessing in `scripts/` (user-path TODOs).
- **The manipulation-centricity metric (Grad-CAM ∩ SAM-2 Jaccard) is NOT in the repo** — confirmed by
  README/file inspection. If we want the diagnostic (redesign doc §3/§5), we build it: binarized Grad-CAM
  vs SAM-2 masks of EE+objects, threshold unspecified in the paper.
- **Port assessment:** the losses are encoder-pretraining objectives, orthogonal to the policy — portable
  onto Perceiver pretraining wherever we have paired (frames, proprio, action) batches; the
  dynamics-alignment InfoNCE is the piece the ablation says to lift (83.2→66.2 without it). Written for a
  single pooled ResNet embedding, so we adapt the projector heads to our token set (e.g., align a pooled
  readout or per-query projections). Metric = build ourselves (SAM-2 + Grad-CAM, both off-the-shelf).

### 5. SpawnNet — `johnrso/spawnnet` (ICRA 2024)

- **URL:** https://github.com/johnrso/spawnnet — official (linked from project page
  xingyu-lin.github.io/spawnnet; authors Lin, So, Mahalingam, Liu, Abbeel). MIT. PyTorch + Hydra;
  sim experiments use IsaacGym/RLAfford (vendored), real experiments a simple BC harness.
- **Mechanism files (directory listing verified; file internals UNVERIFIED — not read line-by-line):**
  - `simple_bc/encoder/spawnnet/spawnnet.py` — the SpawnNet encoder (two-stream: frozen pretrained ViT
    layers tapped multi-layer, adapter convs fused into a learned stream).
  - `simple_bc/encoder/spawnnet/vit_conv.py` — ViT feature extraction / conv-adapter plumbing.
  - Baselines for the Table-3 mechanism comparison: `simple_bc/encoder/{r3m_encoder,impala,vit_descriptor}.py`.
  - Training/eval: `simple_bc/train.py`, `simple_bc/eval.py`, Hydra confs in `conf/`.
- **Completeness:** production-grade for reproduction — sim checkpoints, real dataset (Drive), debug docs.
- **Port assessment:** the adapter-insertion pattern (tap frozen ViT-B/16 at multiple depths, lightweight
  conv adapters, fuse into a trainable stream) drops straight into our stack as an alternative/addition to
  PerceiverFuse — our Perceiver can consume the adapter pyramid instead of last-layer patches, which is
  precisely the "multi-layer, not last-layer" rule the redesign doc extracted. `simple_bc/encoder/` is
  reasonably self-contained; leave IsaacGym/RLAfford behind.

### 6. Patch Policy — `gaoyuezhou/patch_policy` (arXiv 2607.18236, LeCun/Pinto NYU)

- **URL:** https://github.com/gaoyuezhou/patch_policy — linked from https://patch-policy.github.io/.
  **Placeholder only:** README says "Code will be released soon. Stay tuned!" No license, no description
  (API: null/null), created 2026-07-21, updated 2026-07-29 — repo is 9 days old and being touched, so a
  release may be imminent. 13 MB size is likely media, not code (UNVERIFIED).
- **Port assessment:** no code exists → BUILD. The mechanism is the simplest of the ten (frozen
  DINOv2/WebSSL patch tokens fed densely to a block-causal transformer policy); the paper + our existing
  diffusion_policy transformer suffice. Re-check the repo before implementation freeze — if it lands with
  a permissive license their block-causal masking and token-handling details are worth a diff.

### 7. DiT-Block Policy — `sudeepdasari/dit-policy` (arXiv 2410.10088)

- **URL:** https://github.com/sudeepdasari/dit-policy — official, MIT, PyTorch + Hydra
  (`data4robotics` codebase). Entry point `finetune.py`.
- **Mechanism files (verified by raw-file fetch):**
  - `data4robotics/models/diffusion.py` — the DiT noise net: **`_DiTDecoder`** blocks with
    `_ShiftScaleMod` + **`_ZeroScaleMod`** (adaLN-zero: `nn.init.zeros_(self.scale.weight)` and
    `nn.init.zeros_(self.scale.bias)`), **`_FinalLayer`** with
    `adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(..., 2*hidden_size))`.
  - `data4robotics/agent.py` — **`BaseAgent`**: proprio enters via `use_obs` ∈ {"add_token",
    "pad_img_tokens"}, in both paths through `nn.Dropout(p=0.2)` applied to the raw obs vector
    (`nn.Dropout(p=0.2), nn.Linear(odim, token_dim)` / `self._obs_proc = nn.Dropout(p=0.2)`). Since
    `nn.Dropout` zeroes elements independently, this IS the paper's per-dimension proprio dropout —
    hardcoded p=0.2, no dedicated flag. (Our reading of fetched source; the "per-dim" label is the
    paper's, the code is a plain elementwise Dropout on the state vector.)
  - Per-camera ResNet tokenizer: `data4robotics/models/resnet.py`; trainers in `data4robotics/trainers/bc.py`.
- **Port assessment:** cleanest lift of the survey. adaLN-zero block classes are self-contained
  `nn.Module`s and MIT — copy into our fork for the E5 "zero-init gated insertion" arm; the proprio
  dropout is one line in our obs encoder. Note our diffusion_policy fork's `TransformerForDiffusion`
  differs structurally from their `_DiTDecoder`; copying the two Mod classes + wiring is easier than
  adopting their whole model file.

### 8. FACTR — `RaindragonD/factr` (RSS 2025, CMU)

- **URL:** https://github.com/RaindragonD/factr — official (linked from jasonjzliu.com/factr),
  Apache-2.0. Skeleton is visibly forked from dit-policy/data4robotics (same `agent.py`,
  `models/action_transformer.py`, `trainers/` layout). Companion `factr_teleop` repo (ROS2) holds the
  force-feedback teleop; policy repo is hardware-light.
- **Mechanism files (verified by raw-file fetch):**
  - `factr/agent.py` — **`BaseAgent.tokenize_obs()`** applies the curriculum corruption: pixel space
    `gaussian_2d_smoothing(v, scale)` or `downsample_2d(...)`, latent space `gaussian_1d_smoothing` /
    `downsample_1d` on tokens; intensity from `get_scale(scheduler=..., start=curriculum.start_scale,
    end=curriculum.stop_scale, cur_step=misc.GLOBAL_STEP, max_step=curriculum.max_step)` — i.e., the
    decaying-corruption curriculum is a per-step scale on an image/token smoothing op.
  - `factr/cfg/train_bc.yaml` — `curriculum:` block: `space: pixel|latent`, `operator: blur|downsample`,
    `scheduler: linear` (options no/const/linear/step/exp/cos), `start_scale: 5`, `stop_scale: 0`,
    `max_step: ${max_iterations}`.
  - Force/proprio enters as obs tokens through the same `add_token` path (dropout p=0.2 inherited from
    data4robotics); no exotic force encoder in this repo. Trainer: `factr/train_bc_policy.py`,
    `factr/trainers/bc.py` (plain BC — verified no curriculum logic there; it all lives in the agent).
- **Port assessment:** the entire mechanism is ~50 lines (2 corruption ops + a scale scheduler + a global
  step) sitting in the obs-tokenization path — drop-in for our encoder wrapper on the E6 force arm, fully
  orthogonal to Perceiver/diffusion_policy internals, Apache-licensed. Note a FACTR 2 paper exists
  (arXiv 2606.12406, external force sensing for commodity arms) — not surveyed here.

### 9. Burns et al. attention-Jaccard — two repos

- **Policy/eval harness:** https://github.com/stanford-iris-lab/segmenting_feats — MIT, Stanford IRIS.
  Built on the facebookresearch R3M eval codebase; policy training + kitchen-shift transfer tests under
  `evaluation/r3meval/core/`, `run.sh`/`eval.sh`; project page points at the `eval` branch.
- **The metric itself:** https://github.com/kayburns/Intriguing-Properties-of-Vision-Transformers —
  kayburns' copy of the IP-ViT codebase (GitHub API says `fork: false`, so it's a re-upload, not a fork
  object; last updated 2023-09). **`evaluate_segmentation.py`** (+ `evaluate_segmentation.sh`) computes
  the Jaccard index of binarized ViT attention maps vs PASCAL VOC masks — exactly the R²=0.85 predictor.
  The project page (kayburns.github.io/segmentingfeatures) documents how to add new models to this
  script. **No license** on this repo (and upstream IP-ViT license UNVERIFIED).
- **Port assessment:** run, don't port — it's a standalone diagnostic needing only PASCAL VOC + a model
  hook; wire our frozen ViT-B/16 (and any finetuned variants) into `evaluate_segmentation.py` per the
  project-page instructions. One real adaptation for us: the metric is defined on CLS→patch attention
  (last block, all heads); PerceiverFuse has no CLS, so scoring the *encoder output* means substituting
  query→patch cross-attention maps — that variant is our extrapolation, not Burns'.

### 10. Discrete-latent-target aux — `RUCKBReasoning/From_Pixels_to_Tokens` (arXiv 2605.04678, ICML 2026)

- **URL:** https://github.com/RUCKBReasoning/From_Pixels_to_Tokens — official, 37 stars, created
  2026-03-31. **License: NONE** (API spdx null) — reimplement, don't copy. Stack: Qwen3-VL-2B VLA
  backbone, PyTorch 2.8, RLDS data, LIBERO eval (`experiments/robot/libero/`).
- **Mechanism files (directory/README verified; file internals UNVERIFIED):**
  - `exp/train_vla.py` with `--vla_id ∈ {baseline, la_align, la_direct, la_cond, la_tok}` — all four
    supervision variants from the paper (LA-Direct = discrete latent tokens decoded directly; LA-Align =
    internal-representation alignment; LA-Cond; LA-Tok), implementations under `latentvla/models/vla/`.
  - `data_preprocess/image_based_lam/` — trains the image-based latent-action model (the discrete-token
    target source); `data_preprocess/action_based_lam/` — VQ-style action tokenizer.
- **Completeness:** functional per README (LIBERO support recent); placeholder paths in preprocessing,
  robot constants need manual config.
- **Port assessment:** what transfers to E4-redux is the TARGET FORMAT, not the harness: train a small
  VQ tokenizer over future latents (their `*_lam` recipes) and supervise prefix tokens with cross-entropy
  on discrete codes instead of continuous cosine. The VLA side (Qwen3-VL, autoregressive decoding) shares
  nothing with diffusion_policy — extract the tokenizer training + discrete-supervision head as ideas,
  reimplement (~small VQ-VAE + CE loss) inside our FRAPPE-style aux.

---

## Cross-cutting build-vs-copy summary for our stack

1. **Copy with license clean (MIT/Apache):** DiT-Block adaLN-zero mods + obs dropout (dit-policy),
   FACTR curriculum (~50 lines), SpawnNet encoder module, MCR loss functions, AdapTac aux-head +
   force-query-attention pattern, Burns eval scripts (metric repo license murky — the script is a thin
   layer over VOC eval; worst case rewrite it).
2. **Reimplement from working reference (no license):** FRAPPE prefix-token alignment (the E4 redo core —
   mechanism is 4 small pieces), From_Pixels_to_Tokens discrete-target recipe.
3. **Build from paper (no code exists):** FLARE (use FRAPPE as code proxy; GR00T stub confirms token
   slotting), Patch Policy (trivial mechanism; watch the 9-day-old placeholder repo), MCR
   manipulation-centricity metric (SAM-2 + Grad-CAM, both off-the-shelf).
4. **Biggest practical wins first:** FRAPPE mechanism + dit-policy adaLN-zero cover the two
   highest-ranked redesign changes (E4 prefix-aux, E5 insertion) with the least code; FACTR curriculum
   and the AdapTac aux head cover the E6 force arm; Burns script is a same-day diagnostic on existing
   checkpoints.
