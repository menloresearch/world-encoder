# multicam follow-ups + downstream (2026-07-24)

@freelerobot @Yip-Jia-Qi 


**1. v0.2 encode-single-view then average**

```
                                ee R²    motor R²
mc4 full 4-view                 0.544    0.400
encode-per-view, averaged       0.425    0.346
best single view (cam3)         0.478    0.325
worst single view (cam2)        0.218    0.188
```

(ee/motor here are from the per-cam probe harness — compare within these tables, not to the pose numbers in the earlier comment)

average loses to the best single view → the averaging step loses the information, not the backbone. same ordering as the v0.1 late-fusion baseline. geometrically the average is closest to the full embedding (0.86× vs 1.22–1.78× for singletons) yet probes worse — embedding distance does not track information.

**2. camera dropout — doesn't work**

expectation was less convergence pattern, world representation stays constant regardless of view. measured the opposite:

```
                              no-dropout    dropout
mc4 ee / motor                0.544/0.400   0.461/0.356
mc1 ee                        0.292         0.327
RankMe                        153           113
cross-instant distance        0.320         0.065
d(1-view, full) / cross-inst  1.49          2.04
```

full-view quality drops, single-view gains only +0.035 (still << v0.1's 0.482), and the whole space contracts 5× → *relative* invariance gets worse. dropout shrinks the space instead of making subsets converge.

this also closes the OOD caveat on #1: reran encode+average on the dropout ckpt (single views now in-distribution) — average still flat (0.421), still below full fusion (0.461). #1's ordering is not an OOD artifact.

**decision:** no dropout. train with all views, fuse early. single-view robustness → v0.1 per view. if we need invariance → explicit subset→full consistency loss (dropout only pressures it implicitly).

**3. per-camera breakdown**

no wrist cam in this set — the precompute excludes in-hand serials, all 4 are external views. among them:

```
d(singleton, full) / cross-instant:  cam0 1.35 | cam1 1.59 | cam2 1.78 | cam3 1.22
```

real spread, ordering tracks single-view probe R². but even the closest view is farther from full than a different moment (1.22 > 1.0) → no single view carries most of the information; the fusion is compositional. testing global-vs-wrist needs a wrist-cam precompute.

**4. downstream encoder swap (latent as the policy's only vision: fails)**

the runs from the last update. RoboCasa PickPlaceCounterToCabinet, ~500 demos, 3 cams, diffusion policy, only the encoder differs, 50 rollouts/ckpt (→ ±6-7% noise):

```
success @         ep50   ep100   ep150
baseline          24%    28%     32%
kepler e2e         0%    12%     14%
kepler pt-enc      4%     2%      2%
```

same on OpenDrawer: pt-enc 22% vs baseline 74%. pretraining itself worked in-domain (state R² 0.673 vs raw 0.538) → the failure is architectural, not the init. probes show why:

```
action-readout R² (predict the commanded action seq from vision)
proprio only           0.176
random-init fuse       0.287
pretrained fuse        0.335   (q32/q64 same; pooled = set)
raw frozen patches     0.413
```

the raw patches hold more task information than our fused latent. the state-prediction objective gives no pressure to retain object detail — same failure mode as the averaging in #1.

**5. new row: hybrid — no lift**

since replacement failed, added the row FLARE's evidence actually supports (their latent also sits NEXT TO the visual stream, not instead of it). hybrid = baseline policy with its stock image encoder unchanged + the frozen pretrained encoder from row 4 (pt-enc weights) feeding its 256-d latent, concatenated to the image features before the policy head. tests whether the latent ADDS anything:

```
success @        ep50   ep100   ep150
baseline s0       24%    28%     32%
baseline s1       26%    40%*    34%
hybrid s0         22%    24%     26%
hybrid s1         32%*   24%     24%
                  * single-point outliers
```

hybrid flat at 24-26% while baseline reaches 32-34% by ep150, both seeds → the latent adds nothing the policy's own encoder doesn't already get. curves run to ep400 for completeness; conclusion unlikely to change.

**downstream conclusion:** the latent can't replace a policy's vision and adds nothing beside it. the limitation is what the pretraining objective retains, not where the latent is inserted. next: add object/scene pressure to the objective (patch recon / DINO-style distillation into the fuse), then rerun the hybrid row.

**6. predictor replicates on robocasa**

reran the next-embedding predictor on robocasa (linear and MLP predictors on the frozen in-domain encoder's latents; linear performed best). same result as RH20T:

```
horizon (s)              0.25    0.5    0.75
error vs carry-forward   0.89    0.85   0.81    (= 11-19% cut)
```

the predictor result holds in a second domain (sim, different encoder, different robot) — not an RH20T artifact.

**7. action conditioning**

robocasa logs the commanded 12-D actions, so added the command sequence over the gap to the predictor above:

```
horizon (s)              0.25     0.5     0.75
with actions / without   0.996    0.989   0.984   (= 0.4-1.6% gain)
```

actions add almost nothing. reason: in demo data the action is endogenous — the operator reacts to the scene, so the command is predictable from the state the latent already encodes. consequence: demos can't validate an action-conditioned world model; needs action-diverse data (perturbations/exploration) or action-sensitivity evals. the gain does grow with horizon (0.4→1.6%) → worth revisiting at longer gaps.



