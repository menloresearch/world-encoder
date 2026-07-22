# Kepler — world encoder

A self-supervised world encoder for robots. Vision, proprioception, and force/torque
are fused into a single 256-d latent by 8 learned queries (Perceiver-style bottleneck),
trained JEPA-style: hide one sense, predict its embedding from the others. Trained on
RH20T (~54M frames, 4 embodiments). One encoder transfers across robot bodies, and the
vision-only latent linearly reads out motor / end-effector / force state — signals raw
pixels alone don't carry ("sensory entanglement").

- **Paper:** https://arxiv.org/abs/2607.13522
- **Blog:** https://menlo.ai/research/kepler-v01
- **Docs:** all research docs — plan, experiment journals, live run docs (RH20T temporal,
  RoboCasa N1), results, paper source — on the [`docs` branch](../../tree/docs)

## Code layout

- `world_tokenizer/` — encoder, JEPA pretraining, probes, predictors
- `preprocessing/` + `run_*.sh` — data prep and run entrypoints
- `metrics/`, `visualizer/`, `scripts/`, `splits/`
