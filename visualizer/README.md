# visualizer

A small local website for browsing the RH20T training data and, once runs finish, the
metrics. Stdlib-only Python server + vanilla HTML/JS — nothing to install.

## Run

```bash
python visualizer/server.py --port 8000
# then open http://127.0.0.1:8000
```

The frames root resolves in order: `--frames-root`, `--data-root`/frames, `$RH20T`/frames,
`/mnt/nas/data/RH20T/frames` — the same layout the pipeline writes
(`frames/<cfg>/<scene>/<cam>/<stream>/<timestamp_ms>.jpg`).

To view it from another machine, either bind `--host 0.0.0.0` or (safer) tunnel:
`ssh -L 8000:127.0.0.1:8000 menlo`.

## Training data tab

cfg → scene (searchable, 800 in cfg3) → camera. Frames play back at recorded timestamps
(0.5–4× speed), with a scrubber and arrow-key stepping. Frames are served with immutable
cache headers, so scrubbing back is instant.

Below the player, the **robot state** is plotted exactly as the dataloader sees it: the
server imports `world_tokenizer.state.SceneState` (so preprocessing can never drift from
training) and evaluates the 28-dim vector at the selected camera's frame timestamps —
joints→sin/cos, tcp pos→symlog, quat→6D, F/T zeroed→symlog, gripper width→symlog. A
playhead tracks video playback and clicking any chart seeks the video. Needs numpy+scipy
on the server's python and raw scenes under `--raw-root` (default `<data-root>/raw`,
accepting either `raw/cfg3/...` or `raw/RH20T_cfg3/...`).

## Metrics tab

Renders every `*.json` in `--metrics-dir` (default `visualizer/metrics/`), one card per
file. Write a file from any eval script with `json.dump` — no coupling to this code:

```python
json.dump({
    "run": "phase1 cfg3 lr2e-5",                     # card title (optional, falls back to filename)
    "note": "anything worth remembering",            # optional
    "scalars": {"RankMe": 285, "force R²": 0.01},    # -> stat tiles
    "charts": [{                                     # -> line charts (crosshair + tooltip)
        "title": "contact accuracy vs checkpoint",
        "x_label": "epoch", "x": [0, 3, 6, 10],
        "series": {"linear": [0.81, 0.82, 0.81, 0.82], "kNN": [0.79, 0.80, 0.80, 0.79]},
    }],
    "bars": [{                                       # -> bar chart with error bars
        "title": "triplet accuracy by tier",
        "labels": ["tier 1", "tier 2", "tier 3", "tier 5"],
        "values": [0.71, 0.88, 0.99, 0.75],
        "errors": [0.02, 0.01, 0.01, 0.02],          # optional
    }],
    "tables": [{"title": "...", "columns": [...], "rows": [[...], ...]}],
}, open("visualizer/metrics/phase1_cfg3.json", "w"), indent=2)
```

All keys are optional; unknown keys are ignored. `metrics/sample_run.json` is a worked
example (Stage-2 numbers from EXPERIMENTS.md) — delete it once real runs write here.

For a long training run, overwrite the same file each epoch and reload the page; the
card shows the file's mtime.

## Layout

- `server.py` — HTTP server: static files, JSON API (`/api/summary`, `/api/scenes`,
  `/api/scene`, `/api/frames`, `/api/metrics`), and frame images (`/frames/...`).
- `static/` — the site (`index.html`, `app.js`, `style.css`). Charts are hand-rolled SVG.
- `metrics/` — drop metrics JSON here.

Not covered yet (future work): robot-state / F-T overlays next to the video (needs
`rh20t_api` to read the raw scene), and live tailing of an in-progress run.
