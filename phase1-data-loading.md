# Phase 1 — Data Loading + LeJEPA (video-only) on RH20T cfg3

**Goal:** prove the data pipeline works and LeJEPA trains on RGB **before adding any other
modality**. If this is green (loss converges, no collapse, linear probe is sane), the dataloader
and encoder are de-risked and we move to Phase 2 (+ robot state).

> Rule: don't add a modality until the previous setup trains clean. video → +robot state → +decoder.

Data location (VM): `/mnt/nas/data/RH20T/RH20T_cfg3.tar.gz` (UR5 + WSG-50, fixed rig).
For Phase 1 we only need **RGB frames on disk** — no alignment/multimodal loader yet.

---

## 0. Environment

```bash
# repos
git clone https://github.com/rh20t/rh20t_api.git
git clone https://github.com/galilai-group/stable-pretraining.git   # LeJEPA + 30 SSL methods
# (or the minimal LeJEPA repo: github.com/galilai-group/lejepa)

# deps
pip install -r rh20t_api/requirements_api.txt
pip install torch torchvision timm pillow
pip install -e stable-pretraining   # ⚠️ verify install command in its README
```

---

## 1. Extract + sanity-check the data (do this FIRST, before any model)

```bash
mkdir -p /mnt/nas/data/RH20T/cfg3 && \
tar -xzf /mnt/nas/data/RH20T/RH20T_cfg3.tar.gz -C /mnt/nas/data/RH20T/cfg3

# RGB is stored as .mp4 -> convert to frames (multiprocessing script)
python -m rh20t_api.extract --help        # ⚠️ confirm exact flags
# typical: point it at the extracted scene root; it writes <scene>/cam_*/color/*.jpg
```

**Eyeball one episode before trusting anything** (this is the cheapest de-risk):

```python
from rh20t_api.configurations import load_conf
from rh20t_api.scene import RH20TScene

robot_configs = load_conf("rh20t_api/configs/configs.json")
scene = RH20TScene("/mnt/nas/data/RH20T/cfg3/<one_scene_folder>", robot_configs)

# pull a few aligned readings at a timestamp to confirm the API works
ts = ...  # ⚠️ get a valid timestamp (e.g. from a color frame filename, which is the ts in ms)
print(scene.get_image_path_pairs(ts, image_types=["color"]))  # RGB path(s)
print(scene.get_tcp_aligned(ts))            # 7D pose (xyz + quaternion)
print(scene.get_joint_angles_aligned(ts))   # joint angles
print(scene.get_ft_aligned(ts))             # 6D force-torque
print(scene.get_audio_path())               # audio file
```

**Confirm sync is real:** plot the force-torque magnitude over an episode and check the spike
lines up with the video frame where contact happens. If they match, alignment is correct — this
is the single most important data check.

✅ **Gate:** you can open frames and the getters return sane values. Don't proceed until true.

---

## 2. Minimal video-only Dataset (chunked RGB → LeJEPA views)

Phase 1 needs nothing but extracted frames. LeJEPA wants `[B, V, 3, 128, 128]` — `V` augmented
**views** per sample. (cfg3 RGB is 640×360; we crop/resize to 128.)

```python
# phase1_video_dataset.py
import glob, os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class RH20TVideoFrames(Dataset):
    """Video-only. One RGB frame -> V augmented views (LeJEPA-native)."""
    def __init__(self, frames_root, img_size=128, n_views=4):
        # frames_root: output of rh20t extract.py, e.g. .../cfg3/**/color/*.jpg
        self.paths = sorted(glob.glob(os.path.join(frames_root, "**", "*.jpg"), recursive=True))
        assert self.paths, f"no frames under {frames_root}"
        self.n_views = n_views
        self.aug = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.4, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.4, 0.4, 0.2, 0.1),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        views = torch.stack([self.aug(img) for _ in range(self.n_views)])  # [V,3,H,W]
        return views
```

> Variant (video-native, later): instead of augmentation views, use **temporally adjacent
> frames** as the views — that turns LeJEPA's invariance into temporal prediction. Keep the
> augmentation version for Phase 1; it's the simplest thing that validates the data.

---

## 3. Train LeJEPA (video-only)

**Recommended — use `stable-pretraining`** (it implements LeJEPA + the SIGReg loss + a Lightning
trainer, so you only supply the Dataset + a backbone). Plug `RH20TVideoFrames` into its LeJEPA
LightningModule; pick a `timm` ViT backbone. ⚠️ Confirm the exact module/class name in
`stable-pretraining/METHODS.md` (`spt`/`stable_pretraining.methods.lejepa`).

**Fallback — standalone loop** (loss/encoder copied verbatim from LeJEPA `MINIMAL.md`; do **not**
reimplement the loss — import `SIGReg`/`MLP` from the repo):

```python
# phase1_train.py
import torch, timm, torch.nn as nn
from torch.utils.data import DataLoader
from phase1_video_dataset import RH20TVideoFrames
from lejepa import SIGReg, MLP   # ⚠️ confirm import path (copy from MINIMAL.md if needed)

class ViTEncoder(nn.Module):
    def __init__(self, proj_dim=16):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch8_224", pretrained=False,
            num_classes=512, drop_path_rate=0.1, img_size=128)
        self.proj = MLP(512, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)
    def forward(self, x):                 # x: [B, V, 3, 128, 128]
        B, V = x.shape[:2]
        emb = self.backbone(x.flatten(0, 1))      # [B*V, 512]
        proj = self.proj(emb)                     # [B*V, proj_dim]
        return emb.view(B, V, -1), proj.view(B, V, -1)

# recommended launch hyperparams (from MINIMAL.md): lamb=0.02 V=4 proj_dim=16 lr=2e-3 bs=256 epochs=800
lamb = 0.02
ds = RH20TVideoFrames("/mnt/nas/data/RH20T/cfg3", img_size=128, n_views=4)
dl = DataLoader(ds, batch_size=256, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)

net = ViTEncoder(proj_dim=16).cuda()
sigreg = SIGReg().cuda()
opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=0.05)

for epoch in range(800):
    for views in dl:
        views = views.cuda(non_blocking=True)              # [B,V,3,128,128]
        emb, proj = net(views)
        # LeJEPA loss — keep this EXACTLY as in MINIMAL.md:
        inv_loss = (proj.mean(0) - proj).square().mean()   # prediction/invariance (MSE)
        sigreg_loss = sigreg(proj.flatten(0, 1))           # anti-collapse regularizer
        loss = sigreg_loss * lamb + inv_loss * (1 - lamb)
        opt.zero_grad(); loss.backward(); opt.step()
```

Start tiny: **1 scene, a few hundred frames, 5 epochs** to confirm it runs end-to-end, then scale.

---

## 4. Validation — "did the data work?"

No decoder yet, so judge the encoder by:
1. **Loss curves** — `inv_loss` drops, `sigreg_loss` stays bounded (no collapse). If `sigreg`
   blows up or `inv_loss → 0` instantly, something's wrong (degenerate views / bad data).
2. **No collapse** — embedding std across the batch stays > 0 (SIGReg's whole job).
3. **Linear probe** — freeze the encoder, train a linear head on the embedding to predict
   **task id** (147 tasks) or a **contact/no-contact** label (threshold F/T magnitude). A
   non-trivial accuracy means the latent carries real signal → data + encoder are good.

✅ **Phase-1 done when:** trains without collapse + linear probe beats chance. This video-only
encoder is also your **vision-only baseline** for the later joint-vs-independent experiment.

---

## 5. Gotchas checklist
- [ ] Frames actually extracted (count them) before training.
- [ ] F/T spike ↔ video contact alignment verified on ≥1 episode.
- [ ] Handle episodes with missing/short streams (skip or pad).
- [ ] Image size: cfg3 is 640×360 → crop/resize to 128 (matches LeJEPA).
- [ ] `proj_dim` small (16) per LeJEPA's recommendation — not the embedding dim.
- [ ] Lock chunking/rate handling in the loader now; you'll reuse it for Phase 2.

## Next (don't start until Phase 1 is green)
- **Phase 2:** + robot state — switch to the API's aligned getters
  (`get_tcp_aligned`, `get_joint_angles_aligned`, `get_ft_aligned`) to emit
  `{rgb, joints, torque, tcp, gripper, ft}` per timestamp; add the robot-state branch + L2/L3.
- **Phase 3:** + PixNerd decoder (encoder frozen first → watch the freeze-ceiling).
