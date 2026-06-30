# Phase 1 (finalized) — Data Loading + LeJEPA fine-tune (video-only), RH20T cfg3

**Goal:** prove the data pipeline works and the encoder learns on RGB **before adding any other
modality**. Green = loss converges, no collapse, linear probe beats chance → move to Phase 2.

**Setup assumed:**
- Data on the VM: `/mnt/nas/data/RH20T/RH20T_cfg3.tar.gz` (UR5 + WSG-50, fixed rig).
- Compute: 8× RTX PRO 6000 (96 GB each). Way more than Phase 1 needs — **debug on 1 GPU, then DDP on 8.**
- Encoder init: **warm-start** from `OK-AI/lejepa-vitb16-pretrain-in1k` (LeJEPA ViT-B/16, ImageNet-1k,
  DINOv2-style, 768-dim, Apache-2.0) and **continue LeJEPA** on cfg3. Do **not** freeze.

> Rule: don't add a modality until the previous setup trains clean. video → +robot state → +decoder.

---

## 0. Environment

```bash
git clone https://github.com/rh20t/rh20t_api.git
git clone https://github.com/galilai-group/stable-pretraining.git    # LeJEPA + trainer + multi-crop
pip install -r rh20t_api/requirements_api.txt
pip install torch torchvision timm pillow transformers safetensors
pip install -e stable-pretraining        # ⚠️ confirm install cmd in its README
```

---

## 1. Stage data on local NVMe + extract frames + eyeball  ⟵ DO THIS FIRST

IO from the NAS will starve 8 GPUs — **extract frames to fast local disk**, not the NAS mount.

```bash
LOCAL=/local/rh20t/cfg3            # fast NVMe, NOT the NAS
mkdir -p "$LOCAL"
tar -xzf /mnt/nas/data/RH20T/RH20T_cfg3.tar.gz -C "$LOCAL"

# RGB is stored as .mp4 -> frames (timestamped .jpg). multiprocessing script:
python -m rh20t_api.extract --help          # ⚠️ confirm exact flags, then run it
# expected output: $LOCAL/<scene>/cam_*/color/<timestamp>.jpg
```

**Eyeball one episode (cheapest, most important de-risk):**

```python
from rh20t_api.configurations import load_conf
from rh20t_api.scene import RH20TScene

cfgs = load_conf("rh20t_api/configs/configs.json")
scene = RH20TScene("/local/rh20t/cfg3/<one_scene>", cfgs)
ts = ...   # ⚠️ a valid timestamp (color frame filenames are the ts in ms)
print(scene.get_image_path_pairs(ts, image_types=["color"]))
print(scene.get_tcp_aligned(ts), scene.get_joint_angles_aligned(ts), scene.get_ft_aligned(ts))
```

**Confirm sync:** plot force-torque magnitude over an episode; the spike must line up with the
video frame where contact happens. If it does, alignment is right — this is the key check.

✅ **Gate:** frames open, getters return sane values, F/T↔video aligned. Don't proceed until true.

---

## 2. Multi-crop video-only `Dataset`

The checkpoint was trained with DINOv2-style multi-crop (**2× 224 global + N× 96 local views**),
ImageNet normalization. Match it. (Set `n_local=0` for the very first debug run to keep it simple.)

```python
# phase1_dataset.py
import glob, os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # in1k stats

class MultiCropRGB(Dataset):
    """Video-only. Each RGB frame -> global + local crops (LeJEPA views)."""
    def __init__(self, frames_root, n_global=2, n_local=6, gsize=224, lsize=96):
        self.paths = sorted(glob.glob(os.path.join(frames_root, "**", "*.jpg"), recursive=True))
        assert self.paths, f"no frames under {frames_root}"
        self.n_global, self.n_local = n_global, n_local
        self.g = T.Compose([T.RandomResizedCrop(gsize, scale=(0.4, 1.0)),
                            T.RandomHorizontalFlip(), T.ColorJitter(0.4, 0.4, 0.2, 0.1),
                            T.ToTensor(), _NORM])
        self.l = T.Compose([T.RandomResizedCrop(lsize, scale=(0.05, 0.4)),
                            T.RandomHorizontalFlip(), T.ColorJitter(0.4, 0.4, 0.2, 0.1),
                            T.ToTensor(), _NORM])
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        g = [self.g(img) for _ in range(self.n_global)]
        l = [self.l(img) for _ in range(self.n_local)]
        return g, l

def collate(batch):                       # crops differ in size -> group by type
    gs = torch.stack([torch.stack(g) for g, _ in batch])   # [B, n_global, 3, 224, 224]
    ls = torch.stack([torch.stack(l) for _, l in batch])   # [B, n_local , 3,  96,  96]
    return gs, ls
```

---

## 3. Model — warm-start the LeJEPA ViT-B/16

```python
# phase1_model.py
import torch.nn as nn
from transformers import AutoModel

class LeJEPANet(nn.Module):
    def __init__(self, proj_dim=16, embed_dim=768):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            "OK-AI/lejepa-vitb16-pretrain-in1k", trust_remote_code=True)  # ⚠️ confirm trust_remote_code
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, 2048), nn.BatchNorm1d(2048), nn.GELU(),
            nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.GELU(),
            nn.Linear(2048, proj_dim))

    def forward(self, x):                       # x: [N, 3, H, W]
        out = self.backbone(x)                  # ⚠️ confirm call: backbone(x) vs backbone(pixel_values=x)
        cls = out.cls_tokens                    # ⚠️ confirm attr: cls_tokens / pooler_output / out[0]
        return cls, self.proj(cls)
```

> **One-time check:** in a REPL, run `out = model.backbone(torch.randn(2,3,224,224).cuda())` and
> `print(type(out), out.keys() if hasattr(out,'keys') else dir(out))` to lock the exact output
> attribute, then fix line `cls = ...` accordingly.

---

## 4. Train

### Option A (recommended) — `stable-pretraining` + Lightning DDP
It implements the LeJEPA loss + multi-crop + the trainer; you only supply the backbone + Dataset.
- Wrap `LeJEPANet.backbone` as the SSL backbone, point its LeJEPA method at `MultiCropRGB`.
- ⚠️ Confirm the exact class in `stable-pretraining/METHODS.md` (`stable_pretraining.methods.lejepa`).
- Multi-GPU + precision: `Trainer(devices=8, strategy="ddp", precision="bf16-mixed", max_epochs=30)`.

### Option B (transparent fallback) — standalone loop, global views only
Simplest debuggable path; loss lines copied from LeJEPA `MINIMAL.md` (don't reimplement SIGReg —
import it). Use `n_local=0` so all views are 224.

```python
# phase1_train.py   (single GPU first; DDP notes in §6)
import torch
from torch.utils.data import DataLoader
from phase1_dataset import MultiCropRGB, collate
from phase1_model import LeJEPANet
from lejepa import SIGReg                  # ⚠️ confirm import path (or copy from MINIMAL.md)

LAMB = 0.02                                # SIGReg weight (LeJEPA default)
ds = MultiCropRGB("/local/rh20t/cfg3", n_global=2, n_local=0)   # global-only for first run
dl = DataLoader(ds, batch_size=128, shuffle=True, num_workers=24,
                pin_memory=True, persistent_workers=True, drop_last=True, collate_fn=collate)

net = LeJEPANet(proj_dim=16).cuda()
sigreg = SIGReg().cuda()
opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=0.05)   # 2e-4: fine-tune, not 2e-3
scaler = torch.cuda.amp.GradScaler()

for epoch in range(30):                    # ~30 epochs: warm-start converges fast
    for gs, ls in dl:                      # gs: [B, n_global, 3, 224, 224]
        views = gs.flatten(0, 1).cuda(non_blocking=True)   # [B*n_global, 3, 224, 224]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, proj = net(views)
            inv_loss = (proj.mean(0) - proj).square().mean()   # prediction/invariance — keep as MINIMAL.md
            sigreg_loss = sigreg(proj)                         # anti-collapse
            loss = sigreg_loss * LAMB + inv_loss * (1 - LAMB)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    print(epoch, float(loss), float(sigreg_loss), float(inv_loss))
```

**Start tiny:** 1 scene, a few hundred frames, 3 epochs, 1 GPU → confirm it runs, then scale.

---

## 5. Validate — "did the data work?"

No decoder yet, so judge the encoder by:
1. **Loss curves** — `inv_loss` drops, `sigreg_loss` stays bounded. If `inv_loss→0` instantly or
   `sigreg` blows up, views/data are degenerate.
2. **No collapse** — embedding std across the batch stays > 0.
3. **Linear probe** — freeze the backbone, train a linear head on the **CLS embedding (768-d)** to
   predict **task id** (147) or **contact / no-contact** (threshold F/T magnitude). Beats chance →
   the latent carries real signal → data + encoder are good.

✅ **Phase 1 done** when it trains without collapse and the probe beats chance. This video-only
encoder is also your **vision-only baseline** for the later joint-vs-independent experiment.

---

## 6. Multi-GPU + IO

- **Debug on 1 GPU**, then scale. Option A: `Trainer(devices=8, strategy="ddp", precision="bf16-mixed")`.
  Option B: `torchrun --nproc_per_node=8 phase1_train.py` + wrap `net` in `DistributedDataParallel`
  and use a `DistributedSampler`.
- **Batch:** 96 GB/GPU fits ViT-B easily. Global bs grows with 8 GPUs → **scale LR linearly** if you raise it.
- **IO is the real bottleneck**, not the GPUs: frames on **local NVMe**; `num_workers` 24–32,
  `pin_memory`, `persistent_workers`. If still IO-bound, pack frames into **WebDataset tar shards**.

---

## 7. Pre-flight ⚠️ verify-list (lock these once, up front)
- [ ] `rh20t_api.extract` exact flags.
- [ ] `transformers` call signature + output attr for the CLS embedding (§3 one-time check).
- [ ] `trust_remote_code=True` needed for this checkpoint (likely yes — custom DINOv2-style arch).
- [ ] `stable-pretraining` LeJEPA class path + how to pass a custom backbone (Option A).
- [ ] `SIGReg` import path (Option B) — or copy from `MINIMAL.md`; **don't reimplement the loss**.
- [ ] Frames extracted to local NVMe; F/T↔video alignment confirmed on ≥1 episode.

## Next (only after Phase 1 is green)
- **Phase 2:** + robot state — switch to the API's aligned getters (`get_tcp_aligned`,
  `get_joint_angles_aligned`, `get_ft_aligned`) → `{rgb, joints, torque, tcp, gripper, ft}` per
  timestamp; add the robot-state branch + L2/L3 fusion.
- **Phase 3:** + PixNerd decoder (encoder frozen first → watch the freeze-ceiling).
