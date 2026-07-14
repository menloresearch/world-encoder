"""Sample robot-frame reconstructions from a trained latent-conditioned PixNerd checkpoint.
For N test chunks: take z_v -> generate an image (EulerSampler, CFG) -> lay it next to the
REAL frame. Shows what the frozen world-encoder latent reconstructs (lossy by design).

    python sample_decode.py --ckpt /mnt/nas/data/RH20T/decode/ur5/run/.../last.ckpt \
        --manifest /mnt/nas/data/RH20T/decode/ur5 --n 8 --out figures/decode/recon.png
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from functools import partial

from src.models.transformer.pixnerd_t2i import PixNerDiT
from src.models.conditioner.latent import LatentConditioner
from src.models.autoencoder.pixel import PixelAE
from src.diffusion.flow_matching.sampling import EulerSampler, ode_step_fn
from src.diffusion.flow_matching.scheduling import LinearScheduler
from src.diffusion.base.guidance import simple_guidance_fn
from src.data.dataset.imagenet import center_crop_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default="/mnt/nas/data/RH20T/decode/ur5")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--out", default="figures/decode/recon.png")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    dev = "cuda"

    # rebuild the denoiser exactly as in the config, load the EMA weights from the ckpt
    net = PixNerDiT(in_channels=3, patch_size=16, num_groups=16, hidden_size=512,
                    decoder_hidden_size=64, num_encoder_blocks=8, num_decoder_blocks=2,
                    num_text_blocks=2, txt_embed_dim=256, txt_max_length=1).to(dev)
    sd = torch.load(args.ckpt, map_location="cpu")["state_dict"]
    # use the ONLINE denoiser (EMA lags badly at early checkpoints with decay 0.9999)
    w = {k[len("denoiser."):]: v for k, v in sd.items() if k.startswith("denoiser.")}
    net.load_state_dict(w); net.eval()

    cond = LatentConditioner(dim=256)
    vae = PixelAE(scale=1.0)
    sampler = EulerSampler(scheduler=LinearScheduler(), w_scheduler=LinearScheduler(),
                           num_steps=args.steps, guidance=args.guidance,
                           guidance_fn=simple_guidance_fn, step_fn=ode_step_fn)

    zv = np.load(f"{args.manifest}/zv_{args.split}.npy")
    paths = open(f"{args.manifest}/frames_{args.split}.txt").read().splitlines()
    rng = np.random.default_rng(0)
    idx = rng.choice(len(zv), args.n, replace=False)
    y = torch.from_numpy(zv[idx]).float()
    crop = partial(center_crop_fn, image_size=args.res)
    reals = torch.stack([to_tensor(crop(Image.open(paths[i]).convert("RGB"))) for i in idx])  # [n,3,H,W] 0..1

    with torch.no_grad():
        condition, uncondition = cond(y)
        xT = torch.randn(len(idx), 3, args.res, args.res, device=dev)
        samples = sampler(net, xT, condition, uncondition)          # [-1,1]-ish
        samples = vae.decode(samples).float().clamp(-1, 1)
        gen = ((samples + 1) / 2).cpu()                             # 0..1

    # grid: top row real, bottom row decoded, per sample
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(idx)
    fig, axes = plt.subplots(2, n, figsize=(1.7 * n, 3.6))
    for j in range(n):
        axes[0, j].imshow(reals[j].permute(1, 2, 0).numpy()); axes[0, j].axis("off")
        axes[1, j].imshow(gen[j].permute(1, 2, 0).numpy()); axes[1, j].axis("off")
    axes[0, 0].set_ylabel("real", rotation=0, ha="right"); axes[1, 0].set_ylabel("decoded", rotation=0, ha="right")
    fig.suptitle("Robot frame decoded from frozen vision-only latent z_v (top: real, bottom: decoded)")
    fig.savefig(args.out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
