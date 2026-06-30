"""Phase 1 — video-only LeJEPA fine-tune on RH20T cfg3.

Modules:
    extract_frames : RH20T mp4 -> timestamped jpg frames (wraps rh20t_api.extract)
    dataset        : MultiCropRGB (DINOv2-style multi-crop) + collate
    model          : ViTv2 warm-start backbone + LeJEPAVideo (reuses stable-pretraining loss)
    train          : single-GPU debug loop (DDP notes inside)
    validate       : linear-probe / collapse checks
"""
