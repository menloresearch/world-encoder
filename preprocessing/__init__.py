"""Data preprocessing — raw RH20T scenes -> jpg frames -> WebDataset shards.

Modules:
    extract_frames : RH20T mp4 -> timestamped jpg frames (wraps rh20t_api.extract)
    make_shards    : frames -> WebDataset shards (writes count.txt when done)
    gate           : F/T <-> video alignment sanity check on one scene
    analyze_cfgs   : per-cfg dataset analysis (regenerates DATA.md numbers)

Run everything via preprocess_all.sh (resumable, per-cfg).
"""
