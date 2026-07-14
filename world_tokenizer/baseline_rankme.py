"""Compute RankMe of the two feature baselines (raw mean-pooled frozen ViT, PCA-256)
on each embodiment's held-out test rows -- the same split logic as
train_chunks.eval_embodiment. No model needed; fills the baseline RankMe cells of the
main quantitative table."""
import json
import os
import sys

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from world_tokenizer.dataloader import ChunkDataset, load_split  # noqa: E402
from world_tokenizer.train_chunks import EMBODIMENTS, rankme  # noqa: E402

CACHE = "/mnt/nas/data/RH20T/caches"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "checkpoints", "baseline_rankme.json")


def main():
    split = load_split()
    res = {}
    for emb, cfgs in EMBODIMENTS.items():
        ds = ChunkDataset(CACHE, cfgs)
        is_test = np.array([split[g] == "test" for g in
                            (ds.groups[i] for i in ds._group_idx)])
        patch = ds._d["patch"]                        # [N,1,196,768] or [N,196,768]
        patch = patch.reshape(len(ds), -1, patch.shape[-1])
        # chunked mean-pool over patches to float32
        raw = np.empty((len(ds), patch.shape[-1]), dtype=np.float32)
        for i in range(0, len(ds), 4096):
            raw[i:i + 4096] = patch[i:i + 4096].astype(np.float32).mean(1)
        del ds, patch
        tr, te = ~is_test, is_test
        pca = PCA(n_components=min(256, raw.shape[1], tr.sum()),
                  random_state=0).fit(raw[tr])
        res[emb] = {
            "n_test": int(te.sum()),
            "rankme_raw": rankme(raw[te]),
            "rankme_pca256": rankme(pca.transform(raw[te])),
        }
        print(emb, res[emb], flush=True)
        del raw
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
