#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from collections import defaultdict

import pandas as pd


def clean_id(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def split_items(x):
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return []
    return [clean_id(v) for v in s.split(",") if clean_id(v)]


def load_pred(path):
    df = pd.read_csv(path)
    return {clean_id(r.uid): split_items(r.prediction) for r in df.itertuples(index=False)}


def rank_fuse(pred_maps, weights):
    out = {}
    uids = pred_maps[0].keys()
    for uid in uids:
        score = defaultdict(float)
        for mp, w in zip(pred_maps, weights):
            for r, it in enumerate(mp[uid]):
                score[it] += w * (10 - r) / 10.0
        out[uid] = [it for it, _ in sorted(score.items(), key=lambda kv: -kv[1])[:10]]
    return out


def save_submission(sample_path, uid_to_pred, out_path):
    sample = pd.read_csv(sample_path)
    sample["uid"] = sample["uid"].map(clean_id)
    sample["prediction"] = sample["uid"].map(lambda u: ",".join(uid_to_pred[u][:10]))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sample.to_csv(out_path, index=False)
    print("Saved:", out_path, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--component-dir", default="components", help="Directory containing component A2 csv files")
    ap.add_argument("--sample", default="/home/cyp/speedsci/AFAC2026/v1/framework/data/rec_data/sample_submission.csv")
    ap.add_argument("--output", default="output1214_ssl_transfer_x_stable/A2.csv")
    args = ap.parse_args()

    paths = [
        os.path.join(args.component_dir, "1120_deep_seed20_A2.csv"),
        os.path.join(args.component_dir, "1113_sasrec_pre_deep_A2.csv"),
        os.path.join(args.component_dir, "1200_ssl_testmix_A2.csv"),
        os.path.join(args.component_dir, "1210_ssl_lowlr_A2.csv"),
        os.path.join(args.component_dir, "1211_ssl_freeze4_A2.csv"),
    ]
    weights = [1.10, 0.95, 1.00, 0.85, 0.75]
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(p)
    print("components:", flush=True)
    for p, w in zip(paths, weights):
        print(f"  weight={w:.2f} {p}", flush=True)
    fused = rank_fuse([load_pred(p) for p in paths], weights)
    save_submission(args.sample, fused, args.output)


if __name__ == "__main__":
    main()
