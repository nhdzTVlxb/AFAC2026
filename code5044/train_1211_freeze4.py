#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch

from train_neural_rec_1100_1105 import RANDOM_STATE, build_maps, clean_id
from train_ssl_seq_pretrain_1200_1204 import ssl_pretrain
from train_ssl_transfer_1210_1214 import train_one_transfer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="Path containing train.csv/test.csv/user.csv/item.csv/sample_submission.csv")
    ap.add_argument("--output-dir", default="output_1211_freeze4", help="Directory for checkpoint, logs, and submission")
    ap.add_argument("--ssl-epochs", type=int, default=8, help="Self-supervised next-item pretrain epochs")
    ap.add_argument("--reuse-ssl", action="store_true", help="Reuse output-dir/ssl_nextitem_sasrec_l3.pt if it already exists")
    args = ap.parse_args()

    random.seed(RANDOM_STATE + 1211)
    np.random.seed(RANDOM_STATE + 1211)
    torch.manual_seed(RANDOM_STATE + 1211)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)

    train = pd.read_csv(os.path.join(args.data_root, "train.csv"))
    test = pd.read_csv(os.path.join(args.data_root, "test.csv"))
    user = pd.read_csv(os.path.join(args.data_root, "user.csv"))
    item = pd.read_csv(os.path.join(args.data_root, "item.csv"))
    sample = pd.read_csv(os.path.join(args.data_root, "sample_submission.csv"))

    train["uid"] = train["uid"].map(clean_id)
    train["target_iid"] = train["target_iid"].map(clean_id)
    test["uid"] = test["uid"].map(clean_id)
    item["iid"] = item["iid"].map(clean_id)

    maps = build_maps(train, test, user, item)
    item2idx = maps[0]
    os.makedirs(args.output_dir, exist_ok=True)

    ssl_ckpt = os.path.join(args.output_dir, "ssl_nextitem_sasrec_l3.pt")
    if args.reuse_ssl and os.path.exists(ssl_ckpt):
        print("reuse ssl checkpoint", ssl_ckpt, flush=True)
        ssl_state = torch.load(ssl_ckpt, map_location="cpu")
    else:
        ssl_state = ssl_pretrain(
            train,
            test,
            item2idx,
            ssl_ckpt,
            device,
            max_len=50,
            epochs=args.ssl_epochs,
        )

    # Same validation split seed family as the original 1210/1211 transfer run.
    uniq = np.array(pd.unique(train["uid"]))
    rng = np.random.default_rng(RANDOM_STATE + 1210)
    rng.shuffle(uniq)
    val_u = set(uniq[:max(1, int(len(uniq) * 0.15))])
    tr_rows = train.loc[~train["uid"].isin(val_u)].to_dict("records")
    va_rows = train.loc[train["uid"].isin(val_u)].to_dict("records")
    print("train rows", len(tr_rows), "val rows", len(va_rows), flush=True)

    cfg = {
        "name": "output1211_ssl_freeze4",
        "modes": ["testmix", "last3", "empty"],
        "seed": 11,
        "epochs": 20,
        "enc_lr_mult": 0.30,
        "freeze_epochs": 4,
        "load_mode": "full",
        "dropout": 0.28,
        "lr": 3.2e-4,
        "wd": 0.025,
        "ls": 0.025,
    }
    train_one_transfer(cfg, tr_rows, va_rows, maps, sample, test, args.output_dir, device, ssl_state)
    print("final submission:", os.path.join(args.output_dir, "output1211_ssl_freeze4", "A2.csv"), flush=True)


if __name__ == "__main__":
    main()
