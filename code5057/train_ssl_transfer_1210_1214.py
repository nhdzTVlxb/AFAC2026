#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_neural_rec_1100_1105 import (
    RANDOM_STATE,
    RecDataset,
    SeqClassifier,
    build_maps,
    clean_id,
    eval_model,
    predict_model,
    rank_fuse,
    save_submission,
    split_items,
)


def load_pred(path):
    df = pd.read_csv(path)
    return {clean_id(r.uid): split_items(r.prediction) for r in df.itertuples(index=False)}


def load_ssl_into(model, ssl_state, mode):
    params = dict(model.named_parameters())
    copied = []
    if mode in {"full", "emb_only"}:
        for name in ["item_emb.weight", "pos_emb.weight"]:
            if name in ssl_state and name in params and tuple(ssl_state[name].shape) == tuple(params[name].shape):
                with torch.no_grad():
                    params[name].copy_(ssl_state[name].to(params[name].device))
                copied.append(name)
    if mode == "full":
        enc = {k.replace("encoder.", ""): v for k, v in ssl_state.items() if k.startswith("encoder.")}
        missing, unexpected = model.encoder.load_state_dict(enc, strict=False)
    else:
        missing, unexpected = [], []
    return copied, missing, unexpected


def make_optimizer(model, base_lr, wd, enc_mult):
    slow, fast = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("item_emb.") or name.startswith("pos_emb.") or name.startswith("encoder."):
            slow.append(p)
        else:
            fast.append(p)
    return torch.optim.AdamW(
        [
            {"params": slow, "lr": base_lr * enc_mult},
            {"params": fast, "lr": base_lr},
        ],
        weight_decay=wd,
    )


def set_ssl_trainable(model, trainable):
    for name, p in model.named_parameters():
        if name.startswith("item_emb.") or name.startswith("pos_emb.") or name.startswith("encoder."):
            p.requires_grad = trainable


def train_one_transfer(cfg, tr_rows, va_rows, maps, sample, test, out_dir, device, ssl_state):
    item2idx, target_ids, target2idx, target_item_idx, user_arr, uid2urow, cards = maps
    max_len = cfg.get("max_len", 50)
    train_ds = RecDataset(tr_rows, cfg["modes"], item2idx, target2idx, user_arr, uid2urow, max_len, RANDOM_STATE + cfg["seed"])
    val_orig = RecDataset(va_rows, ["orig"], item2idx, target2idx, user_arr, uid2urow, max_len, RANDOM_STATE + 12101)
    val_test = RecDataset(va_rows, ["testmix"], item2idx, target2idx, user_arr, uid2urow, max_len, RANDOM_STATE + 12102)
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=2, pin_memory=True)
    vo_loader = DataLoader(val_orig, batch_size=1024, shuffle=False, num_workers=0)
    vt_loader = DataLoader(val_test, batch_size=1024, shuffle=False, num_workers=0)

    model = SeqClassifier(
        "sasrec", len(item2idx), len(target_ids), target_item_idx, cards,
        dim=96, max_len=max_len, heads=4, layers=3, dropout=cfg.get("dropout", 0.28),
    ).to(device)
    copied, missing, unexpected = load_ssl_into(model, ssl_state, cfg.get("load_mode", "full"))
    print(cfg["name"], "loaded", copied, "missing", len(missing), "unexpected", len(unexpected), flush=True)

    freeze_epochs = cfg.get("freeze_epochs", 0)
    if freeze_epochs > 0:
        set_ssl_trainable(model, False)
    opt = make_optimizer(model, cfg.get("lr", 3.2e-4), cfg.get("wd", 0.025), cfg.get("enc_lr_mult", 0.35))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.get("ls", 0.025))
    best_state, best_score = None, -1.0
    for ep in range(1, cfg.get("epochs", 18) + 1):
        if freeze_epochs > 0 and ep == freeze_epochs + 1:
            set_ssl_trainable(model, True)
            opt = make_optimizer(model, cfg.get("lr", 3.2e-4), cfg.get("wd", 0.025), cfg.get("enc_lr_mult", 0.35))
            print(cfg["name"], "unfroze ssl backbone", flush=True)
        model.train()
        losses = []
        for seq, ux, y, lengths in train_loader:
            seq, ux, y, lengths = seq.to(device), ux.to(device), y.to(device), lengths.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(seq, ux, lengths)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        nd_o, hit_o = eval_model(model, vo_loader, device)
        nd_t, hit_t = eval_model(model, vt_loader, device)
        score = 0.10 * nd_o + 0.90 * nd_t
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"{cfg['name']} ep={ep:02d} loss={np.mean(losses):.4f} val_orig_ndcg={nd_o:.5f} hit={hit_o:.5f} val_test_ndcg={nd_t:.5f} hit={hit_t:.5f}", flush=True)
    model.load_state_dict(best_state)
    pred = predict_model(model, test, item2idx, target_ids, user_arr, uid2urow, max_len, device)
    save_submission(sample, pred, os.path.join(out_dir, cfg["name"], "A2.csv"))
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--ssl-ckpt", default="/home/cyp/speedsci/AFAC2026/c100_lowmem/ssl_seq_pretrain_1200_1204/ssl_nextitem_sasrec_l3.pt")
    args = ap.parse_args()
    random.seed(RANDOM_STATE + 1210)
    np.random.seed(RANDOM_STATE + 1210)
    torch.manual_seed(RANDOM_STATE + 1210)
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
    ssl_state = torch.load(args.ssl_ckpt, map_location="cpu")
    print("loaded ssl ckpt", args.ssl_ckpt, flush=True)

    uniq = np.array(pd.unique(train["uid"]))
    rng = np.random.default_rng(RANDOM_STATE + 1210)
    rng.shuffle(uniq)
    val_u = set(uniq[:max(1, int(len(uniq) * 0.15))])
    tr_rows = train.loc[~train["uid"].isin(val_u)].to_dict("records")
    va_rows = train.loc[train["uid"].isin(val_u)].to_dict("records")
    print("train rows", len(tr_rows), "val rows", len(va_rows), flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    configs = [
        {"name": "output1210_ssl_lowlr", "modes": ["testmix", "last3", "empty"], "seed": 10, "epochs": 20, "enc_lr_mult": 0.22, "load_mode": "full"},
        {"name": "output1211_ssl_freeze4", "modes": ["testmix", "last3", "empty"], "seed": 11, "epochs": 20, "enc_lr_mult": 0.30, "freeze_epochs": 4, "load_mode": "full"},
        {"name": "output1212_ssl_embonly", "modes": ["testmix", "last3", "empty"], "seed": 12, "epochs": 18, "enc_lr_mult": 1.0, "load_mode": "emb_only"},
    ]
    preds = [train_one_transfer(c, tr_rows, va_rows, maps, sample, test, args.output_dir, device, ssl_state) for c in configs]
    save_submission(sample, rank_fuse(preds, [1.1, 1.0, 0.8]), os.path.join(args.output_dir, "output1213_ssl_transfer_fusion", "A2.csv"))
    try:
        old1120 = load_pred("/home/cyp/speedsci/AFAC2026/c100_lowmem/neural_pretrain_1120_1126/output1120_deep_seed20/A2.csv")
        old1113 = load_pred("/home/cyp/speedsci/AFAC2026/c100_lowmem/neural_pretrain_1110_1116/output1113_sasrec_pre_deep/A2.csv")
        old1200 = load_pred("/home/cyp/speedsci/AFAC2026/c100_lowmem/ssl_seq_pretrain_1200_1204/output1200_ssl_testmix/A2.csv")
        save_submission(sample, rank_fuse([old1120, old1113, old1200] + preds[:2], [1.1, 0.95, 1.0, 0.85, 0.75]), os.path.join(args.output_dir, "output1214_ssl_transfer_x_stable", "A2.csv"))
    except Exception as e:
        print("skip old fusion", repr(e), flush=True)


if __name__ == "__main__":
    main()
