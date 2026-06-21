#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

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


class SSLNextItemDataset(Dataset):
    """Self-supervised next-item prefixes from train + test histories.

    This uses only observed sequences, no target labels from test.  It is
    transductive representation learning: teach the SASRec encoder the item
    transition geometry before Task2 fine-tuning.
    """

    def __init__(self, rows, item2idx, max_len=50, max_examples=850000, seed=2025):
        rng = np.random.default_rng(seed)
        examples = []
        for r in rows:
            raw = [item2idx[x] for x in split_items(r.get("item_seq_raw")) if x in item2idx]
            if len(raw) < 2:
                continue
            L = len(raw)
            # Recent positions matter most for the public test distribution,
            # but include a little random context so the encoder is not only a
            # last-click model.
            pos = list(range(max(1, L - 48), L))
            if L > 64:
                extra = rng.choice(np.arange(1, L), size=min(10, L - 1), replace=False).tolist()
                pos.extend(extra)
            rng.shuffle(pos)
            for p in pos[:36]:
                prefix = raw[max(0, p - max_len):p]
                y = raw[p]
                if prefix and y > 0:
                    examples.append((prefix, y))
                if len(examples) >= max_examples:
                    break
            if len(examples) >= max_examples:
                break
        rng.shuffle(examples)
        self.examples = examples
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        prefix, y = self.examples[i]
        arr = np.zeros(self.max_len, dtype=np.int64)
        tail = prefix[-self.max_len:]
        arr[-len(tail):] = np.asarray(tail, dtype=np.int64)
        return torch.from_numpy(arr), torch.tensor(len(tail), dtype=torch.long), torch.tensor(y, dtype=torch.long)


class SSLSeqEncoder(nn.Module):
    def __init__(self, n_items, dim=96, max_len=50, heads=4, layers=3, dropout=0.25):
        super().__init__()
        self.dim = dim
        self.item_emb = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)
        self.bias = nn.Parameter(torch.zeros(n_items + 1))

    def forward(self, seq, lengths):
        bsz, seqlen = seq.shape
        pos = torch.arange(seqlen, device=seq.device).unsqueeze(0).expand(bsz, seqlen)
        x = self.item_emb(seq) + self.pos_emb(pos)
        h = self.encoder(x, src_key_padding_mask=seq.eq(0))
        idx = seqlen - lengths.clamp(min=1)
        z = h[torch.arange(bsz, device=seq.device), idx]
        z = torch.where(lengths.unsqueeze(1) > 0, z, torch.zeros_like(z))
        z = self.norm(z)
        return z @ self.item_emb.weight.T + self.bias.unsqueeze(0)


def ssl_pretrain(train, test, item2idx, out_path, device, max_len=50, dim=96, layers=3, epochs=8):
    if os.path.exists(out_path):
        print("loading ssl checkpoint", out_path, flush=True)
        return torch.load(out_path, map_location="cpu")
    rows = train.to_dict("records") + test.to_dict("records")
    ds = SSLNextItemDataset(rows, item2idx, max_len=max_len, seed=RANDOM_STATE + 1200)
    dl = DataLoader(ds, batch_size=512, shuffle=True, num_workers=2, pin_memory=True)
    model = SSLSeqEncoder(len(item2idx), dim=dim, max_len=max_len, layers=layers, dropout=0.25).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.03)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.02)
    print("ssl examples", len(ds), flush=True)
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for seq, lengths, y in dl:
            seq, lengths, y = seq.to(device), lengths.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(seq, lengths)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        print(f"ssl ep={ep:02d} loss={np.mean(losses):.5f}", flush=True)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(state, out_path)
    print("saved ssl checkpoint", out_path, flush=True)
    return state


def train_one_ssl(cfg, tr_rows, va_rows, maps, sample, test, out_dir, device, ssl_state):
    item2idx, target_ids, target2idx, target_item_idx, user_arr, uid2urow, cards = maps
    max_len = cfg.get("max_len", 50)
    train_ds = RecDataset(tr_rows, cfg["modes"], item2idx, target2idx, user_arr, uid2urow, max_len, RANDOM_STATE + cfg["seed"])
    val_orig = RecDataset(va_rows, ["orig"], item2idx, target2idx, user_arr, uid2urow, max_len, RANDOM_STATE + 12001)
    val_test = RecDataset(va_rows, ["testmix"], item2idx, target2idx, user_arr, uid2urow, max_len, RANDOM_STATE + 12002)
    train_loader = DataLoader(train_ds, batch_size=cfg.get("batch", 512), shuffle=True, num_workers=2, pin_memory=True)
    vo_loader = DataLoader(val_orig, batch_size=1024, shuffle=False, num_workers=0)
    vt_loader = DataLoader(val_test, batch_size=1024, shuffle=False, num_workers=0)
    model = SeqClassifier(
        "sasrec", len(item2idx), len(target_ids), target_item_idx, cards,
        dim=96, max_len=max_len, heads=4, layers=3, dropout=cfg.get("dropout", 0.27),
    ).to(device)
    with torch.no_grad():
        copied = []
        for name in ["item_emb.weight", "pos_emb.weight"]:
            if name in ssl_state and tuple(ssl_state[name].shape) == tuple(dict(model.named_parameters())[name].shape):
                dict(model.named_parameters())[name].copy_(ssl_state[name].to(device))
                copied.append(name)
        enc = {k.replace("encoder.", ""): v for k, v in ssl_state.items() if k.startswith("encoder.")}
        missing, unexpected = model.encoder.load_state_dict(enc, strict=False)
        print(cfg["name"], "loaded ssl", copied, "enc_missing", len(missing), "enc_unexpected", len(unexpected), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 3.2e-4), weight_decay=cfg.get("wd", 0.025))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.get("ls", 0.025))
    best_state, best_score = None, -1.0
    for ep in range(1, cfg.get("epochs", 18) + 1):
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


def load_pred(path):
    df = pd.read_csv(path)
    return {clean_id(r.uid): split_items(r.prediction) for r in df.itertuples(index=False)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    random.seed(RANDOM_STATE + 1200)
    np.random.seed(RANDOM_STATE + 1200)
    torch.manual_seed(RANDOM_STATE + 1200)
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

    ssl_state = ssl_pretrain(
        train, test, item2idx,
        os.path.join(args.output_dir, "ssl_nextitem_sasrec_l3.pt"),
        device,
        max_len=50,
        epochs=8,
    )

    uniq = np.array(pd.unique(train["uid"]))
    rng = np.random.default_rng(RANDOM_STATE + 1200)
    rng.shuffle(uniq)
    val_u = set(uniq[:max(1, int(len(uniq) * 0.15))])
    tr_rows = train.loc[~train["uid"].isin(val_u)].to_dict("records")
    va_rows = train.loc[train["uid"].isin(val_u)].to_dict("records")
    print("train rows", len(tr_rows), "val rows", len(va_rows), flush=True)

    configs = [
        {"name": "output1200_ssl_testmix", "modes": ["testmix", "last3", "empty"], "seed": 0, "epochs": 20, "dropout": 0.28, "lr": 3.2e-4},
        {"name": "output1201_ssl_noempty", "modes": ["testmix", "last3", "last1"], "seed": 1, "epochs": 18, "dropout": 0.26, "lr": 3.2e-4},
        {"name": "output1202_ssl_last5", "modes": ["testmix", "last5", "last3", "empty"], "seed": 2, "epochs": 18, "dropout": 0.28, "lr": 3.2e-4},
    ]
    preds = [train_one_ssl(c, tr_rows, va_rows, maps, sample, test, args.output_dir, device, ssl_state) for c in configs]
    save_submission(sample, rank_fuse(preds, [1.1, 0.9, 1.0]), os.path.join(args.output_dir, "output1203_ssl_fusion", "A2.csv"))
    try:
        old1120 = load_pred("/home/cyp/speedsci/AFAC2026/c100_lowmem/neural_pretrain_1120_1126/output1120_deep_seed20/A2.csv")
        old1113 = load_pred("/home/cyp/speedsci/AFAC2026/c100_lowmem/neural_pretrain_1110_1116/output1113_sasrec_pre_deep/A2.csv")
        save_submission(sample, rank_fuse([old1120, old1113] + preds, [1.2, 1.0, 0.85, 0.6, 0.75]), os.path.join(args.output_dir, "output1204_ssl_x_stable", "A2.csv"))
    except Exception as e:
        print("skip old fusion", repr(e), flush=True)


if __name__ == "__main__":
    main()
