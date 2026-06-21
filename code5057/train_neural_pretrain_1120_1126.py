#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from train_neural_rec_1100_1105 import (
    RANDOM_STATE,
    SeqClassifier,
    build_maps,
    clean_id,
    eval_model,
    predict_model,
    rank_fuse,
    save_submission,
    split_items,
    train_one,
)


class PairDataset(Dataset):
    def __init__(self, train_rows, item2idx, n_items, max_pairs=900000, seed=2025):
        rng = np.random.default_rng(seed)
        counts = Counter()
        seqs = []
        for r in train_rows:
            raw = [item2idx[x] for x in split_items(r.get("item_seq_raw")) if x in item2idx]
            if raw:
                seqs.append(raw)
                counts.update(raw)
        pop = np.ones(n_items + 1, dtype=np.float64)
        for i, c in counts.items():
            pop[i] = c ** 0.75
        pop[0] = 0
        pop = pop / pop.sum()
        pairs = []
        for raw in seqs:
            L = len(raw)
            if L < 2:
                continue
            # sample local context pairs, recency-biased
            positions = range(max(0, L - 80), L)
            for i in positions:
                lo, hi = max(0, i - 4), min(L, i + 5)
                for j in range(lo, hi):
                    if i != j:
                        pairs.append((raw[i], raw[j]))
                        if len(pairs) >= max_pairs:
                            break
                if len(pairs) >= max_pairs:
                    break
            if len(pairs) >= max_pairs:
                break
        self.pairs = np.asarray(pairs, dtype=np.int64)
        self.neg = rng.choice(np.arange(n_items + 1), size=(len(self.pairs),), p=pop).astype(np.int64)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        a, p = self.pairs[i]
        n = self.neg[i]
        return torch.tensor(a), torch.tensor(p), torch.tensor(n)


class SkipGramBPR(nn.Module):
    def __init__(self, n_items, dim):
        super().__init__()
        self.in_emb = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.out_emb = nn.Embedding(n_items + 1, dim, padding_idx=0)
        nn.init.normal_(self.in_emb.weight, std=0.03)
        nn.init.normal_(self.out_emb.weight, std=0.03)

    def forward(self, a, p, n):
        q = self.in_emb(a)
        pe = self.out_emb(p)
        ne = self.out_emb(n)
        ps = (q * pe).sum(-1)
        ns = (q * ne).sum(-1)
        return -torch.nn.functional.logsigmoid(ps - ns).mean()


def pretrain_item_emb(train_rows, item2idx, dim, device, out_path):
    n_items = len(item2idx)
    ds = PairDataset(train_rows, item2idx, n_items, max_pairs=900000, seed=RANDOM_STATE + 1110)
    dl = DataLoader(ds, batch_size=4096, shuffle=True, num_workers=2, pin_memory=True)
    model = SkipGramBPR(n_items, dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    for ep in range(1, 7):
        losses = []
        model.train()
        for a, p, n in dl:
            a, p, n = a.to(device), p.to(device), n.to(device)
            opt.zero_grad(set_to_none=True)
            loss = model(a, p, n)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        print(f"pretrain ep={ep} loss={np.mean(losses):.5f} pairs={len(ds)}", flush=True)
    emb = model.in_emb.weight.detach().cpu()
    torch.save(emb, out_path)
    return emb


def train_one_pretrained(cfg, train_rows, val_rows, maps, sample, test, out_dir, device, pretrained_emb):
    # Reuse train_one logic by monkey patching initial embedding after model construction:
    # duplicated compactly here to keep this script independent.
    from train_neural_rec_1100_1105 import RecDataset

    item2idx, target_ids, target2idx, target_item_idx, user_arr, uid2urow, cards = maps
    max_len = cfg.get("max_len", 50)
    train_ds = RecDataset(train_rows, cfg["modes"], item2idx, target2idx, user_arr, uid2urow, max_len, RANDOM_STATE + cfg["seed"])
    val_orig = RecDataset(val_rows, ["orig"], item2idx, target2idx, user_arr, uid2urow, max_len, RANDOM_STATE + 1)
    val_test = RecDataset(val_rows, ["testmix"], item2idx, target2idx, user_arr, uid2urow, max_len, RANDOM_STATE + 2)
    train_loader = DataLoader(train_ds, batch_size=cfg.get("batch", 512), shuffle=True, num_workers=2, pin_memory=True)
    vo_loader = DataLoader(val_orig, batch_size=1024, shuffle=False, num_workers=0)
    vt_loader = DataLoader(val_test, batch_size=1024, shuffle=False, num_workers=0)
    model = SeqClassifier(cfg["kind"], len(item2idx), len(target_ids), target_item_idx, cards,
                          dim=cfg.get("dim", 96), max_len=max_len, heads=cfg.get("heads", 4),
                          layers=cfg.get("layers", 2), dropout=cfg.get("dropout", 0.2)).to(device)
    with torch.no_grad():
        model.item_emb.weight.copy_(pretrained_emb.to(device))
        model.item_emb.weight[0].zero_()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 5e-4), weight_decay=cfg.get("wd", 0.025))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.get("ls", 0.025))
    best_state, best_score = None, -1
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    random.seed(RANDOM_STATE + 1110)
    np.random.seed(RANDOM_STATE + 1110)
    torch.manual_seed(RANDOM_STATE + 1110)
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
    uniq = np.array(pd.unique(train["uid"]))
    rng = np.random.default_rng(RANDOM_STATE + 1110)
    rng.shuffle(uniq)
    val_u = set(uniq[:max(1, int(len(uniq) * 0.15))])
    tr_rows = train.loc[~train["uid"].isin(val_u)].to_dict("records")
    va_rows = train.loc[train["uid"].isin(val_u)].to_dict("records")
    print("train rows", len(tr_rows), "val rows", len(va_rows), flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    emb_path = os.path.join(args.output_dir, "item2vec_bpr_emb.pt")
    existing = "/home/cyp/speedsci/AFAC2026/c100_lowmem/neural_pretrain_1110_1116/item2vec_bpr_emb.pt"
    if os.path.exists(existing):
        print(f"loading existing pretrain emb: {existing}", flush=True)
        emb = torch.load(existing, map_location="cpu")
    else:
        emb = pretrain_item_emb(train.to_dict("records"), item2idx, 96, device, emb_path)

    configs = [
        {"name": "output1120_deep_seed20", "kind": "sasrec", "modes": ["testmix", "last3", "empty"], "seed": 20, "epochs": 20, "dim": 96, "layers": 3, "dropout": 0.28, "lr": 3.5e-4},
        {"name": "output1121_deep_seed21", "kind": "sasrec", "modes": ["testmix", "last3", "empty"], "seed": 21, "epochs": 20, "dim": 96, "layers": 3, "dropout": 0.24, "lr": 4.0e-4},
        {"name": "output1122_deep_short_seed22", "kind": "sasrec", "modes": ["testmix", "last3", "last1", "empty"], "seed": 22, "epochs": 20, "dim": 96, "layers": 3, "dropout": 0.26, "lr": 3.5e-4},
        {"name": "output1123_deep_last5_seed23", "kind": "sasrec", "modes": ["testmix", "last5", "last3", "empty"], "seed": 23, "epochs": 20, "dim": 96, "layers": 3, "dropout": 0.28, "lr": 3.5e-4},
        {"name": "output1124_gru_pre_seed24", "kind": "gru", "modes": ["testmix", "last3", "empty"], "seed": 24, "epochs": 18, "dim": 96, "dropout": 0.20, "lr": 4.5e-4},
    ]
    preds = []
    for cfg in configs:
        preds.append(train_one_pretrained(cfg, tr_rows, va_rows, maps, sample, test, args.output_dir, device, emb))
    fused = rank_fuse(preds, [1.15, 1.1, 1.0, 1.0, 0.75])
    save_submission(sample, fused, os.path.join(args.output_dir, "output1125_deep_nn_fusion", "A2.csv"))

    # Conservative blend with the best non-pretrained neural output: still pure NN family, no LGB.
    try:
        old = pd.read_csv("/home/cyp/speedsci/AFAC2026/c100_lowmem/neural_rec_1100_1105/output1100_sasrec_testmix/A2.csv")
        old_map = {clean_id(r.uid): split_items(r.prediction) for r in old.itertuples(index=False)}
        pred_maps = [old_map] + preds
        old1113 = pd.read_csv("/home/cyp/speedsci/AFAC2026/c100_lowmem/neural_pretrain_1110_1116/output1113_sasrec_pre_deep/A2.csv")
        old1113_map = {clean_id(r.uid): split_items(r.prediction) for r in old1113.itertuples(index=False)}
        pred_maps = [old_map, old1113_map] + preds
        fused2 = rank_fuse(pred_maps, [0.8, 1.25, 1.0, 1.0, 0.9, 0.9, 0.65])
        save_submission(sample, fused2, os.path.join(args.output_dir, "output1126_x_1113_fusion", "A2.csv"))
    except Exception as e:
        print("skip 1115", repr(e), flush=True)


if __name__ == "__main__":
    main()
