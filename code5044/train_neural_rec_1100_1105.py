#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


RANDOM_STATE = 2025


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


def dedup(items):
    seen, out = set(), []
    for it in items:
        if it and it not in seen:
            seen.add(it)
            out.append(it)
    return out


def truncate(raw, rng, mode):
    if mode == "orig":
        return raw
    if mode == "empty":
        return []
    if mode == "last1":
        return raw[-1:] if raw else []
    if mode == "last3":
        k = min(len(raw), int(rng.integers(1, 4)))
        return raw[-k:] if k > 0 else []
    if mode == "last5":
        k = min(len(raw), int(rng.integers(2, 6)))
        return raw[-k:] if k > 0 else []
    # testmix: observed public test length distribution
    r = rng.random()
    if r < 0.3515:
        k = 0
    elif r < 0.4518:
        k = 1
    elif r < 0.8992:
        k = int(rng.integers(2, 4))
    elif r < 0.9006:
        k = int(rng.integers(4, 6))
    elif r < 0.9053:
        k = int(rng.integers(6, 11))
    elif r < 0.9202:
        k = int(rng.integers(11, 31))
    else:
        k = int(rng.integers(31, min(200, max(len(raw), 31)) + 1))
    return raw[-min(k, len(raw)):] if k > 0 else []


def ndcg_hit_from_logits(logits, y, k=10):
    top = torch.topk(logits, k=k, dim=1).indices.cpu().numpy()
    yy = y.cpu().numpy()
    total, hit = 0.0, 0
    for pred, t in zip(top, yy):
        where = np.where(pred == t)[0]
        if len(where):
            r = int(where[0])
            hit += 1
            total += 1.0 / math.log2(r + 2.0)
    return total, hit, len(yy)


def save_submission(sample, uid_to_pred, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = sample.copy()
    out["uid"] = out["uid"].map(clean_id)
    out["prediction"] = out["uid"].map(lambda u: ",".join(uid_to_pred[u][:10]))
    out.to_csv(path, index=False)
    print("Saved:", path, flush=True)


class RecDataset(Dataset):
    def __init__(self, rows, modes, item2idx, target2idx, user_arr, uid2urow, max_len, seed):
        self.samples = []
        rng = np.random.default_rng(seed)
        for row in rows:
            raw0 = split_items(row["item_seq_raw"])
            for mode in modes:
                raw = truncate(raw0, rng, mode)
                seq = [item2idx[x] for x in raw[-max_len:] if x in item2idx]
                self.samples.append((row["uid"], seq, target2idx[row["target_iid"]]))
        self.user_arr = user_arr
        self.uid2urow = uid2urow
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        uid, seq, y = self.samples[i]
        arr = np.zeros(self.max_len, dtype=np.int64)
        if seq:
            arr[-len(seq):] = np.asarray(seq, dtype=np.int64)
        u = self.user_arr[self.uid2urow[uid]]
        return torch.from_numpy(arr), torch.from_numpy(u), torch.tensor(y, dtype=torch.long), torch.tensor(len(seq), dtype=torch.long)


class SeqClassifier(nn.Module):
    def __init__(self, kind, n_items, n_targets, target_item_idx, user_cardinalities, dim=96, max_len=50, heads=4, layers=2, dropout=0.2):
        super().__init__()
        self.kind = kind
        self.dim = dim
        self.max_len = max_len
        self.item_emb = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.user_embs = nn.ModuleList([nn.Embedding(c + 1, dim // 4) for c in user_cardinalities])
        user_dim = len(user_cardinalities) * (dim // 4)
        self.user_mlp = nn.Sequential(
            nn.Linear(user_dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        if kind == "sasrec":
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
        elif kind == "gru":
            self.encoder = nn.GRU(dim, dim, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        else:
            raise ValueError(kind)
        self.gate = nn.Sequential(nn.Linear(dim * 2 + 1, dim), nn.ReLU(), nn.Linear(dim, 1), nn.Sigmoid())
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.out_bias = nn.Parameter(torch.zeros(n_targets))
        self.register_buffer("target_item_idx", torch.tensor(target_item_idx, dtype=torch.long))

    def forward(self, seq, user_x, lengths):
        bsz, seqlen = seq.shape
        pos = torch.arange(seqlen, device=seq.device).unsqueeze(0).expand(bsz, seqlen)
        x = self.item_emb(seq) + self.pos_emb(pos)
        pad = seq.eq(0)
        if self.kind == "sasrec":
            h = self.encoder(x, src_key_padding_mask=pad)
            # last non-pad position
            nonzero = lengths.clamp(min=1)
            idx = seqlen - nonzero
            seq_repr = h[torch.arange(bsz, device=seq.device), idx]
            seq_repr = torch.where(lengths.unsqueeze(1) > 0, seq_repr, torch.zeros_like(seq_repr))
        else:
            packed_lengths = lengths.clamp(min=1).cpu()
            x2 = x.clone()
            # GRU cannot pack length 0; empty rows use one zero token.
            _, h = self.encoder(x2)
            seq_repr = h[-1]
            seq_repr = torch.where(lengths.unsqueeze(1) > 0, seq_repr, torch.zeros_like(seq_repr))
        uemb = []
        for j, emb in enumerate(self.user_embs):
            uemb.append(emb(user_x[:, j]))
        u = self.user_mlp(torch.cat(uemb, dim=1))
        len_feat = torch.log1p(lengths.float()).unsqueeze(1) / math.log1p(200.0)
        g = self.gate(torch.cat([seq_repr, u, len_feat], dim=1))
        z = self.norm(g * seq_repr + (1.0 - g) * u)
        z = self.dropout(z)
        target_emb = self.item_emb(self.target_item_idx)
        return z @ target_emb.T + self.out_bias


def build_maps(train, test, user, item):
    item_ids = set(item["iid"].map(clean_id))
    for df in [train, test]:
        for col in ["item_seq_raw", "item_seq_dedup"]:
            if col in df.columns:
                for x in df[col].tolist():
                    item_ids.update(split_items(x))
    item_ids = sorted(x for x in item_ids if x)
    item2idx = {it: i + 1 for i, it in enumerate(item_ids)}
    target_ids = sorted(train["target_iid"].map(clean_id).unique())
    target2idx = {it: i for i, it in enumerate(target_ids)}
    target_item_idx = [item2idx[it] for it in target_ids]

    user = user.copy()
    user["uid"] = user["uid"].map(clean_id)
    ucols = [c for c in user.columns if c.startswith("u_cat_")]
    cards = []
    arrs = []
    for c in ucols:
        vals = user[c].fillna(-1).astype(int)
        mn = vals.min()
        if mn < 0:
            vals = vals - mn
        cards.append(int(vals.max()) + 1)
        arrs.append(vals.to_numpy(dtype=np.int64))
    user_arr = np.stack(arrs, axis=1).astype(np.int64)
    uid2urow = {u: i for i, u in enumerate(user["uid"].tolist())}
    return item2idx, target_ids, target2idx, target_item_idx, user_arr, uid2urow, cards


def eval_model(model, loader, device):
    model.eval()
    total, hit, n = 0.0, 0, 0
    with torch.no_grad():
        for seq, ux, y, lengths in loader:
            seq, ux, y, lengths = seq.to(device), ux.to(device), y.to(device), lengths.to(device)
            logits = model(seq, ux, lengths)
            a, b, c = ndcg_hit_from_logits(logits, y, 10)
            total += a; hit += b; n += c
    return total / max(n, 1), hit / max(n, 1)


def predict_model(model, test, item2idx, target_ids, user_arr, uid2urow, max_len, device, batch=512):
    model.eval()
    preds = {}
    rows = test.to_dict("records")
    with torch.no_grad():
        for st in range(0, len(rows), batch):
            chunk = rows[st:st+batch]
            seqs, ux, lens = [], [], []
            for r in chunk:
                raw = split_items(r.get("item_seq_raw"))
                seq = [item2idx[x] for x in raw[-max_len:] if x in item2idx]
                arr = np.zeros(max_len, dtype=np.int64)
                if seq:
                    arr[-len(seq):] = np.asarray(seq, dtype=np.int64)
                seqs.append(arr)
                ux.append(user_arr[uid2urow[clean_id(r["uid"])]])
                lens.append(len(seq))
            seqs = torch.tensor(np.stack(seqs), dtype=torch.long, device=device)
            ux = torch.tensor(np.stack(ux), dtype=torch.long, device=device)
            lens = torch.tensor(lens, dtype=torch.long, device=device)
            logits = model(seqs, ux, lens)
            top = torch.topk(logits, k=10, dim=1).indices.cpu().numpy()
            for r, idxs in zip(chunk, top):
                preds[clean_id(r["uid"])] = [target_ids[i] for i in idxs]
    return preds


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


def train_one(cfg, train_rows, val_orig_rows, val_test_rows, maps, sample, test, out_dir, device):
    item2idx, target_ids, target2idx, target_item_idx, user_arr, uid2urow, cards = maps
    max_len = cfg.get("max_len", 50)
    train_ds = RecDataset(train_rows, cfg["modes"], item2idx, target2idx, user_arr, uid2urow, max_len, RANDOM_STATE + cfg["seed"])
    val_orig = RecDataset(val_orig_rows, ["orig"], item2idx, target2idx, user_arr, uid2urow, max_len, RANDOM_STATE + 1)
    val_test = RecDataset(val_test_rows, ["testmix"], item2idx, target2idx, user_arr, uid2urow, max_len, RANDOM_STATE + 2)
    train_loader = DataLoader(train_ds, batch_size=cfg.get("batch", 512), shuffle=True, num_workers=2, pin_memory=True)
    vo_loader = DataLoader(val_orig, batch_size=1024, shuffle=False, num_workers=0)
    vt_loader = DataLoader(val_test, batch_size=1024, shuffle=False, num_workers=0)
    model = SeqClassifier(cfg["kind"], len(item2idx), len(target_ids), target_item_idx, cards,
                          dim=cfg.get("dim", 96), max_len=max_len, heads=cfg.get("heads", 4),
                          layers=cfg.get("layers", 2), dropout=cfg.get("dropout", 0.2)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 8e-4), weight_decay=cfg.get("wd", 0.03))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.get("ls", 0.02))
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
        score = 0.35 * nd_o + 0.65 * nd_t
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
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
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

    uids = train["uid"].to_numpy()
    uniq = np.array(pd.unique(train["uid"]))
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(uniq)
    val_u = set(uniq[:max(1, int(len(uniq) * 0.15))])
    tr_rows = train.loc[~train["uid"].isin(val_u)].to_dict("records")
    va_rows = train.loc[train["uid"].isin(val_u)].to_dict("records")
    print("train rows", len(tr_rows), "val rows", len(va_rows), flush=True)

    configs = [
        {"name": "output1100_sasrec_testmix", "kind": "sasrec", "modes": ["testmix", "last3", "empty"], "seed": 0, "epochs": 18, "dim": 96, "dropout": 0.22},
        {"name": "output1101_sasrec_shortheavy", "kind": "sasrec", "modes": ["testmix", "last3", "last1", "empty"], "seed": 1, "epochs": 18, "dim": 96, "dropout": 0.24},
        {"name": "output1102_gru_testmix", "kind": "gru", "modes": ["testmix", "last3", "empty"], "seed": 2, "epochs": 16, "dim": 96, "dropout": 0.18},
        {"name": "output1103_gru_shortheavy", "kind": "gru", "modes": ["testmix", "last3", "last1", "empty"], "seed": 3, "epochs": 16, "dim": 96, "dropout": 0.20},
        {"name": "output1105_sasrec_noempty", "kind": "sasrec", "modes": ["testmix", "last3", "last1"], "seed": 5, "epochs": 16, "dim": 96, "dropout": 0.22},
    ]
    preds = []
    for cfg in configs:
        preds.append(train_one(cfg, tr_rows, va_rows, va_rows, maps, sample, test, args.output_dir, device))
    fused = rank_fuse(preds[:4], [1.1, 1.0, 0.9, 0.9])
    save_submission(sample, fused, os.path.join(args.output_dir, "output1104_nn_rank_fusion", "A2.csv"))


if __name__ == "__main__":
    main()
