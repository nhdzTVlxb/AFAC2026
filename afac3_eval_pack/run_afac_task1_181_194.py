#!/usr/bin/env python3
import argparse
import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix, diags, eye as speye
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(root):
    base = Path(root) / "A分类" / "A分类"
    data = np.load(base / "A1.npz", allow_pickle=True)
    adj = csr_matrix(
        (data["adj_data"], data["adj_indices"], data["adj_indptr"]),
        shape=tuple(data["adj_shape"]),
    )
    features = csr_matrix(
        (data["attr_data"], data["attr_indices"], data["attr_indptr"]),
        shape=tuple(data["attr_shape"]),
    )
    sample = pd.read_csv(base / "sample_submission.csv")
    return adj, features, data["labels"], data["train_idx"], data["test_idx"], sample


def preprocess_graph(adj, symmetrize=True, norm_mode="symmetric"):
    if symmetrize:
        adj = adj + adj.T
        adj.data = np.ones_like(adj.data)
    adj = adj + speye(adj.shape[0], format="csr", dtype=np.float32)
    deg = np.array(adj.sum(axis=1)).flatten()
    if norm_mode == "symmetric":
        deg_inv_sqrt = np.power(deg, -0.5)
        deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
        d = diags(deg_inv_sqrt)
        return d @ adj @ d
    deg_inv = np.power(deg, -1.0)
    deg_inv[np.isinf(deg_inv)] = 0.0
    return diags(deg_inv) @ adj


def to_sparse_tensor(mat):
    mat = mat.tocoo().astype(np.float32)
    idx = torch.from_numpy(np.vstack((mat.row, mat.col)).astype(np.int64))
    val = torch.from_numpy(mat.data)
    return torch.sparse_coo_tensor(idx, val, torch.Size(mat.shape)).coalesce()


def preprocess_features(features, mode):
    if mode == "standard":
        return StandardScaler().fit_transform(features.toarray()).astype(np.float32)
    if mode == "binary":
        f = features.copy().astype(np.float32)
        f.data = np.ones_like(f.data, dtype=np.float32)
        return f.toarray().astype(np.float32)
    if mode == "row":
        f = features.tocsr(copy=True).astype(np.float32)
        rowsum = np.array(f.sum(1)).flatten()
        inv = np.power(rowsum, -1, where=rowsum != 0)
        inv[rowsum == 0] = 0
        return (diags(inv) @ f).toarray().astype(np.float32)
    return features.toarray().astype(np.float32)


class GCNLayer(nn.Module):
    def __init__(self, in_d, out_d):
        super().__init__()
        self.w = nn.Parameter(torch.empty(in_d, out_d))
        self.b = nn.Parameter(torch.zeros(out_d))
        nn.init.xavier_uniform_(self.w)

    def forward(self, x, adj):
        return torch.sparse.mm(adj, x @ self.w) + self.b


class GCN(nn.Module):
    def __init__(self, in_d, hid, out_d, n_layers=3, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList([GCNLayer(in_d, hid)])
        for _ in range(n_layers - 2):
            self.layers.append(GCNLayer(hid, hid))
        self.layers.append(GCNLayer(hid, out_d))
        self.drop = dropout

    def forward(self, x, adj):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x, adj))
            x = F.dropout(x, self.drop, self.training)
        return self.layers[-1](x, adj)


class GCNII(nn.Module):
    def __init__(self, in_d, hid, out_d, n_layers=8, dropout=0.3, alpha=0.15):
        super().__init__()
        self.alpha = alpha
        self.drop = dropout
        self.inp = nn.Linear(in_d, hid)
        self.layers = nn.ModuleList([nn.Linear(hid, hid) for _ in range(max(1, n_layers - 1))])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hid) for _ in range(max(1, n_layers - 1))])
        self.out = nn.Linear(hid, out_d)

    def forward(self, x, adj):
        h0 = F.relu(self.inp(x))
        h = F.dropout(h0, self.drop, self.training)
        for layer, bn in zip(self.layers, self.bns):
            support = (1 - self.alpha) * torch.sparse.mm(adj, h) + self.alpha * h0
            h2 = layer(support)
            if h2.shape == h.shape:
                h2 = h2 + h
            h = F.dropout(F.relu(bn(h2)), self.drop, self.training)
        return self.out(h)


def make_model(config):
    if config["model_type"] == "gcnii":
        return GCNII(767, config["hidden_dim"], 10, config["num_layers"], config["dropout"], config.get("alpha", 0.15))
    return GCN(767, config["hidden_dim"], 10, config["num_layers"], config["dropout"])


def effective_class_weights(labels, trn):
    cnt = np.bincount(labels[trn], minlength=10)
    beta = 0.9999
    eff = 1.0 - np.power(beta, cnt)
    w = (1.0 - beta) / (eff + 1e-6)
    w = w / w.sum() * 10
    return torch.from_numpy(w.astype(np.float32)).to(DEVICE), cnt


def train_one(adj_t, feat_t, labels_t, labels, train_idx, config):
    trn, val = train_test_split(
        np.array(train_idx),
        test_size=config["val_ratio"],
        random_state=config["seed"],
        stratify=labels[train_idx],
    )
    trn_t = torch.from_numpy(trn.astype(np.int64)).to(DEVICE)
    val_t = torch.from_numpy(val.astype(np.int64)).to(DEVICE)
    w_t, cnt = effective_class_weights(labels, trn)
    model = make_model(config).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, config["epochs"])
    best_acc = 0.0
    best_state = None
    bad = 0
    for ep in range(1, config["epochs"] + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        out = model(feat_t, adj_t)
        loss = F.cross_entropy(out[trn_t], labels_t[trn_t], weight=w_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            logits = model(feat_t, adj_t)
            acc = (logits[val_t].argmax(1) == labels_t[val_t]).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if ep == 1 or ep % 20 == 0:
            print(f"  ep={ep:03d} loss={loss.item():.4f} val={acc:.4f} best={best_acc:.4f}")
        if bad >= config["patience"]:
            break
    model.load_state_dict(best_state)
    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(feat_t, adj_t), dim=1).cpu().numpy()
    return model, probs, best_acc, trn, val


def run_lp(adj, labels, train_idx, val_sub, alpha=0.2, num_iters=30):
    n = adj.shape[0]
    adj_sym = adj + adj.T
    adj_sym.data = np.ones_like(adj_sym.data, dtype=np.float32)
    adj_sym = adj_sym + speye(n, format="csr", dtype=np.float32)
    deg = np.array(adj_sym.sum(axis=1)).flatten()
    deg_inv = np.power(deg, -1.0)
    deg_inv[np.isinf(deg_inv)] = 0.0
    trans = diags(deg_inv) @ adj_sym

    y = np.ones((n, 10), dtype=np.float64) / 10
    for idx in train_idx:
        y[idx] = 0
        y[idx, labels[idx]] = 1
    y_init = y.copy()
    for _ in range(num_iters):
        y_new = (1 - alpha) * (trans @ y) + alpha * y_init
        for idx in train_idx:
            y_new[idx] = 0
            y_new[idx, labels[idx]] = 1
        y = y_new
    y = y / np.maximum(y.sum(axis=1, keepdims=True), 1e-12)
    return y


def deg_lp_heavy(gcn_probs, lp_probs, deg):
    median_deg = np.median(deg)
    std_deg = deg.std()
    weight = 1.0 / (1.0 + np.exp(-(deg - median_deg) / max(std_deg, 1.0)))
    weight = weight.reshape(-1, 1)
    gcn_w = 0.1 + 0.3 * weight
    return gcn_w * gcn_probs + (1 - gcn_w) * lp_probs


def apply_best_fusion(model_probs, lp_all, lp_val, labels, train_idx, val_sub, adj):
    adj_sym = adj + adj.T
    adj_sym.data = np.ones_like(adj_sym.data)
    deg = np.array(adj_sym.sum(axis=1)).flatten()
    strategies = {
        "gcn_only": lambda g, l: g,
        "average": lambda g, l: 0.5 * g + 0.5 * l,
        "lp_heavy": lambda g, l: 0.3 * g + 0.7 * l,
        "lp_0109": lambda g, l: 0.1 * g + 0.9 * l,
        "deg_lp_heavy": lambda g, l: deg_lp_heavy(g, l, deg),
    }
    best_name, best_acc, best_probs = None, -1, None
    for name, fn in strategies.items():
        probs = fn(model_probs, lp_val)
        acc = (probs[val_sub].argmax(1) == labels[val_sub]).mean()
        if acc > best_acc:
            best_name, best_acc, best_probs = name, acc, probs
    thresholds = np.ones(10)
    for cls in range(10):
        for boost in [1.0, 1.02, 1.05, 1.08, 1.1, 1.15]:
            t = thresholds.copy()
            t[cls] = boost
            acc = ((best_probs[val_sub] * t).argmax(1) == labels[val_sub]).mean()
            if acc > best_acc:
                best_acc = acc
                thresholds[cls] = boost
    final_probs = strategies[best_name](model_probs, lp_all) * thresholds
    return final_probs, best_acc, best_name, thresholds


def experiment_configs():
    base = {
        "model_type": "gcn", "hidden_dim": 256, "num_layers": 3, "dropout": 0.5,
        "lr": 0.01, "weight_decay": 5e-4, "epochs": 300, "patience": 50,
        "feature_norm": "standard", "symmetrize": True, "norm_mode": "symmetric",
        "val_ratio": 0.2, "seed": 42,
    }
    exps = []
    def add(num, name, **kw):
        cfg = base.copy()
        cfg.update(kw)
        exps.append((num, name, cfg))
    add(181, "afac original")
    add(182, "afac binary feature", feature_norm="binary")
    add(183, "afac dropout03", dropout=0.3)
    add(184, "afac wd1e-3", weight_decay=1e-3)
    add(185, "afac binary dropout03", feature_norm="binary", dropout=0.3)
    add(186, "afac binary wd1e-3", feature_norm="binary", weight_decay=1e-3)
    add(187, "afac dropout03 wd1e-3", dropout=0.3, weight_decay=1e-3)
    add(188, "afac binary dropout03 wd1e-3", feature_norm="binary", dropout=0.3, weight_decay=1e-3)
    add(189, "afac valratio010", val_ratio=0.1)
    add(190, "afac valratio005", val_ratio=0.05)
    add(191, "gcnii alpha015 lp", model_type="gcnii", alpha=0.15, dropout=0.3, weight_decay=1e-3, num_layers=8)
    add(192, "gcnii alpha015 binary lp", model_type="gcnii", alpha=0.15, dropout=0.3, weight_decay=1e-3, num_layers=8, feature_norm="binary")
    add(193, "gcn plus gcnii avg lp", ensemble="gcn_gcnii")
    add(194, "afac gcn multiseed avg lp", ensemble="multiseed")
    return exps


def run_one(root, out_root, exp_num, name, cfg, data, fixed_a2):
    adj, features, labels, train_idx, test_idx, sample = data
    out_dir = out_root / f"output{exp_num}"
    sub_dir = out_dir / "submission"
    sub_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n===== output{exp_num}: {name} =====")
    print(json.dumps(cfg, indent=2))

    if cfg.get("ensemble") == "gcn_gcnii":
        cfg1 = cfg.copy(); cfg1.pop("ensemble")
        cfg2 = cfg1.copy(); cfg2.update({"model_type": "gcnii", "num_layers": 8, "dropout": 0.3, "weight_decay": 1e-3, "alpha": 0.15})
        probs_list, accs, val_sub = [], [], None
        for c in [cfg1, cfg2]:
            feat = torch.from_numpy(preprocess_features(features, c["feature_norm"])).to(DEVICE)
            adj_t = to_sparse_tensor(preprocess_graph(adj, c["symmetrize"], c["norm_mode"])).to(DEVICE)
            _, probs, acc, trn, val = train_one(adj_t, feat, torch.from_numpy(labels.astype(np.int64)).to(DEVICE), labels, train_idx, c)
            probs_list.append(probs); accs.append(acc); val_sub = val
        model_probs = 0.5 * probs_list[0] + 0.5 * probs_list[1]
        train_for_lp = np.setdiff1d(train_idx, val_sub)
        base_acc = max(accs)
    elif cfg.get("ensemble") == "multiseed":
        probs_list, accs, val_sub = [], [], None
        for seed in [42, 2026, 3407]:
            c = cfg.copy(); c.pop("ensemble"); c["seed"] = seed
            feat = torch.from_numpy(preprocess_features(features, c["feature_norm"])).to(DEVICE)
            adj_t = to_sparse_tensor(preprocess_graph(adj, c["symmetrize"], c["norm_mode"])).to(DEVICE)
            _, probs, acc, trn, val = train_one(adj_t, feat, torch.from_numpy(labels.astype(np.int64)).to(DEVICE), labels, train_idx, c)
            probs_list.append(probs); accs.append(acc)
            if seed == 42:
                val_sub = val
        model_probs = np.mean(probs_list, axis=0)
        train_for_lp = np.setdiff1d(train_idx, val_sub)
        base_acc = max(accs)
    else:
        feat = torch.from_numpy(preprocess_features(features, cfg["feature_norm"])).to(DEVICE)
        adj_t = to_sparse_tensor(preprocess_graph(adj, cfg["symmetrize"], cfg["norm_mode"])).to(DEVICE)
        _, model_probs, base_acc, trn, val_sub = train_one(adj_t, feat, torch.from_numpy(labels.astype(np.int64)).to(DEVICE), labels, train_idx, cfg)
        train_for_lp = trn

    lp_all = run_lp(adj, labels, train_idx, val_sub)
    lp_val = run_lp(adj, labels, train_for_lp, val_sub)
    final_probs, fusion_acc, strategy, thresholds = apply_best_fusion(model_probs, lp_all, lp_val, labels, train_idx, val_sub, adj)
    pred = final_probs[test_idx].argmax(axis=1)
    a1 = sample.copy()
    a1["label"] = pred
    a1.to_csv(sub_dir / "A1.csv", index=False)
    if fixed_a2 and Path(fixed_a2).exists():
        import shutil
        shutil.copy(fixed_a2, sub_dir / "A2.csv")
        with zipfile.ZipFile(out_dir / "prediction.zip", "w", zipfile.ZIP_DEFLATED) as z:
            z.write(sub_dir / "A1.csv", "A1.csv")
            z.write(sub_dir / "A2.csv", "A2.csv")
    metrics = {
        "output": f"output{exp_num}",
        "experiment": name,
        "base_model_acc": float(base_acc),
        "fusion_acc": float(fusion_acc),
        "strategy": strategy,
        "thresholds": thresholds.tolist(),
        "label_dist": np.bincount(pred, minlength=10).tolist(),
        "zip": (out_dir / "prediction.zip").exists(),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out_root", default="../v1/framework")
    ap.add_argument("--fixed_a2", default="../v1/framework/output_afac3/submission/A2.csv")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    out_root = Path(args.out_root).resolve()
    data = load_data(root)
    rows = []
    for num, name, cfg in experiment_configs():
        rows.append(run_one(root, out_root, num, name, cfg, data, args.fixed_a2))
    summary = out_root / "ablation_181_194_summary.md"
    lines = [
        "# AFAC Task1 Trick Transfer 181-194 Summary",
        "",
        "| output | experiment | base_model_acc | fusion_acc | strategy | zip |",
        "|---|---|---:|---:|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['output']} | {r['experiment']} | {r['base_model_acc']:.6f} | {r['fusion_acc']:.6f} | {r['strategy']} | {'yes' if r['zip'] else 'no'} |")
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.read_text())


if __name__ == "__main__":
    main()
