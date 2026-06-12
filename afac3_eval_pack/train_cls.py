"""
产品分类任务 - 训练脚本
GCN/GAT/MLP 基线，支持 early stopping 和模型保存
"""

import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix, diags
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────

def load_data():
    npz_path = os.path.join(DATA_ROOT, "A分类", "A分类", "A1.npz")
    data = np.load(npz_path, allow_pickle=True)
    adj = csr_matrix(
        (data["adj_data"], data["adj_indices"], data["adj_indptr"]),
        shape=tuple(data["adj_shape"]),
    )
    features = csr_matrix(
        (data["attr_data"], data["attr_indices"], data["attr_indptr"]),
        shape=tuple(data["attr_shape"]),
    )
    return adj, features, data["labels"], data["train_idx"], data["test_idx"]


# ─────────────────────────────────────────────
# 图预处理
# ─────────────────────────────────────────────

def preprocess_graph(adj, symmetrize=True, norm_mode="symmetric"):
    if symmetrize:
        adj = adj + adj.T
        adj.data = np.ones_like(adj.data)
    adj = adj + csr_matrix(np.eye(adj.shape[0]), dtype=np.float32)
    deg = np.array(adj.sum(axis=1)).flatten()
    if norm_mode == "symmetric":
        deg_inv_sqrt = np.power(deg, -0.5)
        deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
        D = diags(deg_inv_sqrt)
        adj_norm = D @ adj @ D
    else:
        deg_inv = np.power(deg, -1.0)
        deg_inv[np.isinf(deg_inv)] = 0.0
        D = diags(deg_inv)
        adj_norm = D @ adj
    return adj_norm


def to_sparse_tensor(mat):
    mat = mat.tocoo().astype(np.float32)
    idx = torch.from_numpy(np.vstack((mat.row, mat.col)).astype(np.int64))
    val = torch.from_numpy(mat.data)
    return torch.sparse_coo_tensor(idx, val, torch.Size(mat.shape))


# ─────────────────────────────────────────────
# 模型
# ─────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal Loss: 降低易分样本权重，聚焦难分样本"""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # 类别权重 (tensor, shape=[num_classes])
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class GCNLayer(nn.Module):
    def __init__(self, in_d, out_d):
        super().__init__()
        self.w = nn.Parameter(torch.empty(in_d, out_d))
        nn.init.xavier_uniform_(self.w)
        self.b = nn.Parameter(torch.zeros(out_d))

    def forward(self, x, adj):
        return torch.sparse.mm(adj, x @ self.w) + self.b


class GCN(nn.Module):
    def __init__(self, in_d, hid, out_d, n_layers=2, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.drop = dropout
        self.layers.append(GCNLayer(in_d, hid))
        for _ in range(n_layers - 2):
            self.layers.append(GCNLayer(hid, hid))
        self.layers.append(GCNLayer(hid, out_d))

    def forward(self, x, adj):
        for l in self.layers[:-1]:
            x = F.relu(l(x, adj))
            x = F.dropout(x, self.drop, self.training)
        return self.layers[-1](x, adj)


class MLP(nn.Module):
    def __init__(self, in_d, hid, out_d, n_layers=2, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.drop = dropout
        self.layers.append(nn.Linear(in_d, hid))
        for _ in range(n_layers - 2):
            self.layers.append(nn.Linear(hid, hid))
        self.layers.append(nn.Linear(hid, out_d))

    def forward(self, x, adj=None):
        for l in self.layers[:-1]:
            x = F.relu(l(x))
            x = F.dropout(x, self.drop, self.training)
        return self.layers[-1](x)


# ─────────────────────────────────────────────
# 训练
# ─────────────────────────────────────────────

def train():
    print("加载数据...")
    adj, features, labels, train_idx, test_idx = load_data()

    # 超参数
    config = {
        "model_type": "GCN",
        "hidden_dim": 256,
        "num_layers": 3,
        "dropout": 0.5,
        "lr": 0.01,
        "weight_decay": 5e-4,
        "epochs": 300,
        "patience": 50,
        "feature_norm": True,
        "symmetrize": True,
        "norm_mode": "symmetric",
    }

    print(f"配置: {json.dumps(config, indent=2)}")
    print(f"设备: {DEVICE}")

    # 特征预处理
    if config["feature_norm"]:
        feat = StandardScaler().fit_transform(features.toarray())
    else:
        feat = features.toarray()
    feat_t = torch.from_numpy(feat.astype(np.float32)).to(DEVICE)

    # 图预处理
    adj_norm = preprocess_graph(adj, config["symmetrize"], config["norm_mode"])
    adj_t = to_sparse_tensor(adj_norm).to(DEVICE)
    labels_t = torch.from_numpy(labels.astype(np.int64)).to(DEVICE)

    # 训练/验证划分
    idx = np.array(train_idx)
    trn, val = train_test_split(idx, test_size=0.2, random_state=42, stratify=labels[idx])
    trn_mask = torch.zeros(len(labels), dtype=torch.bool, device=DEVICE)
    val_mask = torch.zeros(len(labels), dtype=torch.bool, device=DEVICE)
    trn_mask[trn] = True
    val_mask[val] = True

    # 类别权重 (基于有效样本数: w = (1-beta) / (1-beta^n))
    cnt = np.bincount(labels[trn], minlength=10)
    beta = 0.9999
    effective_num = 1.0 - np.power(beta, cnt)
    w = (1.0 - beta) / (effective_num + 1e-6)
    w = w / w.sum() * 10
    w_t = torch.from_numpy(w.astype(np.float32)).to(DEVICE)
    print(f"类别分布: {cnt.tolist()}")
    print(f"有效样本权重: {w.round(3).tolist()}")

    # 模型
    if config["model_type"] == "GCN":
        model = GCN(767, config["hidden_dim"], 10, config["num_layers"], config["dropout"]).to(DEVICE)
    else:
        model = MLP(767, config["hidden_dim"], 10, config["num_layers"], config["dropout"]).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, config["epochs"])

    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型: {config['model_type']}, 参数量: {n_params:,}")
    print(f"训练: {len(trn)}, 验证: {len(val)}, 测试: {len(test_idx)}")

    # 训练循环
    best_acc = 0
    patience_cnt = 0
    best_state = None
    history = {"train_loss": [], "val_acc": []}

    print("\n开始训练...")
    for ep in range(1, config["epochs"] + 1):
        t0 = time.time()

        # 训练
        model.train()
        opt.zero_grad()
        out = model(feat_t, adj_t)
        loss = F.cross_entropy(out[trn_mask], labels_t[trn_mask], weight=w_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()

        # 验证
        model.eval()
        with torch.no_grad():
            out = model(feat_t, adj_t)
            pred = out[val_mask].argmax(1)
            acc = (pred == labels_t[val_mask]).float().mean().item()

        history["train_loss"].append(loss.item())
        history["val_acc"].append(acc)

        if acc > best_acc:
            best_acc = acc
            patience_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1

        if ep % 20 == 0 or ep == 1:
            lr = sched.get_last_lr()[0]
            print(f"  Ep {ep:3d} | loss={loss.item():.4f} | val_acc={acc:.4f} | lr={lr:.5f} | {time.time()-t0:.1f}s")

        if patience_cnt >= config["patience"]:
            print(f"  早停于 epoch {ep}")
            break

    print(f"\n最优验证准确率: {best_acc:.4f}")

    # 保存模型
    save_dir = os.path.join(DATA_ROOT, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    model.load_state_dict(best_state)
    torch.save({
        "model_state": best_state,
        "config": config,
        "best_val_acc": best_acc,
        "history": history,
    }, os.path.join(save_dir, "cls_best.pt"))
    print(f"模型已保存至 checkpoints/cls_best.pt")

    # 保存训练历史
    with open(os.path.join(save_dir, "cls_history.json"), "w") as f:
        json.dump(history, f)


if __name__ == "__main__":
    train()
