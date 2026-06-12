"""
Task A: Degree-weighted ensemble (GCN + LP)
High-degree nodes -> trust GCN more
Low-degree nodes -> trust LP more
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix, diags, eye as speye
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from train_cls import GCN, MLP, load_data, preprocess_graph, to_sparse_tensor

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_label_propagation(adj, labels, train_idx, test_idx, num_iters=30, alpha=0.2):
    """Sparse graph label propagation"""
    N = adj.shape[0]
    num_classes = 10

    adj_sym = adj + adj.T
    adj_sym.data = np.ones_like(adj_sym.data, dtype=np.float32)
    adj_sym = adj_sym + speye(N, format="csr", dtype=np.float32)

    deg = np.array(adj_sym.sum(axis=1)).flatten()
    deg_inv = np.power(deg, -1.0)
    deg_inv[np.isinf(deg_inv)] = 0.0
    T = diags(deg_inv) @ adj_sym

    t0 = time.time()

    Y = np.ones((N, num_classes), dtype=np.float64) / num_classes
    Y_init = Y.copy()

    for idx in train_idx:
        Y[idx] = 0
        Y[idx, labels[idx]] = 1.0
    Y_init[:] = Y

    for it in range(num_iters):
        Y_new = (1 - alpha) * (T @ Y) + alpha * Y_init
        for idx in train_idx:
            Y_new[idx] = 0
            Y_new[idx, labels[idx]] = 1.0
        diff = np.abs(Y_new - Y).max()
        Y = Y_new
        if (it + 1) % 10 == 0:
            print(f"  LP iter {it+1}: max_diff={diff:.6f}")
        if diff < 1e-6:
            print(f"  LP converged at iter {it+1}")
            break

    elapsed = time.time() - t0
    row_sums = Y.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    lp_probs = Y / row_sums

    # No-leakage validation
    trn_sub, val_sub = train_test_split(
        train_idx, test_size=0.2, random_state=42, stratify=labels[train_idx]
    )

    Y_val = np.ones((N, num_classes), dtype=np.float64) / num_classes
    Y_val_init = Y_val.copy()
    for idx in trn_sub:
        Y_val[idx] = 0
        Y_val[idx, labels[idx]] = 1.0
    Y_val_init[:] = Y_val
    for it in range(num_iters):
        Y_val_new = (1 - alpha) * (T @ Y_val) + alpha * Y_val_init
        for idx in trn_sub:
            Y_val_new[idx] = 0
            Y_val_new[idx, labels[idx]] = 1.0
        Y_val = Y_val_new
    row_sums_val = Y_val.sum(axis=1, keepdims=True)
    row_sums_val[row_sums_val == 0] = 1
    lp_val_probs = Y_val / row_sums_val
    val_pred = lp_val_probs[val_sub].argmax(axis=1)
    val_acc = (val_pred == labels[val_sub]).mean()

    print(f"[LP] Done | {elapsed:.1f}s | val_acc={val_acc:.4f} (no-leak)")
    return lp_probs, lp_val_probs, val_acc, val_sub


def run_gcn_predict(config, adj, features, labels, train_idx, test_idx):
    """GCN prediction"""
    if config["feature_norm"]:
        feat = StandardScaler().fit_transform(features.toarray())
    else:
        feat = features.toarray()
    feat_t = torch.from_numpy(feat.astype(np.float32)).to(DEVICE)

    adj_norm = preprocess_graph(adj, config["symmetrize"], config["norm_mode"])
    adj_t = to_sparse_tensor(adj_norm).to(DEVICE)

    ckpt_path = os.path.join(DATA_ROOT, "checkpoints", "cls_best.pt")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    if config["model_type"] == "GCN":
        model = GCN(767, config["hidden_dim"], 10, config["num_layers"], config["dropout"]).to(DEVICE)
    else:
        model = MLP(767, config["hidden_dim"], 10, config["num_layers"], config["dropout"]).to(DEVICE)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        out = model(feat_t, adj_t)
        gcn_all_probs = F.softmax(out, dim=1).cpu().numpy()

    trn_sub, val_sub = train_test_split(
        train_idx, test_size=0.2, random_state=42, stratify=labels[train_idx]
    )
    val_pred = gcn_all_probs[val_sub].argmax(axis=1)
    val_acc = (val_pred == labels[val_sub]).mean()

    print(f"[GCN] val_acc={val_acc:.4f}")
    return gcn_all_probs, val_acc


def predict():
    print("=" * 60)
    print("  Degree-Weighted Ensemble (GCN + LP)")
    print("=" * 60)

    adj, features, labels, train_idx, test_idx = load_data()

    # Load GCN config
    ckpt_path = os.path.join(DATA_ROOT, "checkpoints", "cls_best.pt")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    config = ckpt["config"]
    print(f"GCN config: val_acc={ckpt['best_val_acc']:.4f}")

    # 1. GCN probabilities
    gcn_all_probs, gcn_val_acc = run_gcn_predict(config, adj, features, labels, train_idx, test_idx)

    # 2. LP probabilities
    lp_all_probs, lp_val_probs, lp_val_acc, val_sub = run_label_propagation(adj, labels, train_idx, test_idx)

    # 3. Compute node degrees
    adj_sym = adj + adj.T
    adj_sym.data = np.ones_like(adj_sym.data)
    deg = np.array(adj_sym.sum(axis=1)).flatten()
    median_deg = np.median(deg)
    std_deg = deg.std()
    print(f"\nDegree stats: median={median_deg:.1f}, std={std_deg:.1f}, min={deg.min():.0f}, max={deg.max():.0f}")

    # 4. Degree-weighted ensemble on validation set
    print("\n[Validation] Degree-weighted ensemble strategies:")

    # Use no-leak LP probs for validation
    strategies = {
        "gcn_only": lambda g, l, d: g,
        "lp_only": lambda g, l, d: l,
        "average": lambda g, l, d: 0.5 * g + 0.5 * l,
        "deg_sigmoid": lambda g, l, d: _deg_weighted(g, l, d, median_deg, std_deg),
        "deg_linear": lambda g, l, d: _deg_linear(g, l, d, median_deg, std_deg),
        "deg_sigmoid2": lambda g, l, d: _deg_weighted(g, l, d, median_deg, std_deg * 2),
        "deg_sigmoid3": lambda g, l, d: _deg_weighted(g, l, d, median_deg, std_deg * 0.5),
        "gcn_heavy": lambda g, l, d: 0.7 * g + 0.3 * l,
        "lp_heavy": lambda g, l, d: 0.3 * g + 0.7 * l,
        "gcn_vheavy": lambda g, l, d: 0.8 * g + 0.2 * l,
        "lp_0208": lambda g, l, d: 0.2 * g + 0.8 * l,
        "lp_0109": lambda g, l, d: 0.1 * g + 0.9 * l,
        "lp_025": lambda g, l, d: 0.25 * g + 0.75 * l,
        "lp_035": lambda g, l, d: 0.35 * g + 0.65 * l,
        "lp_040": lambda g, l, d: 0.4 * g + 0.6 * l,
        "deg_lp_heavy": lambda g, l, d: _deg_lp_heavy(g, l, d, median_deg, std_deg),
        "deg_lp_heavy2": lambda g, l, d: _deg_lp_heavy2(g, l, d, median_deg, std_deg),
        "deg_lp_heavy3": lambda g, l, d: _deg_lp_heavy3(g, l, d, median_deg, std_deg),
    }

    best_strategy = None
    best_val_acc = 0
    for name, fn in strategies.items():
        probs = fn(gcn_all_probs, lp_val_probs, deg)
        val_pred = probs[val_sub].argmax(axis=1)
        vac = (val_pred == labels[val_sub]).mean()
        print(f"  {name:15s}: {vac:.4f}")
        if vac > best_val_acc:
            best_val_acc = vac
            best_strategy = name

    print(f"\nBest: {best_strategy} ({best_val_acc:.4f})")

    # 5. Per-class threshold optimization on validation
    fn = strategies[best_strategy]
    val_probs = fn(gcn_all_probs, lp_val_probs, deg)
    val_probs_sub = val_probs[val_sub]
    val_labels_sub = labels[val_sub]

    thresholds = np.ones(10)
    for cls in range(10):
        for boost in [1.0, 1.02, 1.05, 1.08, 1.1]:
            t = thresholds.copy()
            t[cls] = boost
            adj_probs = val_probs_sub * t
            acc = (adj_probs.argmax(1) == val_labels_sub).mean()
            if acc > best_val_acc:
                best_val_acc = acc
                thresholds[cls] = boost

    print(f"After threshold opt: {best_val_acc:.4f}")
    print(f"Thresholds: {thresholds.round(3).tolist()}")

    # 6. Generate test predictions with best strategy + thresholds
    final_probs = fn(gcn_all_probs, lp_all_probs, deg)
    final_probs_adj = final_probs * thresholds
    test_pred = final_probs_adj[test_idx].argmax(axis=1)

    # 6. Save
    sample = pd.read_csv(os.path.join(DATA_ROOT, "A分类", "A分类", "sample_submission.csv"))
    sample["label"] = test_pred
    out_path = os.path.join(DATA_ROOT, "A1.csv")
    sample.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    print(f"Rows: {len(sample)}")
    print(f"Label dist: {np.bincount(test_pred, minlength=10).tolist()}")

    return best_val_acc


def _deg_weighted(gcn_probs, lp_probs, deg, median_deg, std_deg):
    """Sigmoid degree weighting: high deg -> GCN, low deg -> LP"""
    weight = 1.0 / (1.0 + np.exp(-(deg - median_deg) / max(std_deg, 1.0)))
    weight = weight.reshape(-1, 1)
    return weight * gcn_probs + (1 - weight) * lp_probs


def _deg_linear(gcn_probs, lp_probs, deg, median_deg, std_deg):
    """Linear degree weighting"""
    weight = np.clip((deg - median_deg + std_deg) / (2 * std_deg), 0, 1)
    weight = weight.reshape(-1, 1)
    return weight * gcn_probs + (1 - weight) * lp_probs


def _deg_lp_heavy(gcn_probs, lp_probs, deg, median_deg, std_deg):
    """Degree-aware LP-heavy: high deg -> more GCN, but always LP-biased"""
    weight = 1.0 / (1.0 + np.exp(-(deg - median_deg) / max(std_deg, 1.0)))
    weight = weight.reshape(-1, 1)
    # Scale: max GCN weight is 0.4, min is 0.1
    gcn_w = 0.1 + 0.3 * weight
    return gcn_w * gcn_probs + (1 - gcn_w) * lp_probs


def _deg_lp_heavy2(gcn_probs, lp_probs, deg, median_deg, std_deg):
    weight = 1.0 / (1.0 + np.exp(-(deg - median_deg) / max(std_deg, 1.0)))
    weight = weight.reshape(-1, 1)
    gcn_w = 0.05 + 0.25 * weight
    return gcn_w * gcn_probs + (1 - gcn_w) * lp_probs


def _deg_lp_heavy3(gcn_probs, lp_probs, deg, median_deg, std_deg):
    weight = 1.0 / (1.0 + np.exp(-(deg - median_deg) / max(std_deg, 1.0)))
    weight = weight.reshape(-1, 1)
    gcn_w = 0.15 + 0.35 * weight
    return gcn_w * gcn_probs + (1 - gcn_w) * lp_probs


if __name__ == "__main__":
    predict()
