#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys


HASHSEED_ENV = {
    "PYTHONHASHSEED": "0",
}
THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "OMP_DYNAMIC",
    "MKL_DYNAMIC",
)
os.environ["PYTHONUNBUFFERED"] = "1"

# PYTHONHASHSEED is read only when Python starts. Re-exec before importing
# NumPy/SciPy when the caller did not provide the required seed.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ.update(HASHSEED_ENV)
    os.execv(sys.executable, [sys.executable] + sys.argv)

os.environ.update(HASHSEED_ENV)

import json
import random
import numpy as np
import pandas as pd
import gc

from scipy.sparse import csr_matrix, diags, eye as speye

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier

import warnings
warnings.filterwarnings("ignore")


GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)


# ─────────────────────────────────────────────
# 路径配置
# ─────────────────────────────────────────────

VERSION_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(VERSION_DIR, "..", "..", ".."))

DATA_ROOT = os.environ.get(
    "AFAC_T1_DATA_ROOT",
    os.path.join(PROJECT_ROOT, "Dataset", "A分类", "A分类"),
)
DATA_ROOT1 = os.environ.get(
    "AFAC_T1_OUTPUT_DIR",
    os.path.join(VERSION_DIR, "output"),
)

OUTPUT_NAME = "A1.csv"
SUMMARY_NAME = "summary_baseline.json"
OOF_ARTIFACT_NAME = "base_oof_probs.npz"
META_MAX_ITER = 5000


# ─────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────

def load_data():
    npz_path = os.path.join(DATA_ROOT, "A1.npz")
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
# 特征工程
# ─────────────────────────────────────────────

def symmetrize_adj(adj):
    adj = adj + adj.T
    adj.data = np.ones_like(adj.data)
    return adj.tocsr()


def normalize_adj(adj):
    adj_sym = symmetrize_adj(adj)
    adj_sym = adj_sym + speye(adj_sym.shape[0], format="csr")

    deg = np.asarray(adj_sym.sum(axis=1)).reshape(-1)
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0

    D_inv_sqrt = diags(deg_inv_sqrt)

    return (D_inv_sqrt @ adj_sym @ D_inv_sqrt).tocsr()


def compute_svd_adj(adj, n_dim=32):
    """
    对邻接矩阵 A 进行 SVD 分解，获取结构嵌入。
    """
    print("  计算 Adjacency SVD (Structural Embedding)...")

    adj_sym = symmetrize_adj(adj)

    try:
        svd = TruncatedSVD(
            n_components=n_dim,
            random_state=42,
            n_iter=10
        )

        vecs = svd.fit_transform(adj_sym)
        vecs = normalize(vecs, norm="l2", axis=1)

        return vecs.astype(np.float32)

    except Exception as e:
        print(f"    SVD(A) failed: {e}")
        return np.zeros((adj.shape[0], n_dim), dtype=np.float32)


def compute_pagerank(adj, alpha=0.15):
    n = adj.shape[0]
    adj_sym = symmetrize_adj(adj)

    deg = np.asarray(adj_sym.sum(axis=1)).reshape(-1)
    deg_inv = np.power(deg, -1.0)
    deg_inv[np.isinf(deg_inv)] = 0.0

    transition = diags(deg_inv) @ adj_sym

    pr = np.ones(n, dtype=np.float32) / n

    for _ in range(30):
        pr = (1 - alpha) * (transition.T @ pr) + alpha / n

    return pr.reshape(-1, 1)


def compute_2hop_deg(adj):
    adj_sym = symmetrize_adj(adj)
    deg = np.asarray(adj_sym.sum(axis=1)).reshape(-1)

    hop2 = adj_sym @ deg

    return hop2.reshape(-1, 1)


# ─────────────────────────────────────────────
# 通用工具
# ─────────────────────────────────────────────

def softmax_np(z, temperature=1.0):
    z = z.astype(np.float64) / temperature
    z = z - z.max(axis=1, keepdims=True)

    ez = np.exp(z)
    out = ez / ez.sum(axis=1, keepdims=True)

    return out.astype(np.float32)


def decision_to_proba_full(clf, x_dense, num_classes, temperature=1.0):
    """
    RidgeClassifier 没有 predict_proba。
    使用 decision_function + softmax 转成 stacking 可用概率。
    """
    scores = clf.decision_function(x_dense)

    if scores.ndim == 1:
        scores = np.vstack([-scores, scores]).T

    probs_local = softmax_np(scores, temperature=temperature)

    out = np.zeros((x_dense.shape[0], num_classes), dtype=np.float32)

    for i, cls in enumerate(clf.classes_):
        out[:, cls] = probs_local[:, i]

    row_sums = out.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0

    out = out / row_sums

    return out.astype(np.float32)


def decision_to_proba_full_sparse(clf, x_sparse, num_classes, temperature=1.0):
    """
    Sparse RidgeClassifier 的 decision_function + softmax。
    """
    scores = clf.decision_function(x_sparse)

    if scores.ndim == 1:
        scores = np.vstack([-scores, scores]).T

    probs_local = softmax_np(scores, temperature=temperature)

    out = np.zeros((x_sparse.shape[0], num_classes), dtype=np.float32)

    for i, cls in enumerate(clf.classes_):
        out[:, cls] = probs_local[:, i]

    row_sums = out.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0

    out = out / row_sums

    return out.astype(np.float32)


# ─────────────────────────────────────────────
# 模型定义
# ─────────────────────────────────────────────

def get_smooth_labels(labels, n_classes, idx, epsilon=0.1):
    y = np.full(
        (len(idx), n_classes),
        epsilon / (n_classes - 1),
        dtype=np.float32
    )

    for i, node in enumerate(idx):
        y[i, labels[node]] = 1.0 - epsilon

    return y


def run_lr(x_dense, labels, trn, num_classes, C=0.1):
    clf = LogisticRegression(
        penalty="l2",
        C=C,
        max_iter=1000,
        solver="lbfgs",
        n_jobs=-1,
        random_state=42
    )

    clf.fit(x_dense[trn], labels[trn])

    all_probs = np.zeros((x_dense.shape[0], num_classes), dtype=np.float32)
    pred_probs = clf.predict_proba(x_dense)

    for i, cls in enumerate(clf.classes_):
        all_probs[:, cls] = pred_probs[:, i]

    return all_probs.astype(np.float32)


def run_gbdt(x_dense, labels, trn, num_classes):
    clf = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )

    clf.fit(x_dense[trn], labels[trn])

    all_probs = np.zeros((x_dense.shape[0], num_classes), dtype=np.float32)
    pred_probs = clf.predict_proba(x_dense)

    for i, cls in enumerate(clf.classes_):
        all_probs[:, cls] = pred_probs[:, i]

    return all_probs.astype(np.float32)


def run_ridge(
    x_dense,
    labels,
    trn,
    num_classes,
    alpha=10.0,
    balanced=True,
    temperature=1.5
):
    clf = RidgeClassifier(
        alpha=alpha,
        class_weight="balanced" if balanced else None,
        random_state=42
    )

    clf.fit(x_dense[trn], labels[trn])

    return decision_to_proba_full(
        clf,
        x_dense,
        num_classes,
        temperature=temperature
    )


def run_sparse_ridge(
    x_sparse,
    labels,
    trn,
    num_classes,
    alpha=30.0,
    balanced=True,
    temperature=1.8
):
    """
    新增：Sparse Ridge on original sparse features.
    使用原始稀疏属性特征，补充 SVD 压缩后丢失的细粒度信息。
    """
    clf = RidgeClassifier(
        alpha=alpha,
        class_weight="balanced" if balanced else None,
        random_state=42
    )

    clf.fit(x_sparse[trn], labels[trn])

    return decision_to_proba_full_sparse(
        clf,
        x_sparse,
        num_classes,
        temperature=temperature
    )


def run_extra_trees(
    x_dense,
    labels,
    trn,
    num_classes,
    n_estimators=700,
    max_depth=16,
    min_samples_leaf=3,
    max_features=0.35,
    balanced=True
):
    clf = ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight="balanced" if balanced else None,
        n_jobs=-1,
        random_state=42
    )

    clf.fit(x_dense[trn], labels[trn])

    all_probs = np.zeros((x_dense.shape[0], num_classes), dtype=np.float32)
    pred_probs = clf.predict_proba(x_dense)

    for i, cls in enumerate(clf.classes_):
        all_probs[:, cls] = pred_probs[:, i]

    return all_probs.astype(np.float32)


def run_cs(
    adj_norm,
    base_probs,
    labels,
    seed_idx,
    num_classes,
    alpha,
    beta,
    smooth_eps=0.0
):
    n_nodes = base_probs.shape[0]

    Y = np.zeros((n_nodes, num_classes), dtype=np.float32)

    if smooth_eps > 0:
        y_soft = get_smooth_labels(labels, num_classes, seed_idx, smooth_eps)
        Y[seed_idx] = y_soft
    else:
        Y[seed_idx, labels[seed_idx]] = 1.0

    E = np.zeros_like(base_probs)
    E[seed_idx] = Y[seed_idx] - base_probs[seed_idx]

    E_smooth = E.copy()

    for _ in range(50):
        E_smooth = (1 - alpha) * (adj_norm @ E_smooth) + alpha * E

    P_corr = base_probs + beta * E_smooth

    P_final = P_corr.copy()

    for _ in range(50):
        P_final = (1 - alpha) * (adj_norm @ P_final) + alpha * P_corr

    P_final = np.clip(P_final, 1e-9, 1.0)
    P_final = P_final / P_final.sum(axis=1, keepdims=True)

    return P_final.astype(np.float32)


def run_ppr_lp(
    adj_norm,
    labels,
    seed_idx,
    num_classes,
    alpha=0.20,
    smooth_eps=0.10
):
    n_nodes = adj_norm.shape[0]

    Y = np.zeros((n_nodes, num_classes), dtype=np.float32)

    if smooth_eps > 0:
        y_soft = get_smooth_labels(labels, num_classes, seed_idx, smooth_eps)
        Y[seed_idx] = y_soft
    else:
        Y[seed_idx, labels[seed_idx]] = 1.0

    r = Y.copy()

    for _ in range(50):
        r = alpha * Y + (1 - alpha) * (adj_norm @ r)

    row_sums = r.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0

    return (r / row_sums).astype(np.float32)


def run_lp(
    adj,
    labels,
    seed_idx,
    num_classes,
    alpha=0.20,
    smooth_eps=0.10
):
    n_nodes = adj.shape[0]

    y = np.ones((n_nodes, num_classes), dtype=np.float64) / num_classes

    if smooth_eps > 0:
        y_soft = get_smooth_labels(labels, num_classes, seed_idx, smooth_eps)
        y[seed_idx] = y_soft
    else:
        y[seed_idx, labels[seed_idx]] = 1.0

    y_init = y.copy()

    for _ in range(50):
        y = (1 - alpha) * (adj @ y) + alpha * y_init

    row_sums = y.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0

    return (y / row_sums).astype(np.float32)


def run_lgc(
    adj_norm,
    labels,
    seed_idx,
    num_classes,
    alpha=0.99
):
    n_nodes = adj_norm.shape[0]

    Y = np.zeros((n_nodes, num_classes), dtype=np.float32)
    Y[seed_idx, labels[seed_idx]] = 1.0

    F = Y.copy()

    for _ in range(50):
        F = alpha * (adj_norm @ F) + (1 - alpha) * Y

    row_sums = F.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0

    return (F / row_sums).astype(np.float32)


def run_knn_multi_k(
    x_dense_norm,
    labels,
    seed_idx,
    num_classes,
    top_ks
):
    """
    一次计算多个 k 的 KNN 概率。
    本基线只使用 knn20 / knn50。
    """
    n_nodes = x_dense_norm.shape[0]

    train_x = x_dense_norm[seed_idx]
    train_y = labels[seed_idx]

    sim_matrix = x_dense_norm @ train_x.T
    sim_matrix = np.asarray(sim_matrix, dtype=np.float32)

    max_k = max(top_ks)
    max_k = min(max_k, sim_matrix.shape[1])
    max_k = max(1, max_k)

    kth = max_k - 1

    top_idx = np.argpartition(-sim_matrix, kth, axis=1)[:, :max_k]
    top_sims = np.take_along_axis(sim_matrix, top_idx, axis=1)

    order = np.argsort(-top_sims, axis=1)
    top_idx = np.take_along_axis(top_idx, order, axis=1)
    top_sims = np.take_along_axis(top_sims, order, axis=1)

    out = {}
    rows_base = np.arange(n_nodes)

    for k in top_ks:
        actual_k = min(k, max_k)
        actual_k = max(1, actual_k)

        idx_k = top_idx[:, :actual_k]
        sims_k = top_sims[:, :actual_k]
        labels_k = train_y[idx_k]

        knn_probs = np.zeros((n_nodes, num_classes), dtype=np.float32)

        rows = np.repeat(rows_base, actual_k)
        cols = labels_k.reshape(-1)
        vals = sims_k.reshape(-1)

        mask = vals > 0

        if np.any(mask):
            np.add.at(
                knn_probs,
                (rows[mask], cols[mask]),
                vals[mask]
            )

        row_sums = knn_probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0

        out[f"knn{k}"] = (knn_probs / row_sums).astype(np.float32)

    del sim_matrix
    gc.collect()

    return out


# ─────────────────────────────────────────────
# Stacking 配置
# ─────────────────────────────────────────────

MODEL_NAMES_BASELINE = [
    "lr003",
    "lr03",
    "gbdt",

    "ridge10",
    "sridge30",
    "et2",

    "ppr015",
    "ppr02",
    "lgc",
    "cs01",
    "cs03",

    "knn20",
    "knn50",
]


def train_base_models_baseline(
    x_model,
    x_knn,
    x_sparse,
    adj_norm,
    labels,
    seed_idx,
    num_classes,
    cfg
):
    """
    传统基线模型池。
    只新增 sridge30，其余保持原逻辑。
    """
    probs = {}

    # ─────────────────────────────
    # LR family
    # ─────────────────────────────

    probs["lr003"] = run_lr(
        x_model,
        labels,
        seed_idx,
        num_classes,
        C=0.03
    )

    probs["lr01"] = run_lr(
        x_model,
        labels,
        seed_idx,
        num_classes,
        C=0.10
    )

    probs["lr03"] = run_lr(
        x_model,
        labels,
        seed_idx,
        num_classes,
        C=0.30
    )

    # ─────────────────────────────
    # GBDT
    # ─────────────────────────────

    probs["gbdt"] = run_gbdt(
        x_model,
        labels,
        seed_idx,
        num_classes
    )

    # ─────────────────────────────
    # Ridge family
    # ─────────────────────────────

    probs["ridge10"] = run_ridge(
        x_model,
        labels,
        seed_idx,
        num_classes,
        alpha=10.0,
        balanced=True,
        temperature=1.5
    )

    probs["sridge30"] = run_sparse_ridge(
        x_sparse,
        labels,
        seed_idx,
        num_classes,
        alpha=30.0,
        balanced=True,
        temperature=1.8
    )

    # ─────────────────────────────
    # ExtraTrees: 只保留 et2
    # ─────────────────────────────

    probs["et2"] = run_extra_trees(
        x_model,
        labels,
        seed_idx,
        num_classes,
        n_estimators=700,
        max_depth=16,
        min_samples_leaf=3,
        max_features=0.35,
        balanced=True
    )

    # ─────────────────────────────
    # Graph propagation family
    # ─────────────────────────────

    probs["ppr015"] = run_ppr_lp(
        adj_norm,
        labels,
        seed_idx,
        num_classes,
        alpha=0.15,
        smooth_eps=0.10
    )

    probs["ppr02"] = run_ppr_lp(
        adj_norm,
        labels,
        seed_idx,
        num_classes,
        alpha=0.20,
        smooth_eps=0.10
    )

    probs["lgc"] = run_lgc(
        adj_norm,
        labels,
        seed_idx,
        num_classes,
        alpha=0.99
    )

    probs["cs01"] = run_cs(
        adj_norm,
        probs["lr01"],
        labels,
        seed_idx,
        num_classes,
        alpha=cfg["cs_a"],
        beta=cfg["cs_b"],
        smooth_eps=0.10
    )

    probs["cs03"] = run_cs(
        adj_norm,
        probs["lr03"],
        labels,
        seed_idx,
        num_classes,
        alpha=cfg["cs_a"],
        beta=cfg["cs_b"],
        smooth_eps=0.10
    )

    # ─────────────────────────────
    # KNN: 只保留更平滑的 k
    # ─────────────────────────────

    knn_probs = run_knn_multi_k(
        x_knn,
        labels,
        seed_idx,
        num_classes,
        top_ks=[20, 50]
    )

    probs.update(knn_probs)

    return probs


def make_stack_features(probs_dict, idx, model_names):
    """
    stacking 特征：
    1. 原始概率
    2. log 概率
    3. confidence
    4. entropy
    """
    feats = []
    eps = 1e-6

    for name in model_names:
        if idx is None:
            p = probs_dict[name]
        else:
            p = probs_dict[name][idx]

        p = np.clip(p, eps, 1.0 - eps)

        feats.append(p.astype(np.float32))
        feats.append(np.log(p).astype(np.float32))

        conf = p.max(axis=1, keepdims=True)
        ent = -(p * np.log(p)).sum(axis=1, keepdims=True)

        feats.append(conf.astype(np.float32))
        feats.append(ent.astype(np.float32))

    return np.hstack(feats).astype(np.float32)


def predict_proba_full(clf, X, num_classes):
    pred = clf.predict_proba(X)

    out = np.zeros((X.shape[0], num_classes), dtype=np.float32)

    for i, cls in enumerate(clf.classes_):
        out[:, cls] = pred[:, i]

    return out.astype(np.float32)


def weighted_blend(probs_dict, idx, model_names, weights):
    out = None

    for w, name in zip(weights, model_names):
        if idx is None:
            p = probs_dict[name]
        else:
            p = probs_dict[name][idx]

        if out is None:
            out = w * p
        else:
            out += w * p

    out = np.clip(out, 1e-9, 1.0)
    out = out / out.sum(axis=1, keepdims=True)

    return out.astype(np.float32)


def search_blend_weights(
    oof_probs,
    labels,
    idx,
    model_names,
    n_iter=5000,
    seed=42
):
    """
    在 OOF 上搜索非负权重。
    """
    rng = np.random.default_rng(seed)

    y = labels[idx]
    m = len(model_names)

    best_w = np.ones(m, dtype=np.float32) / m

    best_acc = accuracy_score(
        y,
        weighted_blend(
            oof_probs,
            idx,
            model_names,
            best_w
        ).argmax(axis=1)
    )

    # one-hot baseline
    for i in range(m):
        w = np.zeros(m, dtype=np.float32)
        w[i] = 1.0

        acc = accuracy_score(
            y,
            weighted_blend(
                oof_probs,
                idx,
                model_names,
                w
            ).argmax(axis=1)
        )

        if acc > best_acc:
            best_acc = acc
            best_w = w.copy()

    # Dirichlet random search
    for temp in [0.10, 0.25, 0.50, 1.0, 3.0]:
        for _ in range(n_iter // 5):
            w = rng.dirichlet(np.ones(m) * temp).astype(np.float32)

            acc = accuracy_score(
                y,
                weighted_blend(
                    oof_probs,
                    idx,
                    model_names,
                    w
                ).argmax(axis=1)
            )

            if acc > best_acc:
                best_acc = acc
                best_w = w.copy()

    return best_w, best_acc


def tune_meta_blend_gamma(
    meta_probs,
    weight_probs,
    labels,
    val_idx
):
    """
    搜索 meta-stacker 和 weighted-blend 的融合比例。
    gamma=1 表示全用 meta。
    gamma=0 表示全用 weight-blend。
    """
    y = labels[val_idx]

    best_gamma = 0.5
    best_acc = 0.0

    for gamma in np.linspace(0.0, 1.0, 21):
        p = gamma * meta_probs + (1.0 - gamma) * weight_probs
        acc = accuracy_score(y, p.argmax(axis=1))

        if acc > best_acc:
            best_acc = acc
            best_gamma = float(gamma)

    return best_gamma, best_acc


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  传统基线：多线程 + 代码内固定 PYTHONHASHSEED=0")
    print("  Full Retraining: Disabled")
    print("  PYTHONHASHSEED=0, global_seed=42, n_jobs=-1, thread limits unset")
    print("=" * 80)

    os.makedirs(DATA_ROOT1, exist_ok=True)

    # 1. Load
    adj, features, labels, train_idx, test_idx = load_data()

    labels = labels.astype(np.int64)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)

    num_classes = int(labels[train_idx].max()) + 1
    n_nodes = adj.shape[0]

    # 2. Feature Engineering
    print("\n[Feature Engineering - Structural]")

    print("  计算 Feature SVD...")
    svd = TruncatedSVD(
        n_components=300,
        random_state=42
    )

    x_svd = svd.fit_transform(features).astype(np.float32)

    x_svd_adj = compute_svd_adj(adj, n_dim=32)

    print("  计算图卷积特征...")
    adj_norm = normalize_adj(adj)

    x_svd_sparse = csr_matrix(x_svd)
    x_gc1 = (adj_norm @ x_svd_sparse).toarray().astype(np.float32)
    x_gc2 = (adj_norm @ csr_matrix(x_gc1)).toarray().astype(np.float32)

    del x_svd_sparse
    gc.collect()

    print("  计算拓扑特征...")
    adj_sym_tmp = symmetrize_adj(adj)

    deg_out = np.asarray(adj.sum(axis=1)).reshape(-1, 1)
    deg_in = np.asarray(adj.sum(axis=0)).reshape(-1, 1)
    deg_sym = np.asarray(adj_sym_tmp.sum(axis=1)).reshape(-1, 1)

    pr_vec = compute_pagerank(adj, alpha=0.15)
    hop2_vec = compute_2hop_deg(adj)

    graph_feats = np.hstack([
        deg_out,
        deg_in,
        deg_sym,
        np.log1p(deg_out),
        np.log1p(deg_in),
        np.log1p(deg_sym),
        pr_vec,
        hop2_vec,
        np.log1p(hop2_vec),
    ])

    scaler = StandardScaler()
    graph_feats = scaler.fit_transform(graph_feats).astype(np.float32)

    x_model = np.hstack([
        x_svd,
        x_svd_adj,
        x_gc1,
        x_gc2,
        graph_feats
    ]).astype(np.float32)

    x_knn = normalize(
        x_svd,
        norm="l2",
        axis=1
    ).astype(np.float32)

    # Sparse Ridge 使用原始 sparse features
    print("  构建 Sparse Ridge feature...")
    x_sparse = normalize(
        features,
        norm="l2",
        axis=1,
        copy=True
    ).tocsr()

    # 3. KFold OOF Base Model Training
    print("\n[Training Base Models - KFold OOF]")

    best_cfg = {
        "lr_c": 0.1,
        "ppr_a": 0.2,
        "lp_a": 0.2,
        "lgc_a": 0.99,
        "cs_a": 0.7,
        "cs_b": 0.8,
    }

    model_names = MODEL_NAMES_BASELINE

    min_class_count = np.bincount(labels[train_idx]).min()
    N_FOLDS = int(min(5, max(2, min_class_count)))

    print(f"  Using {N_FOLDS}-Fold OOF stacking")
    print(f"  Number of base models: {len(model_names)}")
    print("  Base models:")
    for name in model_names:
        print(f"    - {name}")

    oof_probs = {
        name: np.zeros((n_nodes, num_classes), dtype=np.float32)
        for name in model_names
    }

    fold_test_probs = []

    skf = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=42
    )

    for fold, (tr_loc, va_loc) in enumerate(
        skf.split(train_idx, labels[train_idx]),
        1
    ):
        fold_trn = train_idx[tr_loc]
        fold_val = train_idx[va_loc]

        print(f"\n  Fold {fold}/{N_FOLDS}: train={len(fold_trn)}, val={len(fold_val)}")

        fold_probs = train_base_models_baseline(
            x_model=x_model,
            x_knn=x_knn,
            x_sparse=x_sparse,
            adj_norm=adj_norm,
            labels=labels,
            seed_idx=fold_trn,
            num_classes=num_classes,
            cfg=best_cfg,
        )

        for name in model_names:
            oof_probs[name][fold_val] = fold_probs[name][fold_val]

        fold_test_probs.append({
            name: fold_probs[name][test_idx].copy()
            for name in model_names
        })

        fold_blend = np.mean(
            [fold_probs[name][fold_val] for name in model_names],
            axis=0
        )

        fold_acc = accuracy_score(
            labels[fold_val],
            fold_blend.argmax(axis=1)
        )

        print(f"  Fold simple avg acc: {fold_acc:.5f}")

        del fold_probs, fold_blend
        gc.collect()

    # 4. Meta-Learning on OOF
    print("\n[Meta-Learning]")

    meta_trn, meta_val = train_test_split(
        train_idx,
        test_size=0.2,
        random_state=2026,
        stratify=labels[train_idx]
    )

    X_meta_trn = make_stack_features(
        oof_probs,
        meta_trn,
        model_names
    )

    X_meta_val = make_stack_features(
        oof_probs,
        meta_val,
        model_names
    )

    tmp_meta = LogisticRegression(
        penalty="l2",
        C=0.08,
        max_iter=META_MAX_ITER,
        solver="lbfgs",
        n_jobs=-1,
        random_state=42,
    )

    tmp_meta.fit(X_meta_trn, labels[meta_trn])
    tmp_meta_n_iter = int(np.max(tmp_meta.n_iter_))
    tmp_meta_converged = tmp_meta_n_iter < META_MAX_ITER
    print(
        f"  Holdout meta iterations: {tmp_meta_n_iter}/{META_MAX_ITER} "
        f"(converged={tmp_meta_converged})"
    )

    meta_val_probs = predict_proba_full(
        tmp_meta,
        X_meta_val,
        num_classes
    )

    meta_val_acc = accuracy_score(
        labels[meta_val],
        meta_val_probs.argmax(axis=1)
    )

    print(f"  Meta holdout acc: {meta_val_acc:.5f}")

    tmp_w, tmp_w_acc = search_blend_weights(
        oof_probs,
        labels,
        meta_trn,
        model_names,
        n_iter=3000,
        seed=42,
    )

    weight_val_probs = weighted_blend(
        oof_probs,
        meta_val,
        model_names,
        tmp_w
    )

    weight_val_acc = accuracy_score(
        labels[meta_val],
        weight_val_probs.argmax(axis=1)
    )

    print(f"  Weight-blend holdout acc: {weight_val_acc:.5f}")

    gamma_raw, combo_val_acc = tune_meta_blend_gamma(
        meta_val_probs,
        weight_val_probs,
        labels,
        meta_val,
    )

    gamma = min(gamma_raw, 0.65)

    print(f"  Raw gamma={gamma_raw:.2f}, capped gamma={gamma:.2f}")
    print(f"  Combo holdout acc before cap search metric={combo_val_acc:.5f}")

    del X_meta_trn, X_meta_val, tmp_meta
    gc.collect()

    # 4.2 最终 meta，用全部 OOF train
    print("\n[Final Meta Training on OOF]")

    X_oof_all = make_stack_features(
        oof_probs,
        train_idx,
        model_names
    )

    meta_clf = LogisticRegression(
        penalty="l2",
        C=0.08,
        max_iter=META_MAX_ITER,
        solver="lbfgs",
        n_jobs=-1,
        random_state=42,
    )

    meta_clf.fit(X_oof_all, labels[train_idx])
    final_meta_n_iter = int(np.max(meta_clf.n_iter_))
    final_meta_converged = final_meta_n_iter < META_MAX_ITER
    print(
        f"  Final meta iterations: {final_meta_n_iter}/{META_MAX_ITER} "
        f"(converged={final_meta_converged})"
    )

    oof_meta_probs = predict_proba_full(
        meta_clf,
        X_oof_all,
        num_classes
    )

    oof_meta_acc = accuracy_score(
        labels[train_idx],
        oof_meta_probs.argmax(axis=1)
    )

    print(f"  Full OOF meta acc: {oof_meta_acc:.5f}")

    blend_w, blend_oof_acc = search_blend_weights(
        oof_probs,
        labels,
        train_idx,
        model_names,
        n_iter=20000,
        seed=2026,
    )

    print(f"  Full OOF weight-blend acc: {blend_oof_acc:.5f}")

    print("  Blend weights:")
    for name, w in zip(model_names, blend_w):
        print(f"    {name:8s}: {w:.4f}")

    del X_oof_all, oof_meta_probs
    gc.collect()

    # 5. Fold-bagged Test Prediction
    print("\n[Test Prediction - Fold Bagging]")

    test_meta_sum = np.zeros(
        (len(test_idx), num_classes),
        dtype=np.float32
    )

    test_weight_sum = np.zeros(
        (len(test_idx), num_classes),
        dtype=np.float32
    )

    for fold, fold_probs_test in enumerate(fold_test_probs, 1):
        print(f"  Predicting fold {fold}/{N_FOLDS}...")

        X_test_fold = make_stack_features(
            fold_probs_test,
            None,
            model_names
        )

        p_meta = predict_proba_full(
            meta_clf,
            X_test_fold,
            num_classes
        )

        p_weight = weighted_blend(
            fold_probs_test,
            None,
            model_names,
            blend_w
        )

        test_meta_sum += p_meta
        test_weight_sum += p_weight

        del X_test_fold, p_meta, p_weight
        gc.collect()

    test_meta_fold = test_meta_sum / N_FOLDS
    test_weight_fold = test_weight_sum / N_FOLDS

    test_combo_fold = gamma * test_meta_fold + (1.0 - gamma) * test_weight_fold
    test_combo_fold = np.clip(test_combo_fold, 1e-9, 1.0)
    test_combo_fold = test_combo_fold / test_combo_fold.sum(axis=1, keepdims=True)

    # 6. Final Prediction without Full Retraining
    print("\n[Final Prediction - No Full Retraining]")

    final_probs = test_combo_fold.copy()
    final_probs = np.clip(final_probs, 1e-9, 1.0)
    final_probs = final_probs / final_probs.sum(axis=1, keepdims=True)

    pred_test = final_probs.argmax(axis=1)

    # 7. Save Submission
    print("\n[Saving Submission]")

    sample = pd.read_csv(
        os.path.join(DATA_ROOT, "sample_submission.csv")
    )

    sample["label"] = pred_test

    sample.to_csv(
        os.path.join(DATA_ROOT1, OUTPUT_NAME),
        index=False
    )

    oof_artifact_path = os.path.join(DATA_ROOT1, OOF_ARTIFACT_NAME)
    np.savez_compressed(
        oof_artifact_path,
        train_idx=train_idx,
        model_names=np.asarray(model_names),
        **{
            name: oof_probs[name][train_idx].astype(np.float32)
            for name in model_names
        },
    )

    summary = {
        "version": "baseline_multithread_hashseed0",
        "base_version": "self_contained_baseline",
        "determinism": {
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "global_seed": GLOBAL_SEED,
            "n_jobs": -1,
            "startup_reexec_guard": True,
            "thread_environment": {
                key: os.environ.get(key)
                for key in THREAD_ENV_KEYS
            },
        },
        "n_folds": N_FOLDS,
        "model_names": model_names,
        "meta_holdout_acc": float(meta_val_acc),
        "weight_holdout_acc": float(weight_val_acc),
        "combo_holdout_acc_raw_gamma": float(combo_val_acc),
        "oof_meta_acc": float(oof_meta_acc),
        "oof_weight_acc": float(blend_oof_acc),
        "gamma_raw": float(gamma_raw),
        "gamma_capped": float(gamma),
        "meta_C": 0.08,
        "meta_max_iter": META_MAX_ITER,
        "tmp_meta_n_iter": tmp_meta_n_iter,
        "tmp_meta_converged": tmp_meta_converged,
        "final_meta_n_iter": final_meta_n_iter,
        "final_meta_converged": final_meta_converged,
        "base_oof_artifact": OOF_ARTIFACT_NAME,
        "blend_weights": {
            name: float(w)
            for name, w in zip(model_names, blend_w)
        },
        "final_blend": {
            "fold_combo": 1.00,
            "full_cs": 0.00,
            "note": "No full retraining. Final prediction uses KFold fold-bagged meta/blend probabilities only.",
        },
        "first_pruning_step": {
            "removed_model": "lp02",
            "model_count_before": 15,
            "model_count_after": 14,
            "stack_feature_count_before": 330,
            "stack_feature_count_after": 308,
            "selection_source": "历史消融与稳定性验证后保留的基线结构",
        },
        "second_pruning_step": {
            "removed_exposed_model": "lr01",
            "internal_dependency_retained_for": "cs01",
            "model_count_before": 14,
            "model_count_after": 13,
            "stack_feature_count_before": 308,
            "stack_feature_count_after": 286,
            "selection_source": "历史条件消融后保留的精简模型集合",
        },
        "added_sparse_feature_model": {
            "added_models": [
                "sridge30"
            ],
            "reason": "Sparse Ridge uses original normalized sparse features to complement dense SVD/graph-convolution features."
        },
        "baseline_pruning": {
            "removed_models": [
                "ridge03",
                "ridge1",
                "svm005",
                "svm02",
                "et1",
                "lp015",
                "knn5",
                "knn10"
            ],
            "kept_new_models": [
                "ridge10",
                "sridge30",
                "et2",
                "knn20",
                "knn50"
            ],
            "reason": "减少 meta-stacker 过拟合，并保留对融合有贡献的模型与稀疏特征补充模型。"
        }
    }

    with open(
        os.path.join(DATA_ROOT1, SUMMARY_NAME),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n" + "=" * 80)
    print(f"  Finished! Saved to {OUTPUT_NAME}")
    print(f"  Summary saved to {SUMMARY_NAME}")
    print(f"  Base OOF saved to {OOF_ARTIFACT_NAME}")
    print(f"  Meta Holdout Acc:    {meta_val_acc:.5f}")
    print(f"  Weight Holdout Acc:  {weight_val_acc:.5f}")
    print(f"  OOF Meta Acc:        {oof_meta_acc:.5f}")
    print(f"  OOF Weight Acc:      {blend_oof_acc:.5f}")
    print(f"  Raw Gamma:           {gamma_raw:.2f}")
    print(f"  Capped Gamma:        {gamma:.2f}")
    print("  Full retraining:     Disabled")
    print("  Added model:         sridge30")
    print("=" * 80)


if __name__ == "__main__":
    main()

