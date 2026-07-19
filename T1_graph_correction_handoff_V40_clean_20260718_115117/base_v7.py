#!/usr/bin/env python
# coding: utf-8

import os
import subprocess
import sys


HASHSEED_ENV = {"PYTHONHASHSEED": "0"}
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# PYTHONHASHSEED and cuBLAS determinism must be configured before importing
# NumPy or PyTorch. Re-exec makes a plain `python train.py` self-contained.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ.update(HASHSEED_ENV)
    completed = subprocess.run(
        [sys.executable, os.path.abspath(__file__), *sys.argv[1:]],
        env=os.environ.copy(),
        check=False,
    )
    raise SystemExit(completed.returncode)

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, normalize
import torch
from torch import nn
import torch.nn.functional as F


VERSION = "t1_gnn_v7_mask_rate_curriculum"
GLOBAL_SEED = 42
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "output")
    parser.add_argument("--cache-dir", type=Path, default=SCRIPT_DIR / "cache")
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=SCRIPT_DIR / "checkpoints"
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--svd-dim", type=int, default=128)
    parser.add_argument("--attr-hops", type=int, default=2)
    parser.add_argument("--label-hops", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--max-epochs", type=int, default=220)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--early-stop-size", type=float, default=0.15)
    parser.add_argument("--label-prior-strength", type=float, default=0.25)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--mask-rate-start", type=float, default=0.20)
    parser.add_argument("--mask-rate-end", type=float, default=0.05)
    parser.add_argument("--curriculum-epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def resolve_data_root(explicit_root):
    if explicit_root is not None:
        root = explicit_root.resolve()
        if not (root / "A1.npz").exists():
            raise FileNotFoundError(root / "A1.npz")
        return root

    env_root = os.environ.get("AFAC_T1_DATA_ROOT")
    if env_root:
        root = Path(env_root).resolve()
        if not (root / "A1.npz").exists():
            raise FileNotFoundError(root / "A1.npz")
        return root

    matches = list((PROJECT_ROOT / "Dataset").rglob("A1.npz"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one A1.npz, found {len(matches)}")
    return matches[0].parent


def load_data(data_root):
    data = np.load(data_root / "A1.npz", allow_pickle=True)
    adj = csr_matrix(
        (data["adj_data"], data["adj_indices"], data["adj_indptr"]),
        shape=tuple(data["adj_shape"]),
        dtype=np.float32,
    )
    features = csr_matrix(
        (data["attr_data"], data["attr_indices"], data["attr_indptr"]),
        shape=tuple(data["attr_shape"]),
        dtype=np.float32,
    )
    labels = np.asarray(data["labels"], dtype=np.int64)
    train_idx = np.asarray(data["train_idx"], dtype=np.int64)
    test_idx = np.asarray(data["test_idx"], dtype=np.int64)
    return adj, features, labels, train_idx, test_idx


def binary_without_self(adj):
    out = adj.copy().tocsr().astype(np.float32)
    out.data = np.ones_like(out.data, dtype=np.float32)
    out.setdiag(0)
    out.eliminate_zeros()
    return out


def row_normalize_sparse(matrix):
    row_sum = np.asarray(matrix.sum(axis=1)).reshape(-1)
    inv = np.zeros_like(row_sum, dtype=np.float32)
    mask = row_sum > 0
    inv[mask] = 1.0 / row_sum[mask]
    return (diags(inv) @ matrix).tocsr().astype(np.float32)


def prepare_graph_views(adj):
    directed = binary_without_self(adj)
    reverse = directed.T.tocsr()
    symmetric = binary_without_self(directed + reverse)
    matrices = {
        "out": row_normalize_sparse(directed),
        "in": row_normalize_sparse(reverse),
        "sym": row_normalize_sparse(symmetric),
    }
    return directed, symmetric, matrices


def build_structural_features(directed, symmetric):
    deg_out = np.asarray(directed.sum(axis=1)).reshape(-1, 1)
    deg_in = np.asarray(directed.sum(axis=0)).reshape(-1, 1)
    deg_sym = np.asarray(symmetric.sum(axis=1)).reshape(-1, 1)
    total = np.maximum(deg_sym, 1.0)
    raw = np.hstack([
        np.log1p(deg_out),
        np.log1p(deg_in),
        np.log1p(deg_sym),
        (deg_out == 0).astype(np.float32),
        (deg_in == 0).astype(np.float32),
        (deg_sym == 0).astype(np.float32),
        deg_out / total,
        deg_in / total,
        np.log1p(np.abs(deg_in - deg_out)),
    ])
    scaled = StandardScaler().fit_transform(raw).astype(np.float32)
    return scaled, deg_sym.reshape(-1)


def build_static_attribute_blocks(features, matrices, svd_dim, attr_hops, cache_path):
    expected_blocks = 1 + len(matrices) * attr_hops
    if cache_path.exists():
        cached = np.load(cache_path)
        blocks = cached["attribute_blocks"]
        if blocks.shape[1:] == (expected_blocks, svd_dim):
            print(f"  Loaded static attribute cache: {cache_path.name}")
            return blocks.astype(np.float32, copy=False)

    print("  Fitting unsupervised sparse attribute SVD...")
    x_sparse = normalize(features, norm="l2", axis=1, copy=True)
    svd = TruncatedSVD(
        n_components=svd_dim,
        n_iter=7,
        random_state=GLOBAL_SEED,
    )
    x0 = svd.fit_transform(x_sparse).astype(np.float32)

    blocks = [x0]
    for name, matrix in matrices.items():
        propagated = x0
        for hop in range(1, attr_hops + 1):
            propagated = (matrix @ propagated).astype(np.float32)
            blocks.append(propagated.copy())
            print(f"    attribute channel={name}, hop={hop}")

    scaled_blocks = []
    for block in blocks:
        scaled_blocks.append(
            StandardScaler().fit_transform(block).astype(np.float32)
        )
    stacked = np.stack(scaled_blocks, axis=1)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, attribute_blocks=stacked)
    print(f"  Saved static attribute cache: {cache_path.name}")
    return stacked


def build_label_blocks(
    matrices,
    labels,
    seed_idx,
    num_classes,
    label_hops,
    prior_strength,
):
    n_nodes = len(labels)
    seed_matrix = np.zeros((n_nodes, num_classes), dtype=np.float32)
    seed_matrix[seed_idx, labels[seed_idx]] = 1.0

    prior = np.bincount(
        labels[seed_idx], minlength=num_classes
    ).astype(np.float32)
    prior = (prior + 1.0) / (prior.sum() + num_classes)

    blocks = []
    for matrix in matrices.values():
        propagated = seed_matrix
        for _ in range(label_hops):
            propagated = (matrix @ propagated).astype(np.float32)
            mass = propagated.sum(axis=1, keepdims=True)
            posterior = (
                propagated + prior_strength * prior.reshape(1, -1)
            ) / (mass + prior_strength)
            blocks.append(np.hstack([posterior, mass]).astype(np.float32))
    return np.stack(blocks, axis=1)


def build_crossfit_label_rows(
    matrices,
    labels,
    supervised_idx,
    num_classes,
    label_hops,
    prior_strength,
    n_splits,
    seed,
):
    min_class_count = int(np.bincount(labels[supervised_idx]).min())
    n_splits = min(n_splits, min_class_count)
    if n_splits < 2:
        raise RuntimeError("Not enough class support for label-feature cross-fit")

    n_blocks = len(matrices) * label_hops
    label_dim = num_classes + 1
    rows = np.zeros(
        (len(supervised_idx), n_blocks, label_dim), dtype=np.float32
    )
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    for inner_fold, (seed_loc, held_loc) in enumerate(
        splitter.split(supervised_idx, labels[supervised_idx]), 1
    ):
        seed_nodes = supervised_idx[seed_loc]
        held_nodes = supervised_idx[held_loc]
        full_blocks = build_label_blocks(
            matrices,
            labels,
            seed_nodes,
            num_classes,
            label_hops,
            prior_strength,
        )
        rows[held_loc] = full_blocks[held_nodes]
        print(
            f"      inner label fold {inner_fold}/{n_splits}: "
            f"seed={len(seed_nodes)}, held={len(held_nodes)}"
        )
    return rows


class DirectedSIGNNet(nn.Module):
    def __init__(
        self,
        attr_dim,
        attr_blocks,
        label_dim,
        label_blocks,
        structural_dim,
        hidden_dim,
        head_dim,
        num_classes,
        dropout,
    ):
        super().__init__()
        self.attr_projection = nn.Linear(attr_dim, hidden_dim)
        self.attr_embedding = nn.Parameter(
            torch.zeros(1, attr_blocks, hidden_dim)
        )
        self.attr_attention = nn.Linear(hidden_dim, 1)

        self.label_projection = nn.Linear(label_dim, hidden_dim)
        self.label_embedding = nn.Parameter(
            torch.zeros(1, label_blocks, hidden_dim)
        )
        self.label_attention = nn.Linear(hidden_dim, 1)

        self.structural_projection = nn.Sequential(
            nn.Linear(structural_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
        )
        combined_dim = hidden_dim * 2 + hidden_dim // 2
        self.head = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Dropout(dropout),
            nn.Linear(combined_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, num_classes),
        )
        nn.init.normal_(self.attr_embedding, std=0.02)
        nn.init.normal_(self.label_embedding, std=0.02)

    @staticmethod
    def aggregate(blocks, projection, embedding, attention):
        hidden = F.gelu(projection(blocks) + embedding)
        weights = torch.softmax(attention(hidden).squeeze(-1), dim=1)
        return torch.sum(hidden * weights.unsqueeze(-1), dim=1), weights

    def forward(self, attribute_blocks, label_blocks, structural):
        attr_hidden, attr_weights = self.aggregate(
            attribute_blocks,
            self.attr_projection,
            self.attr_embedding,
            self.attr_attention,
        )
        label_hidden, label_weights = self.aggregate(
            label_blocks,
            self.label_projection,
            self.label_embedding,
            self.label_attention,
        )
        structural_hidden = self.structural_projection(structural)
        logits = self.head(
            torch.cat([attr_hidden, label_hidden, structural_hidden], dim=1)
        )
        return logits, attr_weights, label_weights


def make_model(config, num_classes, device):
    model = DirectedSIGNNet(
        attr_dim=config["attr_dim"],
        attr_blocks=config["attr_blocks"],
        label_dim=config["label_dim"],
        label_blocks=config["label_blocks"],
        structural_dim=config["structural_dim"],
        hidden_dim=config["hidden_dim"],
        head_dim=config["head_dim"],
        num_classes=num_classes,
        dropout=config["dropout"],
    )
    return model.to(device)


def train_epoch(
    model,
    optimizer,
    attr_tensor,
    label_tensor,
    structural_tensor,
    target_tensor,
    train_idx,
    label_smoothing,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits, _, _ = model(
        attr_tensor[train_idx],
        label_tensor[train_idx],
        structural_tensor[train_idx],
    )
    loss = F.cross_entropy(
        logits,
        target_tensor[train_idx],
        label_smoothing=label_smoothing,
    )
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    return float(loss.detach().cpu())


@torch.no_grad()
def predict_probs(model, attr_tensor, label_tensor, structural_tensor, idx):
    model.eval()
    logits, attr_weights, label_weights = model(
        attr_tensor[idx],
        label_tensor[idx],
        structural_tensor[idx],
    )
    probs = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
    return (
        probs,
        attr_weights.mean(dim=0).cpu().numpy().astype(np.float32),
        label_weights.mean(dim=0).cpu().numpy().astype(np.float32),
    )


def select_epoch(
    config,
    num_classes,
    device,
    attr_tensor,
    label_tensor,
    structural_tensor,
    target_tensor,
    outer_train_idx,
    fold_seed,
    max_epochs,
    patience,
    early_stop_size,
    label_smoothing,
):
    outer_train_tensor = torch.as_tensor(
        outer_train_idx, dtype=torch.long, device=device
    )
    model_train, early_idx = train_test_split(
        outer_train_idx,
        test_size=early_stop_size,
        random_state=fold_seed,
        stratify=target_tensor[outer_train_tensor].cpu().numpy(),
    )
    model_train = torch.as_tensor(model_train, dtype=torch.long, device=device)
    early_idx = torch.as_tensor(early_idx, dtype=torch.long, device=device)

    seed_everything(fold_seed)
    model = make_model(config, num_classes, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    best_epoch = 1
    best_accuracy = -1.0
    stale = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        loss = train_epoch(
            model,
            optimizer,
            attr_tensor,
            label_tensor,
            structural_tensor,
            target_tensor,
            model_train,
            label_smoothing,
        )
        probs, _, _ = predict_probs(
            model, attr_tensor, label_tensor, structural_tensor, early_idx
        )
        accuracy = accuracy_score(
            target_tensor[early_idx].cpu().numpy(), probs.argmax(axis=1)
        )
        history.append({"epoch": epoch, "loss": loss, "accuracy": float(accuracy)})
        if accuracy > best_accuracy + 1e-12:
            best_accuracy = float(accuracy)
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"      epoch={epoch:03d}, loss={loss:.5f}, "
                f"inner_acc={accuracy:.5f}, best={best_accuracy:.5f}@{best_epoch}"
            )
        if stale >= patience:
            break

    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_epoch, best_accuracy, history


def fit_fixed_epochs(
    config,
    num_classes,
    device,
    attr_tensor,
    label_tensor,
    structural_tensor,
    target_tensor,
    train_idx,
    epochs,
    seed,
    label_smoothing,
):
    train_idx = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    seed_everything(seed)
    model = make_model(config, num_classes, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    last_loss = None
    for _ in range(epochs):
        last_loss = train_epoch(
            model,
            optimizer,
            attr_tensor,
            label_tensor,
            structural_tensor,
            target_tensor,
            train_idx,
            label_smoothing,
        )
    return model, float(last_loss)


def split_context_query(nodes, labels, mask_rate, seed):
    context, query = train_test_split(
        np.asarray(nodes, dtype=np.int64),
        test_size=mask_rate,
        random_state=seed,
        stratify=labels[nodes],
    )
    if np.intersect1d(context, query).size:
        raise RuntimeError("Label context/query overlap")
    if len(context) + len(query) != len(nodes):
        raise RuntimeError("Label context/query partition is incomplete")
    return context, query


def mask_rate_for_epoch(epoch, start, end, curriculum_epochs):
    if curriculum_epochs < 1:
        raise ValueError("curriculum_epochs must be positive")
    if not (0.0 < end <= start < 1.0):
        raise ValueError("mask rates must satisfy 0 < end <= start < 1")
    progress = min(max(epoch - 1, 0) / max(curriculum_epochs - 1, 1), 1.0)
    return float(start + progress * (end - start))


def train_masked_epoch(
    model,
    optimizer,
    attr_tensor,
    structural_tensor,
    target_tensor,
    query_nodes,
    query_label_rows,
    device,
    label_smoothing,
):
    query_tensor = torch.as_tensor(query_nodes, dtype=torch.long, device=device)
    label_tensor = torch.from_numpy(
        np.ascontiguousarray(query_label_rows)
    ).to(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits, _, _ = model(
        attr_tensor[query_tensor],
        label_tensor,
        structural_tensor[query_tensor],
    )
    loss = F.cross_entropy(
        logits,
        target_tensor[query_tensor],
        label_smoothing=label_smoothing,
    )
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    return float(loss.detach().cpu())


@torch.no_grad()
def predict_with_label_rows(
    model,
    attr_tensor,
    structural_tensor,
    nodes,
    label_rows,
    device,
):
    node_tensor = torch.as_tensor(nodes, dtype=torch.long, device=device)
    label_tensor = torch.from_numpy(np.ascontiguousarray(label_rows)).to(device)
    model.eval()
    logits, attr_weights, label_weights = model(
        attr_tensor[node_tensor],
        label_tensor,
        structural_tensor[node_tensor],
    )
    return (
        torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32),
        attr_weights.mean(dim=0).cpu().numpy().astype(np.float32),
        label_weights.mean(dim=0).cpu().numpy().astype(np.float32),
    )


def select_epoch_random_mask(
    config,
    num_classes,
    device,
    attr_tensor,
    structural_tensor,
    target_tensor,
    matrices,
    labels,
    outer_train_idx,
    fold_seed,
    max_epochs,
    patience,
    early_stop_size,
    mask_rate_start,
    mask_rate_end,
    curriculum_epochs,
    label_hops,
    prior_strength,
    label_smoothing,
):
    model_train, early_idx = train_test_split(
        np.asarray(outer_train_idx, dtype=np.int64),
        test_size=early_stop_size,
        random_state=fold_seed,
        stratify=labels[outer_train_idx],
    )
    early_blocks = build_label_blocks(
        matrices,
        labels,
        model_train,
        num_classes,
        label_hops,
        prior_strength,
    )[early_idx]
    seed_everything(fold_seed)
    model = make_model(config, num_classes, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    best_epoch = 1
    best_accuracy = -1.0
    stale = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        mask_rate = mask_rate_for_epoch(
            epoch, mask_rate_start, mask_rate_end, curriculum_epochs
        )
        context, query = split_context_query(
            model_train, labels, mask_rate, fold_seed * 1000 + epoch
        )
        label_rows = build_label_blocks(
            matrices,
            labels,
            context,
            num_classes,
            label_hops,
            prior_strength,
        )[query]
        loss = train_masked_epoch(
            model,
            optimizer,
            attr_tensor,
            structural_tensor,
            target_tensor,
            query,
            label_rows,
            device,
            label_smoothing,
        )
        probabilities, _, _ = predict_with_label_rows(
            model,
            attr_tensor,
            structural_tensor,
            early_idx,
            early_blocks,
            device,
        )
        accuracy = accuracy_score(labels[early_idx], probabilities.argmax(axis=1))
        history.append(
            {
                "epoch": epoch,
                "loss": float(loss),
                "accuracy": float(accuracy),
                "context_count": int(len(context)),
                "query_count": int(len(query)),
                "context_query_overlap": 0,
                "mask_rate": float(mask_rate),
            }
        )
        if accuracy > best_accuracy + 1e-12:
            best_accuracy = float(accuracy)
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"      epoch={epoch:03d}, loss={loss:.5f}, "
                f"inner_acc={accuracy:.5f}, best={best_accuracy:.5f}@{best_epoch}"
            )
        if stale >= patience:
            break
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_epoch, best_accuracy, history


def fit_random_mask_epochs(
    config,
    num_classes,
    device,
    attr_tensor,
    structural_tensor,
    target_tensor,
    matrices,
    labels,
    supervised_idx,
    epochs,
    seed,
    mask_rate_start,
    mask_rate_end,
    curriculum_epochs,
    label_hops,
    prior_strength,
    label_smoothing,
):
    seed_everything(seed)
    model = make_model(config, num_classes, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    last_loss = None
    for epoch in range(1, epochs + 1):
        mask_rate = mask_rate_for_epoch(
            epoch, mask_rate_start, mask_rate_end, curriculum_epochs
        )
        context, query = split_context_query(
            supervised_idx, labels, mask_rate, seed * 1000 + epoch
        )
        label_rows = build_label_blocks(
            matrices,
            labels,
            context,
            num_classes,
            label_hops,
            prior_strength,
        )[query]
        last_loss = train_masked_epoch(
            model,
            optimizer,
            attr_tensor,
            structural_tensor,
            target_tensor,
            query,
            label_rows,
            device,
            label_smoothing,
        )
    return model, float(last_loss)


def assert_label_context_isolation(
    matrices,
    labels,
    train_idx,
    num_classes,
    label_hops,
    prior_strength,
    mask_rate,
):
    context, query = split_context_query(train_idx, labels, mask_rate, 6060)
    baseline = build_label_blocks(
        matrices,
        labels,
        context,
        num_classes,
        label_hops,
        prior_strength,
    )[query]
    mutated = labels.copy()
    mutated[query] = (mutated[query] + 1) % num_classes
    repeated = build_label_blocks(
        matrices,
        mutated,
        context,
        num_classes,
        label_hops,
        prior_strength,
    )[query]
    if not np.array_equal(baseline, repeated):
        raise RuntimeError("Query label mutation changed its label context")
    return {
        "context_count": int(len(context)),
        "mutated_query_count": int(len(query)),
        "context_query_overlap": 0,
        "byte_identical": True,
    }


def cpu_state_dict(model):
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def class_recall(labels, predictions, num_classes):
    matrix = confusion_matrix(labels, predictions, labels=np.arange(num_classes))
    denom = matrix.sum(axis=1)
    recall = np.divide(
        np.diag(matrix),
        denom,
        out=np.zeros(num_classes, dtype=np.float64),
        where=denom > 0,
    )
    return {str(i): float(value) for i, value in enumerate(recall)}


def bucket_metrics(labels, predictions, degree, coverage):
    result = {}
    masks = {
        "covered": coverage,
        "uncovered": ~coverage,
        "degree_0": degree == 0,
        "degree_1": degree == 1,
        "degree_2_3": (degree >= 2) & (degree <= 3),
        "degree_4_plus": degree >= 4,
    }
    for name, mask in masks.items():
        result[name] = {
            "count": int(mask.sum()),
            "accuracy": (
                float(accuracy_score(labels[mask], predictions[mask]))
                if mask.any()
                else None
            ),
        }
    return result


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def save_submission(sample_path, output_path, predictions):
    sample = pd.read_csv(sample_path)
    if len(sample) != len(predictions):
        raise ValueError("Submission row count mismatch")
    sample["label"] = predictions.astype(np.int64)
    sample.to_csv(output_path, index=False)


def main():
    args = parse_args()
    seed_everything(args.seed)
    start_time = time.perf_counter()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    data_root = resolve_data_root(args.data_root)
    device = torch.device(args.device)

    print("=" * 88)
    print("  T1 GNN/V7: Directed SIGN with Mask-rate Curriculum")
    print(f"  device={device}, torch={torch.__version__}")
    print("  PYTHONHASHSEED=0 is enforced in code before imports")
    print("=" * 88)

    adj, features, labels, train_idx, test_idx = load_data(data_root)
    num_classes = int(labels[train_idx].max()) + 1
    n_nodes = adj.shape[0]
    print(
        f"  nodes={n_nodes}, edges={adj.nnz}, train={len(train_idx)}, "
        f"test={len(test_idx)}, classes={num_classes}"
    )

    directed, symmetric, matrices = prepare_graph_views(adj)
    structural, sym_degree = build_structural_features(directed, symmetric)
    isolation_audit = assert_label_context_isolation(
        matrices,
        labels,
        train_idx,
        num_classes,
        args.label_hops,
        args.label_prior_strength,
        args.mask_rate_start,
    )
    print(
        "  query-label mutation leaves label context byte-identical: "
        f"{isolation_audit['byte_identical']}"
    )
    cache_path = args.cache_dir / (
        f"static_svd{args.svd_dim}_attrhop{args.attr_hops}.npz"
    )
    attribute_blocks = build_static_attribute_blocks(
        features,
        matrices,
        args.svd_dim,
        args.attr_hops,
        cache_path,
    )
    full_label_blocks = build_label_blocks(
        matrices,
        labels,
        train_idx,
        num_classes,
        args.label_hops,
        args.label_prior_strength,
    )

    config = {
        "attr_dim": int(attribute_blocks.shape[2]),
        "attr_blocks": int(attribute_blocks.shape[1]),
        "label_dim": int(full_label_blocks.shape[2]),
        "label_blocks": int(full_label_blocks.shape[1]),
        "structural_dim": int(structural.shape[1]),
        "hidden_dim": args.hidden_dim,
        "head_dim": args.head_dim,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }
    print(f"  model_config={config}")

    attr_tensor = torch.from_numpy(
        np.ascontiguousarray(attribute_blocks)
    ).to(device)
    structural_tensor = torch.from_numpy(
        np.ascontiguousarray(structural)
    ).to(device)
    full_label_tensor = torch.from_numpy(
        np.ascontiguousarray(full_label_blocks)
    ).to(device)
    target_tensor = torch.from_numpy(labels).long().to(device)

    outer_splitter = StratifiedKFold(
        n_splits=args.outer_folds,
        shuffle=True,
        random_state=args.seed,
    )
    oof_probs = np.zeros((n_nodes, num_classes), dtype=np.float32)
    oof_coverage = np.zeros(n_nodes, dtype=bool)
    fold_test_outer_context = []
    fold_test_full_context = []
    fold_results = []
    completed_folds = 0

    for fold, (outer_train_loc, outer_val_loc) in enumerate(
        outer_splitter.split(train_idx, labels[train_idx]), 1
    ):
        if args.smoke and fold > 1:
            break
        completed_folds += 1
        fold_start = time.perf_counter()
        outer_train = train_idx[outer_train_loc]
        outer_val = train_idx[outer_val_loc]
        print("\n" + "-" * 88)
        print(
            f"  Outer fold {fold}/{args.outer_folds}: "
            f"train={len(outer_train)}, val={len(outer_val)}"
        )
        print("    Building outer-full label context...")
        outer_full_blocks = build_label_blocks(
            matrices,
            labels,
            outer_train,
            num_classes,
            args.label_hops,
            args.label_prior_strength,
        )
        fold_seed = args.seed + fold
        max_epochs = 5 if args.smoke else args.max_epochs
        patience = 3 if args.smoke else args.patience
        best_epoch, inner_acc, history = select_epoch_random_mask(
            config,
            num_classes,
            device,
            attr_tensor,
            structural_tensor,
            target_tensor,
            matrices,
            labels,
            outer_train,
            fold_seed,
            max_epochs,
            patience,
            args.early_stop_size,
            args.mask_rate_start,
            args.mask_rate_end,
            args.curriculum_epochs,
            args.label_hops,
            args.label_prior_strength,
            args.label_smoothing,
        )
        print(f"    Retraining outer model for {best_epoch} fixed epochs...")
        model, final_loss = fit_random_mask_epochs(
            config,
            num_classes,
            device,
            attr_tensor,
            structural_tensor,
            target_tensor,
            matrices,
            labels,
            outer_train,
            best_epoch,
            fold_seed,
            args.mask_rate_start,
            args.mask_rate_end,
            args.curriculum_epochs,
            args.label_hops,
            args.label_prior_strength,
            args.label_smoothing,
        )

        val_probs, attr_attention, label_attention = predict_with_label_rows(
            model,
            attr_tensor,
            structural_tensor,
            outer_val,
            outer_full_blocks[outer_val],
            device,
        )
        test_outer_probs, _, _ = predict_with_label_rows(
            model,
            attr_tensor,
            structural_tensor,
            test_idx,
            outer_full_blocks[test_idx],
            device,
        )
        test_full_probs, _, _ = predict_with_label_rows(
            model,
            attr_tensor,
            structural_tensor,
            test_idx,
            full_label_blocks[test_idx],
            device,
        )
        oof_probs[outer_val] = val_probs
        fold_test_outer_context.append(test_outer_probs)
        fold_test_full_context.append(test_full_probs)

        coverage = np.asarray(
            symmetric[outer_val][:, outer_train].sum(axis=1)
        ).reshape(-1) > 0
        oof_coverage[outer_val] = coverage
        val_predictions = val_probs.argmax(axis=1)
        fold_accuracy = accuracy_score(labels[outer_val], val_predictions)
        fold_result = {
            "fold": fold,
            "accuracy": float(fold_accuracy),
            "best_epoch": int(best_epoch),
            "inner_early_stop_accuracy": float(inner_acc),
            "final_train_loss": float(final_loss),
            "train_count": int(len(outer_train)),
            "validation_count": int(len(outer_val)),
            "validation_seed_neighbor_coverage": float(coverage.mean()),
            "attr_attention": attr_attention.tolist(),
            "label_attention": label_attention.tolist(),
            "selection_history": history,
            "elapsed_seconds": float(time.perf_counter() - fold_start),
        }
        fold_results.append(fold_result)
        torch.save(
            {
                "version": VERSION,
                "fold": fold,
                "config": config,
                "best_epoch": best_epoch,
                "state_dict": cpu_state_dict(model),
            },
            args.checkpoint_dir / f"fold_{fold}.pt",
        )
        print(
            f"    Outer fold accuracy={fold_accuracy:.6f}, "
            f"coverage={coverage.mean():.4f}, elapsed={fold_result['elapsed_seconds']:.1f}s"
        )
        del model, outer_full_blocks
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.smoke:
        smoke_summary = {
            "version": VERSION,
            "status": "smoke_passed",
            "completed_folds": completed_folds,
            "fold_results": fold_results,
            "elapsed_seconds": float(time.perf_counter() - start_time),
        }
        with open(args.output_dir / "smoke_summary.json", "w", encoding="utf-8") as handle:
            json.dump(smoke_summary, handle, indent=2)
        print("\nSmoke test completed successfully; no submission was generated.")
        return

    oof_predictions = oof_probs[train_idx].argmax(axis=1)
    oof_accuracy = accuracy_score(labels[train_idx], oof_predictions)
    _, canonical_idx = train_test_split(
        train_idx,
        test_size=0.2,
        random_state=2026,
        stratify=labels[train_idx],
    )
    canonical_accuracy = accuracy_score(
        labels[canonical_idx], oof_probs[canonical_idx].argmax(axis=1)
    )
    fold_accuracies = np.asarray(
        [result["accuracy"] for result in fold_results], dtype=np.float64
    )
    print("\n" + "=" * 88)
    print(f"  Full OOF accuracy:       {oof_accuracy:.6f}")
    print(f"  Canonical-2026 accuracy: {canonical_accuracy:.6f}")
    print(
        f"  Fold mean/std:           {fold_accuracies.mean():.6f} / "
        f"{fold_accuracies.std(ddof=0):.6f}"
    )

    fold_full_test_probs = np.mean(fold_test_full_context, axis=0).astype(np.float32)
    fold_outer_test_probs = np.mean(fold_test_outer_context, axis=0).astype(np.float32)
    artifact_path = args.output_dir / "artifacts_run1.npz"
    np.savez_compressed(
        artifact_path,
        train_idx=train_idx,
        test_idx=test_idx,
        oof_probs=oof_probs[train_idx],
        test_probs_fold_full_context=fold_full_test_probs,
        test_probs_fold_outer_context=fold_outer_test_probs,
        oof_coverage=oof_coverage[train_idx],
    )
    summary = {
        "version": VERSION,
        "status": "trained",
        "data": {
            "nodes": int(n_nodes),
            "directed_edges": int(adj.nnz),
            "attribute_shape": list(features.shape),
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "num_classes": int(num_classes),
            "class_counts": np.bincount(
                labels[train_idx], minlength=num_classes
            ).tolist(),
        },
        "protocol": {
            "outer_folds": args.outer_folds,
            "random_label_mask_schedule": {
                "start": args.mask_rate_start,
                "end": args.mask_rate_end,
                "curriculum_epochs": args.curriculum_epochs,
            },
            "supervised_query_is_label_seed": False,
            "early_stop_nodes_are_label_seeds": False,
            "outer_validation_is_label_seed": False,
            "outer_validation_used_for_epoch_selection": False,
            "query_label_mutation_check": isolation_audit,
            "submission_generated": False,
        },
        "config": vars(args) | config,
        "metrics": {
            "oof_accuracy": float(oof_accuracy),
            "canonical_2026_accuracy": float(canonical_accuracy),
            "fold_mean": float(fold_accuracies.mean()),
            "fold_std": float(fold_accuracies.std(ddof=0)),
            "class_recall": class_recall(
                labels[train_idx], oof_predictions, num_classes
            ),
            "buckets": bucket_metrics(
                labels[train_idx],
                oof_predictions,
                sym_degree[train_idx],
                oof_coverage[train_idx],
            ),
        },
        "v6_reference": {
            "oof_accuracy": 0.7432051631669848,
            "canonical_2026_accuracy": 0.7555656519763744,
            "covered_accuracy": 0.8568304605093903,
            "uncovered_accuracy": 0.4694762937713046,
            "degree_0_accuracy": 0.35246272028920017,
        },
        "fold_results": fold_results,
        "artifacts": {
            "probabilities": str(artifact_path),
            "probabilities_sha256": sha256(artifact_path),
        },
        "runtime": {
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "elapsed_seconds": float(time.perf_counter() - start_time),
        },
    }
    summary["config"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in summary["config"].items()
    }
    summary_path = args.output_dir / "summary_run1.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"  Saved producer artifacts: {artifact_path}")
    print(f"  Total elapsed: {summary['runtime']['elapsed_seconds']:.1f}s")
    print("  No submission CSV was generated.")
    return

    print("\n[Final full-train model]")
    print("  Building all-train cross-fitted label rows...")
    final_crossfit_rows = build_crossfit_label_rows(
        matrices,
        labels,
        train_idx,
        num_classes,
        args.label_hops,
        args.label_prior_strength,
        args.inner_folds,
        args.seed + 9000,
    )
    final_label_blocks = full_label_blocks.copy()
    final_label_blocks[train_idx] = final_crossfit_rows
    final_label_tensor = torch.from_numpy(
        np.ascontiguousarray(final_label_blocks)
    ).to(device)
    final_epochs = int(np.median([result["best_epoch"] for result in fold_results]))
    final_epochs = max(final_epochs, 1)
    print(f"  Training final model for median best_epoch={final_epochs}...")
    final_model, final_loss = fit_fixed_epochs(
        config,
        num_classes,
        device,
        attr_tensor,
        final_label_tensor,
        structural_tensor,
        target_tensor,
        train_idx,
        final_epochs,
        args.seed + 10000,
        args.label_smoothing,
    )
    test_tensor_idx = torch.as_tensor(test_idx, dtype=torch.long, device=device)
    final_test_probs, final_attr_attention, final_label_attention = predict_probs(
        final_model,
        attr_tensor,
        full_label_tensor,
        structural_tensor,
        test_tensor_idx,
    )
    fold_full_test_probs = np.mean(fold_test_full_context, axis=0).astype(np.float32)
    fold_outer_test_probs = np.mean(fold_test_outer_context, axis=0).astype(np.float32)

    np.save(args.output_dir / "oof_probs.npy", oof_probs[train_idx])
    np.save(args.output_dir / "test_probs_final.npy", final_test_probs)
    np.save(args.output_dir / "test_probs_fold_full_context.npy", fold_full_test_probs)
    np.save(args.output_dir / "test_probs_fold_outer_context.npy", fold_outer_test_probs)

    submission_path = args.output_dir / "A1.csv"
    fold_submission_path = args.output_dir / "A1_fold_full_context.csv"
    outer_submission_path = args.output_dir / "A1_fold_outer_context.csv"
    sample_path = data_root / "sample_submission.csv"
    save_submission(sample_path, submission_path, final_test_probs.argmax(axis=1))
    save_submission(
        sample_path, fold_submission_path, fold_full_test_probs.argmax(axis=1)
    )
    save_submission(
        sample_path, outer_submission_path, fold_outer_test_probs.argmax(axis=1)
    )
    torch.save(
        {
            "version": VERSION,
            "config": config,
            "epochs": final_epochs,
            "state_dict": cpu_state_dict(final_model),
        },
        args.checkpoint_dir / "final.pt",
    )

    full_train_predictions = oof_predictions
    summary = {
        "version": VERSION,
        "status": "trained",
        "data": {
            "nodes": int(n_nodes),
            "directed_edges": int(adj.nnz),
            "attribute_shape": list(features.shape),
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "num_classes": int(num_classes),
            "class_counts": np.bincount(labels[train_idx], minlength=num_classes).tolist(),
        },
        "protocol": {
            "outer_folds": args.outer_folds,
            "inner_label_folds": args.inner_folds,
            "outer_validation_used_for_epoch_selection": False,
            "training_label_features": "inner cross-fitted; supervised row labels excluded from propagation seeds",
            "outer_validation_label_features": "all outer-train labels only",
            "final_training_label_features": "five-fold cross-fitted over all train rows",
            "final_test_label_features": "all train labels",
        },
        "config": vars(args) | config,
        "metrics": {
            "oof_accuracy": float(oof_accuracy),
            "canonical_2026_accuracy": float(canonical_accuracy),
            "fold_mean": float(fold_accuracies.mean()),
            "fold_std": float(fold_accuracies.std(ddof=0)),
            "class_recall": class_recall(
                labels[train_idx], full_train_predictions, num_classes
            ),
            "buckets": bucket_metrics(
                labels[train_idx],
                full_train_predictions,
                sym_degree[train_idx],
                oof_coverage[train_idx],
            ),
        },
        "fold_results": fold_results,
        "final_model": {
            "epochs": final_epochs,
            "train_loss": float(final_loss),
            "attr_attention": final_attr_attention.tolist(),
            "label_attention": final_label_attention.tolist(),
        },
        "prediction_comparison": {
            "final_vs_fold_full_context_differences": int(
                (
                    final_test_probs.argmax(axis=1)
                    != fold_full_test_probs.argmax(axis=1)
                ).sum()
            ),
            "final_vs_fold_outer_context_differences": int(
                (
                    final_test_probs.argmax(axis=1)
                    != fold_outer_test_probs.argmax(axis=1)
                ).sum()
            ),
            "fold_full_vs_outer_context_differences": int(
                (
                    fold_full_test_probs.argmax(axis=1)
                    != fold_outer_test_probs.argmax(axis=1)
                ).sum()
            ),
        },
        "artifacts": {
            "submission": submission_path.name,
            "fold_full_context_submission": fold_submission_path.name,
            "fold_outer_context_submission": outer_submission_path.name,
            "submission_sha256": sha256(submission_path),
        },
        "runtime": {
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "elapsed_seconds": float(time.perf_counter() - start_time),
        },
    }
    summary["config"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in summary["config"].items()
    }
    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\n" + "=" * 88)
    print(f"  Saved submission: {submission_path}")
    print(f"  SHA256: {summary['artifacts']['submission_sha256']}")
    print(f"  Total elapsed: {summary['runtime']['elapsed_seconds']:.1f}s")
    print("=" * 88)


if __name__ == "__main__":
    main()
