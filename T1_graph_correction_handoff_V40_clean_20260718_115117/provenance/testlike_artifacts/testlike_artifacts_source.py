#!/usr/bin/env python
# coding: utf-8

import os
import subprocess
import sys


os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
if os.environ.get("PYTHONHASHSEED") != "0":
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    completed = subprocess.run(
        [sys.executable, os.path.abspath(__file__), *sys.argv[1:]],
        env=env,
        check=False,
    )
    raise SystemExit(completed.returncode)

import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
T1_DIR = SCRIPT_DIR.parents[1]
V25_DIR = T1_DIR / "V25"
V1_DIR = SCRIPT_DIR.parent / "V1"
OUTPUT_PATH = SCRIPT_DIR / "output" / "testlike_validation.json"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def candidate_probabilities(
    base,
    train_probs,
    val_probs,
    labels_local,
    meta_train_loc,
    model_names,
    gamma,
    num_classes,
):
    x_train = base.make_stack_features(train_probs, meta_train_loc, model_names)
    x_val = base.make_stack_features(val_probs, None, model_names)
    meta = LogisticRegression(
        C=0.08,
        penalty="l2",
        max_iter=base.META_MAX_ITER,
        solver="lbfgs",
        n_jobs=-1,
        random_state=42,
    )
    meta.fit(x_train, labels_local[meta_train_loc])
    meta_probs = base.predict_proba_full(meta, x_val, num_classes)
    weights, _ = base.search_blend_weights(
        train_probs,
        labels_local,
        meta_train_loc,
        model_names,
        n_iter=3000,
        seed=42,
    )
    weight_probs = base.weighted_blend(
        val_probs, None, model_names, weights
    )
    return (
        gamma * meta_probs + (1.0 - gamma) * weight_probs
    ).astype(np.float32)


def main():
    started = time.perf_counter()
    v25 = load_module("v25_validate", V25_DIR / "validate.py")
    base = load_module("v25_base", V25_DIR / "train.py")
    gnn = load_module("gnn_v1", V1_DIR / "train.py")
    gnn.seed_everything(42)

    adj, attributes, labels, train_idx, test_idx = base.load_data()
    labels = labels.astype(np.int64)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    num_classes = int(labels[train_idx].max()) + 1

    propensity_frame = pd.read_csv(V25_DIR / "output" / "domain_propensity.csv")
    selected_nodes = propensity_frame.loc[
        propensity_frame["selected_testlike_val"].astype(bool), "node_idx"
    ].to_numpy(dtype=np.int64)
    testlike_mask = np.isin(train_idx, selected_nodes)
    val_loc = np.flatnonzero(testlike_mask)
    meta_train_loc = np.flatnonzero(~testlike_mask)
    testlike_idx = train_idx[val_loc]
    development_idx = train_idx[meta_train_loc]
    if len(val_loc) != 2200:
        raise RuntimeError(f"Unexpected test-like count: {len(val_loc)}")
    print(
        f"  fixed test-like split: development={len(development_idx)}, "
        f"validation={len(testlike_idx)}"
    )

    cache = np.load(V25_DIR / "cache" / "v9_base_oof_probs.npz")
    if not np.array_equal(cache["train_idx"], train_idx):
        raise RuntimeError("V9 cache train_idx mismatch")
    base_names = [
        "lr003", "lr03", "gbdt", "ridge10", "sridge30", "et2",
        "ppr015", "ppr02", "lgc", "cs01", "cs03", "knn20", "knn50",
    ]
    base_train_probs = {
        name: cache[name].astype(np.float32) for name in base_names
    }

    print("  rebuilding clean V33 validation probabilities...")
    prepared = v25.prepare_v9_features(base, adj, attributes)
    clean_full = base.train_base_models_v10(
        x_model=prepared["x_model"],
        x_knn=prepared["x_knn"],
        x_sparse=prepared["x_sparse"],
        adj_norm=prepared["adj_norm"],
        transition_lp=prepared["transition"],
        labels=labels,
        seed_idx=development_idx,
        num_classes=num_classes,
        cfg={
            "lr_c": 0.1,
            "ppr_a": 0.2,
            "lp_a": 0.2,
            "lgc_a": 0.99,
            "cs_a": 0.7,
            "cs_b": 0.8,
        },
    )
    base_val_probs = {
        name: clean_full[name][testlike_idx].copy() for name in base_names
    }

    print("  rebuilding clean GNN validation probabilities...")
    directed, _, matrices = gnn.prepare_graph_views(adj)
    structural, _ = gnn.build_structural_features(
        directed, gnn.binary_without_self(adj + adj.T)
    )
    attribute_blocks = gnn.build_static_attribute_blocks(
        attributes,
        matrices,
        128,
        2,
        V1_DIR / "cache" / "static_svd128_attrhop2.npz",
    )
    outer_blocks = gnn.build_label_blocks(
        matrices, labels, development_idx, num_classes, 3, 0.25
    )
    crossfit_rows = gnn.build_crossfit_label_rows(
        matrices,
        labels,
        development_idx,
        num_classes,
        3,
        0.25,
        5,
        4242,
    )
    outer_blocks[development_idx] = crossfit_rows

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attr_tensor = torch.from_numpy(np.ascontiguousarray(attribute_blocks)).to(device)
    label_tensor = torch.from_numpy(np.ascontiguousarray(outer_blocks)).to(device)
    structural_tensor = torch.from_numpy(np.ascontiguousarray(structural)).to(device)
    target_tensor = torch.from_numpy(labels).long().to(device)
    config = {
        "attr_dim": 128,
        "attr_blocks": 7,
        "label_dim": num_classes + 1,
        "label_blocks": 9,
        "structural_dim": structural.shape[1],
        "hidden_dim": 64,
        "head_dim": 96,
        "dropout": 0.15,
        "lr": 0.005,
        "weight_decay": 2e-4,
    }
    best_epoch, inner_acc, _ = gnn.select_epoch(
        config,
        num_classes,
        device,
        attr_tensor,
        label_tensor,
        structural_tensor,
        target_tensor,
        development_idx,
        4343,
        220,
        25,
        0.15,
        0.02,
    )
    model, _ = gnn.fit_fixed_epochs(
        config,
        num_classes,
        device,
        attr_tensor,
        label_tensor,
        structural_tensor,
        target_tensor,
        development_idx,
        best_epoch,
        4343,
        0.02,
    )
    val_tensor_idx = torch.as_tensor(testlike_idx, dtype=torch.long, device=device)
    gnn_val_probs, _, _ = gnn.predict_probs(
        model,
        attr_tensor,
        label_tensor,
        structural_tensor,
        val_tensor_idx,
    )
    gnn_train_probs = np.load(
        V1_DIR / "output" / "run1" / "oof_probs.npy"
    ).astype(np.float32)

    v33_result = v25.evaluate_candidate(
        base,
        "V33_testlike",
        base_train_probs,
        base_val_probs,
        labels[train_idx],
        meta_train_loc,
        val_loc,
        base_names,
        0.55,
        num_classes,
    )
    v2_train_probs = dict(base_train_probs)
    v2_train_probs["gnn_v1"] = gnn_train_probs
    v2_val_probs = dict(base_val_probs)
    v2_val_probs["gnn_v1"] = gnn_val_probs
    v2_names = base_names + ["gnn_v1"]
    v2_result = v25.evaluate_candidate(
        base,
        "GNN_V2_testlike",
        v2_train_probs,
        v2_val_probs,
        labels[train_idx],
        meta_train_loc,
        val_loc,
        v2_names,
        0.60,
        num_classes,
    )
    v33_combo_probs = candidate_probabilities(
        base,
        base_train_probs,
        base_val_probs,
        labels[train_idx],
        meta_train_loc,
        base_names,
        0.55,
        num_classes,
    )
    np.savez(
        SCRIPT_DIR / "output" / "testlike_router_artifacts.npz",
        v33_combo_probs=v33_combo_probs,
        gnn_val_probs=gnn_val_probs,
        base_val_stack=np.stack(
            [base_val_probs[name] for name in base_names], axis=1
        ),
        testlike_idx=testlike_idx,
        labels=labels[testlike_idx],
        model_names=np.asarray(base_names),
    )
    for result in (v33_result, v2_result):
        result.pop("prediction", None)
    output = {
        "split_source": str(V25_DIR / "output" / "domain_propensity.csv"),
        "development_count": int(len(development_idx)),
        "validation_count": int(len(testlike_idx)),
        "gnn_best_epoch": int(best_epoch),
        "gnn_inner_early_stop_accuracy": float(inner_acc),
        "v33": v33_result,
        "gnn_v2": v2_result,
        "delta": float(v2_result["testlike_acc"] - v33_result["testlike_acc"]),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
