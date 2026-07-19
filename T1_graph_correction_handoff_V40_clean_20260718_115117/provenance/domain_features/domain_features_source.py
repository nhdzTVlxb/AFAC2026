#!/usr/bin/env python
# coding: utf-8

import gc
import importlib.util
import json
import os

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags, eye as speye
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize


VERSION_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.environ.get(
    "AFAC_T1_OUTPUT_DIR",
    os.path.join(VERSION_DIR, "output"),
)
CACHE_PATH = os.path.join(VERSION_DIR, "cache", "v9_base_oof_probs.npz")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "summary_v25_testlike_validation.json")
PROPENSITY_PATH = os.path.join(OUTPUT_DIR, "domain_propensity.csv")

SELECTOR_ATTRIBUTE_MODELS = (
    "lr003", "lr01", "lr03", "gbdt", "ridge10",
    "sridge30", "et2", "knn20", "knn50",
)
ONLINE_SCORES = {
    "V17": 0.7797,
    "V9": 0.7790,
    "V24": 0.7779,
    "V21": 0.7768,
}
CANONICAL_LOCAL_SCORES = {
    "V24": 0.781462971376647,
    "V17": 0.7810086324398001,
    "V9": 0.7791912766924125,
    "V21": 0.7782825988187188,
}


def load_local_train_module():
    path = os.path.join(VERSION_DIR, "train.py")
    spec = importlib.util.spec_from_file_location("v25_base", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_directional_onehop(adj, labels, seed_idx, num_classes, prior_strength=2.0):
    binary = adj.copy().tocsr()
    binary.data = np.ones_like(binary.data, dtype=np.float32)
    binary.setdiag(0)
    binary.eliminate_zeros()
    seed_matrix = np.zeros((adj.shape[0], num_classes), dtype=np.float32)
    seed_matrix[seed_idx, labels[seed_idx]] = 1.0
    out_counts = (binary @ seed_matrix).astype(np.float32)
    in_counts = (binary.T @ seed_matrix).astype(np.float32)
    prior = np.bincount(labels[seed_idx], minlength=num_classes).astype(np.float32)
    prior = (prior + 1.0) / (prior.sum() + num_classes)
    out_total = out_counts.sum(axis=1, keepdims=True)
    in_total = in_counts.sum(axis=1, keepdims=True)
    out_post = (out_counts + prior_strength * prior) / (out_total + prior_strength)
    in_post = (in_counts + prior_strength * prior) / (in_total + prior_strength)

    prior_baseline = float(prior.max())
    seed_out = out_total[seed_idx, 0] > 0
    seed_in = in_total[seed_idx, 0] > 0
    out_acc = accuracy_score(
        labels[seed_idx][seed_out], out_post[seed_idx][seed_out].argmax(axis=1)
    ) if seed_out.any() else prior_baseline
    in_acc = accuracy_score(
        labels[seed_idx][seed_in], in_post[seed_idx][seed_in].argmax(axis=1)
    ) if seed_in.any() else prior_baseline
    out_quality = max(float(out_acc) - prior_baseline, 0.01)
    in_quality = max(float(in_acc) - prior_baseline, 0.01)
    out_top2 = np.partition(out_post, -2, axis=1)[:, -2:]
    in_top2 = np.partition(in_post, -2, axis=1)[:, -2:]
    out_margin = out_top2[:, 1:2] - out_top2[:, 0:1]
    in_margin = in_top2[:, 1:2] - in_top2[:, 0:1]
    out_weight = out_quality * np.sqrt(out_total) * (0.25 + out_margin)
    in_weight = in_quality * np.sqrt(in_total) * (0.25 + in_margin)
    total_weight = out_weight + in_weight
    result = np.tile(prior, (adj.shape[0], 1)).astype(np.float32)
    covered = total_weight[:, 0] > 0
    result[covered] = (
        out_weight[covered] * out_post[covered]
        + in_weight[covered] * in_post[covered]
    ) / total_weight[covered]
    result = np.clip(result, 1e-9, 1.0)
    result /= result.sum(axis=1, keepdims=True)
    return result.astype(np.float32)


def probability_summary_features(probs):
    clipped = np.clip(probs, 1e-6, 1.0)
    top2 = np.partition(clipped, -2, axis=1)[:, -2:]
    confidence = top2[:, 1:2]
    margin = top2[:, 1:2] - top2[:, 0:1]
    entropy = -(clipped * np.log(clipped)).sum(axis=1, keepdims=True)
    return confidence, margin, entropy


def make_ppr_selector_features(probs, structure, num_classes):
    p02 = probs["ppr02"]
    p35 = probs["ppr035"]
    attr = np.mean([probs[name] for name in SELECTOR_ATTRIBUTE_MODELS], axis=0)
    p02_summary = probability_summary_features(p02)
    p35_summary = probability_summary_features(p35)
    attr_summary = probability_summary_features(attr)
    p02_pred = p02.argmax(axis=1)
    p35_pred = p35.argmax(axis=1)
    attr_pred = attr.argmax(axis=1)
    eye = np.eye(num_classes, dtype=np.float32)
    return np.hstack([
        *p02_summary,
        *p35_summary,
        p35_summary[0] - p02_summary[0],
        p35_summary[1] - p02_summary[1],
        p35_summary[2] - p02_summary[2],
        *attr_summary,
        (attr_pred == p02_pred).astype(np.float32).reshape(-1, 1),
        (attr_pred == p35_pred).astype(np.float32).reshape(-1, 1),
        np.abs(attr - p02).sum(axis=1, keepdims=True),
        np.abs(attr - p35).sum(axis=1, keepdims=True),
        structure,
        eye[p02_pred],
        eye[p35_pred],
    ]).astype(np.float32)


def make_selector_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.10,
            penalty="l2",
            max_iter=2000,
            solver="lbfgs",
            random_state=42,
        ),
    )


def build_holdout_safe_pprsel(
    train_raw,
    val_raw,
    labels,
    meta_train_loc,
    structure_train,
    structure_val,
    num_classes,
):
    train_features = make_ppr_selector_features(
        train_raw, structure_train, num_classes
    )
    p02_pred = train_raw["ppr02"].argmax(axis=1)
    p35_pred = train_raw["ppr035"].argmax(axis=1)
    disagreement = p02_pred != p35_pred
    selector_rows = meta_train_loc[disagreement[meta_train_loc]]
    selector_targets = (
        p35_pred[selector_rows] == labels[selector_rows]
    ).astype(np.int64)
    min_count = int(np.bincount(selector_targets, minlength=2).min())
    n_splits = min(5, min_count)
    if n_splits < 2:
        raise RuntimeError("Insufficient PPR selector classes in test-like meta train")

    selected_train = train_raw["ppr02"].copy()
    selector_oof = np.zeros(len(selector_rows), dtype=np.float32)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=31415)
    for fit_loc, pred_loc in skf.split(train_features[selector_rows], selector_targets):
        clf = make_selector_model()
        clf.fit(train_features[selector_rows[fit_loc]], selector_targets[fit_loc])
        selector_oof[pred_loc] = clf.predict_proba(
            train_features[selector_rows[pred_loc]]
        )[:, 1]
    choose_train = selector_oof >= 0.5
    selected_train[selector_rows[choose_train]] = train_raw["ppr035"][
        selector_rows[choose_train]
    ]

    full_selector = make_selector_model()
    full_selector.fit(train_features[selector_rows], selector_targets)
    val_features = make_ppr_selector_features(val_raw, structure_val, num_classes)
    val_p02_pred = val_raw["ppr02"].argmax(axis=1)
    val_p35_pred = val_raw["ppr035"].argmax(axis=1)
    val_disagreement = val_p02_pred != val_p35_pred
    selected_val = val_raw["ppr02"].copy()
    choose_val = np.zeros(len(selected_val), dtype=bool)
    if val_disagreement.any():
        choose_val[val_disagreement] = (
            full_selector.predict_proba(val_features[val_disagreement])[:, 1]
            >= 0.5
        )
        selected_val[choose_val] = val_raw["ppr035"][choose_val]
    stats = {
        "train_disagreements": int(len(selector_rows)),
        "val_disagreements": int(val_disagreement.sum()),
        "train_choose035": int(choose_train.sum()),
        "val_choose035": int(choose_val.sum()),
    }
    return selected_train, selected_val, stats


def build_domain_features(base, adj, attributes, train_idx, test_idx):
    print("[V25] Building label-free domain features...")
    n_nodes = adj.shape[0]
    sym = base.symmetrize_adj(adj)
    sym.setdiag(0)
    sym.eliminate_zeros()
    deg_out = np.asarray(adj.sum(axis=1)).reshape(-1)
    deg_in = np.asarray(adj.sum(axis=0)).reshape(-1)
    deg_sym = np.asarray(sym.sum(axis=1)).reshape(-1)
    pagerank = base.compute_pagerank(adj).reshape(-1)
    hop2 = base.compute_2hop_deg(adj).reshape(-1)
    attr_nnz = np.diff(attributes.indptr).astype(np.float32)
    attr_norm = np.sqrt(np.asarray(attributes.multiply(attributes).sum(axis=1))).reshape(-1)
    attr_svd = TruncatedSVD(n_components=32, random_state=42).fit_transform(
        attributes
    ).astype(np.float32)
    attr_svd = normalize(attr_svd).astype(np.float32)
    adj_svd = base.compute_svd_adj(adj, n_dim=32)

    train_indicator = np.zeros(n_nodes, dtype=np.float32)
    test_indicator = np.zeros(n_nodes, dtype=np.float32)
    train_indicator[train_idx] = 1.0
    test_indicator[test_idx] = 1.0
    neighbor_status = np.column_stack([
        adj @ train_indicator,
        adj.T @ train_indicator,
        sym @ train_indicator,
        adj @ test_indicator,
        adj.T @ test_indicator,
        sym @ test_indicator,
    ]).astype(np.float32)

    n_components, component_id = connected_components(
        sym, directed=False, return_labels=True
    )
    component_size = np.bincount(component_id, minlength=n_components).astype(np.float32)
    component_train = np.bincount(
        component_id, weights=train_indicator, minlength=n_components
    ).astype(np.float32)
    component_test = np.bincount(
        component_id, weights=test_indicator, minlength=n_components
    ).astype(np.float32)
    component_features = np.column_stack([
        np.log1p(component_size[component_id]),
        component_train[component_id] / np.maximum(component_size[component_id], 1.0),
        component_test[component_id] / np.maximum(component_size[component_id], 1.0),
    ]).astype(np.float32)

    scalar = np.column_stack([
        deg_out,
        deg_in,
        deg_sym,
        np.log1p(deg_out),
        np.log1p(deg_in),
        np.log1p(deg_sym),
        pagerank,
        np.log1p(hop2),
        np.log1p(attr_nnz),
        attr_norm,
        neighbor_status,
        component_features,
    ]).astype(np.float32)
    features = np.hstack([scalar, attr_svd, adj_svd]).astype(np.float32)
    features[~np.isfinite(features)] = 0.0
    stats = {
        "feature_count": int(features.shape[1]),
        "component_count": int(n_components),
        "scalar_feature_count": int(scalar.shape[1]),
    }
    return features, stats


def fit_domain_propensity(features, train_idx, test_idx):
    domain_idx = np.concatenate([train_idx, test_idx])
    target = np.concatenate([
        np.zeros(len(train_idx), dtype=np.int64),
        np.ones(len(test_idx), dtype=np.int64),
    ])
    x = features[domain_idx]
    oof = np.zeros(len(domain_idx), dtype=np.float32)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fit_loc, pred_loc in skf.split(x, target):
        clf = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=2.0,
            early_stopping=False,
            random_state=42,
        )
        fit_target = target[fit_loc]
        counts = np.bincount(fit_target, minlength=2).astype(np.float64)
        sample_weight = len(fit_target) / (2.0 * counts[fit_target])
        clf.fit(x[fit_loc], fit_target, sample_weight=sample_weight)
        oof[pred_loc] = clf.predict_proba(x[pred_loc])[:, 1]
    auc = float(roc_auc_score(target, oof))
    propensity = np.zeros(features.shape[0], dtype=np.float32)
    propensity[domain_idx] = oof
    return propensity, auc


def prepare_v9_features(base, adj, attributes):
    print("[V25] Rebuilding exact V9 model features...")
    x_svd = TruncatedSVD(n_components=300, random_state=42).fit_transform(
        attributes
    ).astype(np.float32)
    x_svd_adj = base.compute_svd_adj(adj, n_dim=32)
    adj_norm = base.normalize_adj(adj)
    x_gc1 = (adj_norm @ csr_matrix(x_svd)).toarray().astype(np.float32)
    x_gc2 = (adj_norm @ csr_matrix(x_gc1)).toarray().astype(np.float32)
    sym = base.symmetrize_adj(adj)
    deg_out = np.asarray(adj.sum(axis=1)).reshape(-1, 1)
    deg_in = np.asarray(adj.sum(axis=0)).reshape(-1, 1)
    deg_sym = np.asarray(sym.sum(axis=1)).reshape(-1, 1)
    pagerank = base.compute_pagerank(adj)
    hop2 = base.compute_2hop_deg(adj)
    graph_features = np.hstack([
        deg_out, deg_in, deg_sym,
        np.log1p(deg_out), np.log1p(deg_in), np.log1p(deg_sym),
        pagerank, hop2, np.log1p(hop2),
    ])
    graph_features = StandardScaler().fit_transform(graph_features).astype(np.float32)
    x_model = np.hstack([
        x_svd, x_svd_adj, x_gc1, x_gc2, graph_features
    ]).astype(np.float32)
    x_knn = normalize(x_svd).astype(np.float32)
    x_sparse = normalize(attributes, norm="l2", axis=1, copy=True).tocsr()
    lp_base = adj + adj.T + speye(adj.shape[0], format="csr")
    lp_degree = np.asarray(lp_base.sum(axis=1)).reshape(-1)
    lp_inv = np.power(lp_degree, -1.0)
    lp_inv[~np.isfinite(lp_inv)] = 0.0
    transition = (diags(lp_inv) @ lp_base).tocsr()
    selector_structure = np.hstack([
        StandardScaler().fit_transform(np.hstack([
            np.log1p(deg_out), np.log1p(deg_in), np.log1p(deg_sym)
        ])),
        (deg_out == 0).astype(np.float32),
        (deg_in == 0).astype(np.float32),
        (deg_sym == 0).astype(np.float32),
    ]).astype(np.float32)
    x_struct128 = base.compute_svd_adj(adj, n_dim=128)
    return {
        "x_model": x_model,
        "x_knn": x_knn,
        "x_sparse": x_sparse,
        "adj_norm": adj_norm,
        "transition": transition,
        "selector_structure": selector_structure,
        "x_struct128": x_struct128,
    }


def generate_auxiliary_oof(base, adj, labels, train_idx, prepared, num_classes):
    print("[V25] Reconstructing auxiliary historical OOF...")
    n_train = len(train_idx)
    ppr035 = np.zeros((n_train, num_classes), dtype=np.float32)
    dir1hop = np.zeros_like(ppr035)
    structlr128 = np.zeros_like(ppr035)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fit_loc, val_loc in skf.split(train_idx, labels[train_idx]):
        seed = train_idx[fit_loc]
        val = train_idx[val_loc]
        ppr = base.run_ppr_lp(
            prepared["adj_norm"], labels, seed, num_classes,
            alpha=0.35, smooth_eps=0.10,
        )
        directional = run_directional_onehop(adj, labels, seed, num_classes)
        structural = base.run_lr(
            prepared["x_struct128"], labels, seed, num_classes, C=10.0
        )
        ppr035[val_loc] = ppr[val]
        dir1hop[val_loc] = directional[val]
        structlr128[val_loc] = structural[val]
    return {
        "ppr035": ppr035,
        "dir1hop": dir1hop,
        "structlr128": structlr128,
    }


def evaluate_candidate(
    base,
    name,
    train_probs,
    val_probs,
    labels_local,
    meta_train_loc,
    val_loc,
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
    weights, train_weight_acc = base.search_blend_weights(
        train_probs,
        labels_local,
        meta_train_loc,
        model_names,
        n_iter=3000,
        seed=42,
    )
    weight_probs = base.weighted_blend(val_probs, None, model_names, weights)
    combo = gamma * meta_probs + (1.0 - gamma) * weight_probs
    pred = combo.argmax(axis=1)
    result = {
        "name": name,
        "testlike_acc": float(accuracy_score(labels_local[val_loc], pred)),
        "meta_acc": float(accuracy_score(
            labels_local[val_loc], meta_probs.argmax(axis=1)
        )),
        "weight_acc": float(accuracy_score(
            labels_local[val_loc], weight_probs.argmax(axis=1)
        )),
        "train_weight_acc": float(train_weight_acc),
        "meta_n_iter": int(np.max(meta.n_iter_)),
        "gamma": float(gamma),
        "prediction": pred,
    }
    return result


def pairwise_order_agreement(reference_scores, candidate_scores):
    names = sorted(reference_scores)
    earned = 0.0
    total = 0
    for left_pos, left in enumerate(names):
        for right in names[left_pos + 1:]:
            reference_direction = np.sign(
                reference_scores[left] - reference_scores[right]
            )
            candidate_direction = np.sign(
                candidate_scores[left] - candidate_scores[right]
            )
            total += 1
            if candidate_direction == reference_direction:
                earned += 1.0
            elif candidate_direction == 0:
                earned += 0.5
    return float(earned / total), int(total)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = load_local_train_module()
    adj, attributes, labels, train_idx, test_idx = base.load_data()
    labels = labels.astype(np.int64)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    num_classes = int(labels[train_idx].max()) + 1

    domain_features, domain_feature_stats = build_domain_features(
        base, adj, attributes, train_idx, test_idx
    )
    propensity, domain_auc = fit_domain_propensity(
        domain_features, train_idx, test_idx
    )
    n_val = int(round(0.20 * len(train_idx)))
    train_propensity = propensity[train_idx]
    val_loc = np.argsort(train_propensity, kind="stable")[-n_val:]
    val_loc = np.sort(val_loc)
    meta_train_mask = np.ones(len(train_idx), dtype=bool)
    meta_train_mask[val_loc] = False
    meta_train_loc = np.flatnonzero(meta_train_mask)
    testlike_idx = train_idx[val_loc]
    print(
        f"[V25] Domain AUC={domain_auc:.5f}; "
        f"test-like train={len(meta_train_loc)}, val={len(val_loc)}"
    )

    propensity_frame = pd.DataFrame({
        "node_idx": np.arange(adj.shape[0], dtype=np.int64),
        "split": np.where(
            np.isin(np.arange(adj.shape[0]), test_idx), "test", "train"
        ),
        "test_propensity": propensity,
        "selected_testlike_val": np.isin(np.arange(adj.shape[0]), testlike_idx),
    })
    propensity_frame.to_csv(PROPENSITY_PATH, index=False)

    cache = np.load(CACHE_PATH)
    if not np.array_equal(cache["train_idx"], train_idx):
        raise RuntimeError("Cached V9 OOF train_idx does not match current data")
    base_oof = {
        name: cache[name].astype(np.float32)
        for name in base.MODEL_NAMES_V10
    }

    prepared = prepare_v9_features(base, adj, attributes)
    auxiliary = generate_auxiliary_oof(
        base, adj, labels, train_idx, prepared, num_classes
    )
    print("[V25] Training one clean V9 base fit excluding all test-like nodes...")
    clean_full = base.train_base_models_v10(
        x_model=prepared["x_model"],
        x_knn=prepared["x_knn"],
        x_sparse=prepared["x_sparse"],
        adj_norm=prepared["adj_norm"],
        transition_lp=prepared["transition"],
        labels=labels,
        seed_idx=train_idx[meta_train_loc],
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
    clean_val = {
        name: clean_full[name][testlike_idx].copy()
        for name in base.MODEL_NAMES_V10
    }
    ppr035_clean = base.run_ppr_lp(
        prepared["adj_norm"], labels, train_idx[meta_train_loc], num_classes,
        alpha=0.35, smooth_eps=0.10,
    )[testlike_idx]
    dir_clean = run_directional_onehop(
        adj, labels, train_idx[meta_train_loc], num_classes
    )[testlike_idx]
    struct_clean = base.run_lr(
        prepared["x_struct128"], labels, train_idx[meta_train_loc],
        num_classes, C=10.0,
    )[testlike_idx]

    selector_train_raw = dict(base_oof)
    selector_train_raw["ppr035"] = auxiliary["ppr035"]
    selector_val_raw = dict(clean_val)
    selector_val_raw["ppr035"] = ppr035_clean
    pprsel_train, pprsel_val, selector_stats = build_holdout_safe_pprsel(
        selector_train_raw,
        selector_val_raw,
        labels[train_idx],
        meta_train_loc,
        prepared["selector_structure"][train_idx],
        prepared["selector_structure"][testlike_idx],
        num_classes,
    )

    candidates = {}
    v9_train = dict(base_oof)
    v9_val = dict(clean_val)
    candidates["V9"] = (
        v9_train, v9_val, list(base.MODEL_NAMES_V10), 0.50
    )

    v21_names = [
        "dir1hop" if model == "ppr015" else model
        for model in base.MODEL_NAMES_V10
    ]
    v21_train = dict(base_oof)
    v21_train["dir1hop"] = auxiliary["dir1hop"]
    v21_val = dict(clean_val)
    v21_val["dir1hop"] = dir_clean
    candidates["V21"] = (v21_train, v21_val, v21_names, 0.30)

    v24_names = [
        "structlr128" if model == "ppr015" else model
        for model in base.MODEL_NAMES_V10
    ]
    v24_train = dict(base_oof)
    v24_train["structlr128"] = auxiliary["structlr128"]
    v24_val = dict(clean_val)
    v24_val["structlr128"] = struct_clean
    candidates["V24"] = (v24_train, v24_val, v24_names, 0.50)

    v17_names = []
    for model in base.MODEL_NAMES_V10:
        if model == "ppr015":
            v17_names.append("dir1hop")
        elif model == "ppr02":
            v17_names.append("pprsel")
        else:
            v17_names.append(model)
    v17_train = dict(base_oof)
    v17_train["dir1hop"] = auxiliary["dir1hop"]
    v17_train["pprsel"] = pprsel_train
    v17_val = dict(clean_val)
    v17_val["dir1hop"] = dir_clean
    v17_val["pprsel"] = pprsel_val
    candidates["V17"] = (v17_train, v17_val, v17_names, 0.50)

    results = {}
    for name, (train_probs, val_probs, model_names, gamma) in candidates.items():
        print(f"[V25] Evaluating {name} on test-like validation...")
        results[name] = evaluate_candidate(
            base,
            name,
            train_probs,
            val_probs,
            labels[train_idx],
            meta_train_loc,
            val_loc,
            model_names,
            gamma,
            num_classes,
        )

    v9_prediction = results["V9"]["prediction"]
    for name, result in results.items():
        result["changed_vs_v9"] = int(
            (result["prediction"] != v9_prediction).sum()
        )
        del result["prediction"]

    testlike_ranking = sorted(
        results,
        key=lambda name: results[name]["testlike_acc"],
        reverse=True,
    )
    online_ranking = sorted(ONLINE_SCORES, key=ONLINE_SCORES.get, reverse=True)
    canonical_ranking = sorted(
        CANONICAL_LOCAL_SCORES,
        key=CANONICAL_LOCAL_SCORES.get,
        reverse=True,
    )
    comparison_names = sorted(ONLINE_SCORES)
    testlike_scores = {
        name: results[name]["testlike_acc"]
        for name in comparison_names
    }
    testlike_spearman = float(spearmanr(
        [ONLINE_SCORES[name] for name in comparison_names],
        [testlike_scores[name] for name in comparison_names],
    ).statistic)
    canonical_spearman = float(spearmanr(
        [ONLINE_SCORES[name] for name in comparison_names],
        [CANONICAL_LOCAL_SCORES[name] for name in comparison_names],
    ).statistic)
    testlike_pairwise, pair_count = pairwise_order_agreement(
        ONLINE_SCORES, testlike_scores
    )
    canonical_pairwise, _ = pairwise_order_agreement(
        ONLINE_SCORES, CANONICAL_LOCAL_SCORES
    )
    summary = {
        "version": "v25_testlike_validation_diagnostic",
        "base_version": "v9_meta_convergence",
        "domain_auc": domain_auc,
        "domain_feature_stats": domain_feature_stats,
        "testlike_fraction": float(len(val_loc) / len(train_idx)),
        "testlike_count": int(len(val_loc)),
        "testlike_propensity": {
            "minimum": float(train_propensity[val_loc].min()),
            "mean": float(train_propensity[val_loc].mean()),
            "train_overall_mean": float(train_propensity.mean()),
            "official_test_mean": float(propensity[test_idx].mean()),
        },
        "class_counts": {
            "train_all": np.bincount(
                labels[train_idx], minlength=num_classes
            ).astype(int).tolist(),
            "testlike_val": np.bincount(
                labels[testlike_idx], minlength=num_classes
            ).astype(int).tolist(),
        },
        "selector_stats": selector_stats,
        "results": results,
        "testlike_ranking": testlike_ranking,
        "online_scores": ONLINE_SCORES,
        "online_ranking": online_ranking,
        "canonical_local_scores": CANONICAL_LOCAL_SCORES,
        "canonical_ranking": canonical_ranking,
        "ranking_quality": {
            "testlike_spearman": testlike_spearman,
            "canonical_spearman": canonical_spearman,
            "testlike_pairwise_agreement": testlike_pairwise,
            "canonical_pairwise_agreement": canonical_pairwise,
            "pair_count": pair_count,
        },
        "exact_online_order_match": testlike_ranking == online_ranking,
        "v24_below_v9": (
            results["V24"]["testlike_acc"] < results["V9"]["testlike_acc"]
        ),
        "note": "All test-like validation nodes are excluded together from the clean base-model seed fit. Domain selection uses no class labels.",
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[V25] Saved {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
