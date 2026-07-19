#!/usr/bin/env python
# coding: utf-8

import os
import subprocess
import sys


os.environ["PYTHONUNBUFFERED"] = "1"
if os.environ.get("PYTHONHASHSEED") != "0":
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    completed = subprocess.run(
        [sys.executable, os.path.abspath(__file__), *sys.argv[1:]],
        env=env,
        check=False,
    )
    raise SystemExit(completed.returncode)

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import base_v7 as graph_base


VERSION = "t1_graph_correction_handoff"
SCRIPT_DIR = Path(__file__).resolve().parent
GNN_DIR = SCRIPT_DIR.parent
T1_DIR = GNN_DIR.parent
PROJECT_ROOT = T1_DIR.parents[1]


def discover_data_root():
    local_data_root = SCRIPT_DIR / "data"
    if (local_data_root / "A1.npz").exists():
        return local_data_root
    matches = sorted((PROJECT_ROOT / "Dataset").rglob("A1.npz"))
    if len(matches) != 1:
        raise RuntimeError(
            "未在本目录 data/ 下找到 A1.npz，且无法在工作区 Dataset/ 下唯一定位 A1.npz"
        )
    return matches[0].parent


CONFIG = {
    "determinism": {
        "pythonhashseed": "0",
        "global_seed": 42,
        "selector_split_seed": 31415,
        "canonical_split_seed": 2026,
        "level2_oof_seed": 777,
        "v33_n_jobs": -1,
        "selector_n_jobs": -1,
    },
    "versions": {
        "classic_baseline": "本目录 baseline_train.py",
        "graph_correction_source": "本目录 deps/graph_correction_artifacts.npz",
        "validation_source": "本目录 deps/baseline_testlike_artifacts.npz 与 deps/graph_testlike_probabilities.npz",
        "domain_propensity_source": "本目录 deps/domain_propensity.csv",
    },
    "selector": {
        "model": "extra_trees",
        "threshold": 0.65,
        "threshold_source": "仅根据 OOF 选择的保守阈值",
        "minimum_selected_count": 10,
        "require_strict_positive_fold_net": True,
    },
    "paths": {
        "baseline_train_script": SCRIPT_DIR / "baseline_train.py",
        "data_root": discover_data_root(),
        "data_npz": discover_data_root() / "A1.npz",
        "sample_submission": discover_data_root() / "sample_submission.csv",
        "domain_propensity": SCRIPT_DIR / "deps" / "domain_propensity.csv",
        "graph_correction_artifacts": SCRIPT_DIR / "deps" / "graph_correction_artifacts.npz",
        "graph_testlike_probabilities": SCRIPT_DIR / "deps" / "graph_testlike_probabilities.npz",
        "baseline_testlike_artifacts": SCRIPT_DIR / "deps" / "baseline_testlike_artifacts.npz",
        "base_v7_script": SCRIPT_DIR / "base_v7.py",
    },
    "expected_hashes": {
        "baseline_train_script": "B18B22F14DA5F89B095501DC9C06DFC91A0FB7B0CB4805A57C32E1CFEDB7A959",
        "base_v7_script": "34CC96460972F8664510DB78B4DD256AE5CED580C8D6EE7B2AC48F2D24EEDE22",
        "data_npz": "BCF4CA02E2FAC12590058CEFC941BF46464C4AAB907108C460D7CC55C987DA35",
        "sample_submission": "398B09C01AEBC8997069CF74E290C2467BBB1ABBE60C1A02FE6C9308209E11BA",
        "domain_propensity": "CBC87AD3DB4364DCFC1399A44C3932059C5957CBC69D45D4DE3A468013BDB9E6",
        "graph_correction_artifacts": "03FF6CB4380422CDE949F77F35594B79CCE832293A13BACBFDE38210D77E8B2A",
        "graph_testlike_probabilities": "18F431E74DE3016CCE6F84B65986E3E5FE2667DFAFFEB8E3E8816BE2459A3030",
        "baseline_testlike_artifacts": "D7BBD9F848735FC6BFBFD0BFE0C433A9B6D1976E51EF91609BFC2BB7A247BEC1",
    },
    "v33_model_names": [
        "lr003", "lr03", "gbdt", "ridge10", "sridge30", "et2",
        "ppr015", "ppr02", "lgc", "cs01", "cs03", "knn20", "knn50",
    ],
    "expected_reference": {
        "baseline_submission_sha256": "7CE5239BEAF969FCB6FF72B8CC9644D7009A809B868A91F2875BFE10E54D2160",
        "final_submission_sha256": "F06E40E2A47F9046B557B84A9B7EE11B19EAF7262111EDACBDA8CF3A2E45B4B1",
        "expected_formal_switches": 11,
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "output" / "run1")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument(
        "--reuse-baseline",
        action="store_true",
        help="复用 output-dir/baseline 中已有的基线提交，仅用于快速等价检查。",
    )
    return parser.parse_args()


def path_config():
    return CONFIG["paths"]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def verify_input_hashes():
    observed = {}
    for key, expected in CONFIG["expected_hashes"].items():
        path = path_config()[key]
        if not path.exists():
            raise FileNotFoundError(path)
        actual = sha256(path)
        observed[key] = {"path": str(path), "sha256": actual}
        if actual != expected:
            raise RuntimeError(
                f"{key} 的 SHA256 不一致：期望 {expected}，实际 {actual}"
            )
    return observed


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_v33_training(output_dir, data_root):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = CONFIG["determinism"]["pythonhashseed"]
    env["AFAC_T1_OUTPUT_DIR"] = str(output_dir)
    env["AFAC_T1_DATA_ROOT"] = str(data_root)
    started = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as log:
        completed = subprocess.run(
            [sys.executable, str(path_config()["baseline_train_script"])],
            cwd=str(SCRIPT_DIR),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(f"基线模型训练失败，请查看日志：{log_path}")
    submission = output_dir / "A1.csv"
    summary = output_dir / "summary_baseline.json"
    oof_artifact = output_dir / "base_oof_probs.npz"
    if not submission.exists() or not summary.exists() or not oof_artifact.exists():
        raise RuntimeError("基线模型训练未生成预期的 A1.csv、summary 或 OOF 产物")
    return {
        "output_dir": str(output_dir),
        "submission": str(submission),
        "summary": str(summary),
        "base_oof_artifact": str(oof_artifact),
        "train_log": str(log_path),
        "runtime_seconds": elapsed,
        "submission_sha256": sha256(submission),
        "base_oof_sha256": sha256(oof_artifact),
    }


def confidence_features(probs):
    clipped = np.clip(probs, 1e-7, 1.0)
    ordered = np.sort(probs, axis=1)
    confidence = ordered[:, -1:]
    margin = ordered[:, -1:] - ordered[:, -2:-1]
    entropy = -(clipped * np.log(clipped)).sum(axis=1, keepdims=True)
    return confidence, margin, entropy


def build_features(
    v33_prediction,
    gnn_probs,
    degree,
    coverage,
    attribute_nnz,
    propensity,
    num_classes,
):
    gnn_prediction = gnn_probs.argmax(axis=1)
    row = np.arange(len(gnn_probs))
    v33_onehot = np.eye(num_classes, dtype=np.float32)[v33_prediction]
    gnn_onehot = np.eye(num_classes, dtype=np.float32)[gnn_prediction]
    pair_onehot = np.eye(num_classes * num_classes, dtype=np.float32)[
        v33_prediction * num_classes + gnn_prediction
    ]
    confidence, margin, entropy = confidence_features(gnn_probs)
    graph_features = np.column_stack(
        [
            np.log1p(degree),
            degree == 0,
            coverage,
            np.log1p(attribute_nnz),
            propensity,
            gnn_probs[row, v33_prediction],
            confidence[:, 0] - gnn_probs[row, v33_prediction],
        ]
    ).astype(np.float32)
    return np.hstack(
        [
            gnn_probs.astype(np.float32),
            confidence,
            margin,
            entropy,
            v33_onehot,
            gnn_onehot,
            pair_onehot,
            graph_features,
        ]
    ).astype(np.float32)


def selector_factory():
    selector_name = CONFIG["selector"]["model"]
    if selector_name == "lr":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.05,
                max_iter=3000,
                solver="lbfgs",
                random_state=CONFIG["determinism"]["global_seed"],
            ),
        )
    if selector_name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=800,
            max_depth=6,
            min_samples_leaf=12,
            max_features=0.7,
            n_jobs=CONFIG["determinism"]["selector_n_jobs"],
            random_state=CONFIG["determinism"]["global_seed"],
        )
    if selector_name == "hist_gbdt":
        return HistGradientBoostingClassifier(
            max_iter=160,
            max_depth=3,
            learning_rate=0.035,
            min_samples_leaf=20,
            l2_regularization=8.0,
            random_state=CONFIG["determinism"]["global_seed"],
        )
    raise ValueError(f"未知 selector 模型：{selector_name}")


def route(base_prediction, graph_prediction, disagreement_loc, choose_graph):
    output = base_prediction.copy()
    selected_loc = disagreement_loc[choose_graph]
    output[selected_loc] = graph_prediction[selected_loc]
    return output


def compare_predictions(reference, candidate, labels):
    return {
        "changed": int((reference != candidate).sum()),
        "corrected": int(((reference != labels) & (candidate == labels)).sum()),
        "worsened": int(((reference == labels) & (candidate != labels)).sum()),
        "net_corrected": int((candidate == labels).sum() - (reference == labels).sum()),
    }


def threshold_report(
    probabilities,
    threshold,
    target,
    disagreement_loc,
    v33_prediction,
    gnn_prediction,
    labels,
    split_locs,
    canonical_mask,
):
    choose = probabilities >= threshold
    routed = route(v33_prediction, gnn_prediction, disagreement_loc, choose)
    fold_net_rows = []
    for _, val_loc in split_locs:
        global_loc = disagreement_loc[val_loc]
        selected = choose[val_loc]
        if selected.any():
            selected_global = global_loc[selected]
            fold_net_rows.append(
                int(
                    (gnn_prediction[selected_global] == labels[selected_global]).sum()
                    - (v33_prediction[selected_global] == labels[selected_global]).sum()
                )
            )
        else:
            fold_net_rows.append(0)
    selected_count = int(choose.sum())
    return {
        "threshold": float(threshold),
        "selected_count": selected_count,
        "selected_gnn_correct_rate": float(target[choose].mean()) if selected_count else None,
        "net_correct_rows": int((routed == labels).sum() - (v33_prediction == labels).sum()),
        "accuracy": float(accuracy_score(labels, routed)),
        "delta_vs_v33": float(
            accuracy_score(labels, routed) - accuracy_score(labels, v33_prediction)
        ),
        "canonical_accuracy": float(
            accuracy_score(labels[canonical_mask], routed[canonical_mask])
        ),
        "canonical_delta": float(
            accuracy_score(labels[canonical_mask], routed[canonical_mask])
            - accuracy_score(labels[canonical_mask], v33_prediction[canonical_mask])
        ),
        "selector_fold_net_correct_rows": fold_net_rows,
        "selector_fold_min_net_correct_rows": int(min(fold_net_rows)),
    }


def rebuild_v33_oof_from_base_artifact(base_oof_path, train_idx, labels, num_classes):
    classic = load_module("baseline_for_router", path_config()["baseline_train_script"])
    cache = np.load(base_oof_path)
    if not np.array_equal(cache["train_idx"], train_idx):
        raise RuntimeError("本次基线 OOF 产物的 train_idx 与当前数据不一致")
    model_names = CONFIG["v33_model_names"]
    base_probs = {name: cache[name].astype(np.float32) for name in model_names}
    targets = labels[train_idx]
    v33_oof = np.zeros((len(train_idx), num_classes), dtype=np.float32)
    level2_splitter = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=CONFIG["determinism"]["level2_oof_seed"],
    )
    for fit_loc, val_loc in level2_splitter.split(train_idx, targets):
        meta = LogisticRegression(
            C=0.08,
            penalty="l2",
            max_iter=5000,
            solver="lbfgs",
            n_jobs=CONFIG["determinism"]["v33_n_jobs"],
            random_state=CONFIG["determinism"]["global_seed"],
        )
        meta.fit(
            classic.make_stack_features(base_probs, fit_loc, model_names),
            targets[fit_loc],
        )
        meta_probs = meta.predict_proba(
            classic.make_stack_features(base_probs, val_loc, model_names)
        )
        weights, _ = classic.search_blend_weights(
            base_probs,
            targets,
            fit_loc,
            model_names,
            n_iter=3000,
            seed=CONFIG["determinism"]["global_seed"],
        )
        weighted_probs = classic.weighted_blend(
            base_probs,
            val_loc,
            model_names,
            weights,
        )
        v33_oof[val_loc] = 0.55 * meta_probs + 0.45 * weighted_probs
    return v33_oof.astype(np.float32)


def run_correction(output_dir, data_root, v33_submission_path, base_oof_path, input_hashes):
    started = time.perf_counter()
    correction_dir = output_dir / "correction"
    correction_dir.mkdir(parents=True, exist_ok=True)

    adj, attributes, labels_all, train_idx, test_idx = graph_base.load_data(data_root)
    labels_all = labels_all.astype(np.int64)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    targets = labels_all[train_idx]
    num_classes = int(targets.max()) + 1
    _, symmetric, _ = graph_base.prepare_graph_views(adj)
    degree_all = np.asarray(symmetric.sum(axis=1)).reshape(-1)
    attribute_nnz_all = np.diff(attributes.indptr)

    propensity_frame = pd.read_csv(path_config()["domain_propensity"])
    propensity_all = np.full(len(labels_all), np.nan, dtype=np.float32)
    propensity_all[propensity_frame["node_idx"].to_numpy(dtype=np.int64)] = (
        propensity_frame["test_propensity"].to_numpy(dtype=np.float32)
    )
    if not np.isfinite(propensity_all).all():
        raise RuntimeError("domain propensity 缺少部分节点")

    v33_oof = rebuild_v33_oof_from_base_artifact(
        base_oof_path,
        train_idx,
        labels_all,
        num_classes,
    )
    gnn_train = np.load(path_config()["graph_correction_artifacts"])
    if not np.array_equal(gnn_train["train_idx"], train_idx):
        raise RuntimeError("图纠错训练节点顺序与当前数据不一致")
    if not np.array_equal(gnn_train["test_idx"], test_idx):
        raise RuntimeError("图纠错测试节点顺序与当前数据不一致")

    v33_oof_prediction = v33_oof.argmax(axis=1)
    gnn_oof_probs = gnn_train["oof_probs"].astype(np.float32)
    gnn_oof_prediction = gnn_oof_probs.argmax(axis=1)
    oof_features = build_features(
        v33_oof_prediction,
        gnn_oof_probs,
        degree_all[train_idx],
        gnn_train["oof_coverage"].astype(bool),
        attribute_nnz_all[train_idx],
        propensity_all[train_idx],
        num_classes,
    )
    disagreement = (
        (v33_oof_prediction != gnn_oof_prediction)
        & gnn_train["oof_coverage"].astype(bool)
    )
    disagreement_loc = np.flatnonzero(disagreement)
    disagreement_features = oof_features[disagreement]
    choose_gnn_target = (
        gnn_oof_prediction[disagreement] == targets[disagreement]
    ).astype(np.int64)

    selector_splitter = StratifiedKFold(
        5,
        shuffle=True,
        random_state=CONFIG["determinism"]["selector_split_seed"],
    )
    selector_splits = list(
        selector_splitter.split(disagreement_features, choose_gnn_target)
    )
    _, canonical_loc = train_test_split(
        np.arange(len(train_idx)),
        test_size=0.2,
        random_state=CONFIG["determinism"]["canonical_split_seed"],
        stratify=targets,
    )
    canonical_mask = np.zeros(len(train_idx), dtype=bool)
    canonical_mask[canonical_loc] = True

    crossfit_probabilities = np.zeros(len(disagreement_loc), dtype=np.float32)
    for fit_loc, val_loc in selector_splits:
        model = selector_factory()
        model.fit(disagreement_features[fit_loc], choose_gnn_target[fit_loc])
        crossfit_probabilities[val_loc] = model.predict_proba(
            disagreement_features[val_loc]
        )[:, 1]

    selected_threshold = CONFIG["selector"]["threshold"]
    selected_report = threshold_report(
        crossfit_probabilities,
        selected_threshold,
        choose_gnn_target,
        disagreement_loc,
        v33_oof_prediction,
        gnn_oof_prediction,
        targets,
        selector_splits,
        canonical_mask,
    )
    if selected_report["selected_count"] < CONFIG["selector"]["minimum_selected_count"]:
        raise RuntimeError("selector 在 OOF 上选择的样本数过少")
    if CONFIG["selector"]["require_strict_positive_fold_net"]:
        if selected_report["selector_fold_min_net_correct_rows"] <= 0:
            raise RuntimeError("selector 未满足每个折都严格正收益的门槛")

    final_selector = selector_factory()
    final_selector.fit(disagreement_features, choose_gnn_target)

    testlike_summary = evaluate_testlike(
        final_selector,
        selected_threshold,
        train_idx,
        labels_all,
        symmetric,
        degree_all,
        attribute_nnz_all,
        propensity_all,
        num_classes,
    )

    v33_submission = pd.read_csv(v33_submission_path)
    if not np.array_equal(v33_submission["test_idx"].to_numpy(dtype=np.int64), test_idx):
        raise RuntimeError("本次生成的基线提交文件测试节点顺序不一致")
    v33_test_prediction = v33_submission["label"].to_numpy(dtype=np.int64)
    gnn_test_probs = gnn_train["test_probs_fold_full_context"].astype(np.float32)
    gnn_test_prediction = gnn_test_probs.argmax(axis=1)
    test_coverage = np.asarray(symmetric[test_idx][:, train_idx].sum(axis=1)).reshape(-1) > 0
    test_features = build_features(
        v33_test_prediction,
        gnn_test_probs,
        degree_all[test_idx],
        test_coverage,
        attribute_nnz_all[test_idx],
        propensity_all[test_idx],
        num_classes,
    )
    test_disagreement = (v33_test_prediction != gnn_test_prediction) & test_coverage
    test_disagreement_loc = np.flatnonzero(test_disagreement)
    test_choose = np.zeros(len(test_disagreement_loc), dtype=bool)
    if len(test_disagreement_loc):
        test_choose = (
            final_selector.predict_proba(test_features[test_disagreement])[:, 1]
            >= selected_threshold
        )
    routed_test_prediction = route(
        v33_test_prediction,
        gnn_test_prediction,
        test_disagreement_loc,
        test_choose,
    )

    output_submission = output_dir / "A1.csv"
    submission = v33_submission.copy()
    submission["label"] = routed_test_prediction
    submission.to_csv(output_submission, index=False)

    oof_artifact = correction_dir / "baseline_oof_rebuilt_for_selector.npz"
    np.savez_compressed(
        oof_artifact,
        train_idx=train_idx,
        labels=targets,
        baseline_oof=v33_oof,
        graph_oof=gnn_oof_probs,
        disagreement_loc=disagreement_loc,
        selector_crossfit_probabilities=crossfit_probabilities,
    )

    summary = {
        "version": VERSION,
        "status": "completed",
        "config": serialize_config(),
        "input_hashes": input_hashes,
        "baseline_oof_source": {
            "path": str(base_oof_path),
            "sha256": sha256(base_oof_path),
            "note": "由本次 baseline_train.py 训练阶段导出，而非历史冻结产物。",
        },
        "protocol": {
            "default_prediction": "本次运行训练出的基线 hard prediction",
            "correction_source": "图纠错模型的 covered prediction",
            "selector_training": "在 covered 的基线/图纠错 OOF 分歧样本上做五折交叉拟合",
            "selector_inputs_available_identically_at_test": True,
            "testlike_used_for_selection": False,
        },
        "oracle": {
            "oof_disagreement_count": int(disagreement.sum()),
            "oof_baseline_accuracy": float(accuracy_score(targets, v33_oof_prediction)),
            "oof_graph_accuracy": float(accuracy_score(targets, gnn_oof_prediction)),
            "oof_oracle_accuracy": float(
                ((v33_oof_prediction == targets) | (gnn_oof_prediction == targets)).mean()
            ),
        },
        "selected": {
            "model": CONFIG["selector"]["model"],
            **selected_report,
        },
        "testlike": testlike_summary,
        "formal_test": {
            "baseline_graph_covered_disagreement_count": int(test_disagreement.sum()),
            "selected_switch_count": int(test_choose.sum()),
            "baseline_vs_final_changed_count": int(
                (v33_test_prediction != routed_test_prediction).sum()
            ),
        },
        "artifacts": {
            "submission": str(output_submission),
            "submission_sha256": sha256(output_submission),
            "rebuilt_oof": str(oof_artifact),
            "rebuilt_oof_sha256": sha256(oof_artifact),
        },
        "runtime_seconds": float(time.perf_counter() - started),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def evaluate_testlike(
    final_selector,
    selected_threshold,
    train_idx,
    labels_all,
    symmetric,
    degree_all,
    attribute_nnz_all,
    propensity_all,
    num_classes,
):
    v33_testlike = np.load(path_config()["baseline_testlike_artifacts"])
    gnn_testlike = np.load(path_config()["graph_testlike_probabilities"])
    testlike_idx = v33_testlike["testlike_idx"]
    if not np.array_equal(testlike_idx, gnn_testlike["testlike_idx"]):
        raise RuntimeError("test-like 节点顺序不一致")
    development_idx = train_idx[~np.isin(train_idx, testlike_idx)]
    testlike_coverage = np.asarray(
        symmetric[testlike_idx][:, development_idx].sum(axis=1)
    ).reshape(-1) > 0
    v33_prediction = v33_testlike["v33_combo_probs"].argmax(axis=1)
    gnn_probs = gnn_testlike["v32_probs"].astype(np.float32)
    gnn_prediction = gnn_probs.argmax(axis=1)
    features = build_features(
        v33_prediction,
        gnn_probs,
        degree_all[testlike_idx],
        testlike_coverage,
        attribute_nnz_all[testlike_idx],
        propensity_all[testlike_idx],
        num_classes,
    )
    disagreement = (v33_prediction != gnn_prediction) & testlike_coverage
    disagreement_loc = np.flatnonzero(disagreement)
    choose = np.zeros(len(disagreement_loc), dtype=bool)
    if len(disagreement_loc):
        choose = (
            final_selector.predict_proba(features[disagreement])[:, 1]
            >= selected_threshold
        )
    routed = route(v33_prediction, gnn_prediction, disagreement_loc, choose)
    labels = v33_testlike["labels"]
    base_accuracy = float(accuracy_score(labels, v33_prediction))
    routed_accuracy = float(accuracy_score(labels, routed))
    return {
        "baseline_accuracy": base_accuracy,
        "routed_accuracy": routed_accuracy,
        "delta": float(routed_accuracy - base_accuracy),
        "disagreement_count": int(disagreement.sum()),
        "changes": compare_predictions(v33_prediction, routed, labels),
    }


def serialize_config():
    def convert(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value

    return convert(CONFIG)


def main():
    started = time.perf_counter()
    args = parse_args()
    graph_base.seed_everything(CONFIG["determinism"]["global_seed"])
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.data_root is not None:
        CONFIG["paths"]["data_root"] = args.data_root
    data_root = CONFIG["paths"]["data_root"]

    input_hashes = verify_input_hashes()
    v33_output_dir = output_dir / "baseline"
    if args.reuse_baseline:
        v33_submission = v33_output_dir / "A1.csv"
        base_oof_artifact = v33_output_dir / "base_oof_probs.npz"
        if not v33_submission.exists():
            packaged_baseline = SCRIPT_DIR / "artifacts" / "reference_baseline_A1.csv"
            if packaged_baseline.exists():
                v33_submission = packaged_baseline
        if not base_oof_artifact.exists():
            packaged_oof = SCRIPT_DIR / "artifacts" / "reference_base_oof_probs.npz"
            if packaged_oof.exists():
                base_oof_artifact = packaged_oof
        if not v33_submission.exists():
            raise FileNotFoundError(v33_submission)
        if not base_oof_artifact.exists():
            raise FileNotFoundError(base_oof_artifact)
        v33_run = {
            "output_dir": str(v33_output_dir),
            "submission": str(v33_submission),
            "base_oof_artifact": str(base_oof_artifact),
            "summary": str(v33_output_dir / "summary_baseline.json"),
            "train_log": str(v33_output_dir / "train.log"),
            "runtime_seconds": 0.0,
            "submission_sha256": sha256(v33_submission),
            "base_oof_sha256": sha256(base_oof_artifact),
            "reused": True,
        }
    else:
        v33_run = run_v33_training(v33_output_dir, data_root)

    correction = run_correction(
        output_dir,
        data_root,
        Path(v33_run["submission"]),
        Path(v33_run["base_oof_artifact"]),
        input_hashes,
    )
    run_summary = {
        "version": VERSION,
        "status": "completed",
        "v33_run": v33_run,
        "correction_summary": str(output_dir / "summary.json"),
        "final_submission": correction["artifacts"],
        "total_runtime_seconds": float(time.perf_counter() - started),
    }
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(run_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
