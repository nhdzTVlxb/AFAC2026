#!/usr/bin/env python
# coding: utf-8

import os
import subprocess
import sys


if os.environ.get("PYTHONHASHSEED") != "0":
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    completed = subprocess.run(
        [sys.executable, os.path.abspath(__file__), *sys.argv[1:]],
        env=env,
        check=False,
    )
    raise SystemExit(completed.returncode)

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import base_v7 as v7


VERSION = "t1_gnn_v32_equal_v23_v26_v31_covered_uncovered_v20"
SCRIPT_DIR = Path(__file__).resolve().parent
GNN_DIR = SCRIPT_DIR.parent
V33_TESTLIKE_ACCURACY = 0.864091


def validate_probabilities(name, probs, expected_rows, num_classes):
    if probs.shape != (expected_rows, num_classes):
        raise RuntimeError(
            f"{name} shape mismatch: {probs.shape} != "
            f"{(expected_rows, num_classes)}"
        )
    if not np.isfinite(probs).all():
        raise RuntimeError(f"{name} contains non-finite probabilities")
    if (probs < 0).any():
        raise RuntimeError(f"{name} contains negative probabilities")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError(f"{name} rows do not sum to one")


def blend_covered(v23_probs, v26_probs, v31_probs):
    return ((v23_probs + v26_probs + v31_probs) / 3.0).astype(np.float32)


def metric_bundle(labels, probs, degree, coverage, num_classes):
    prediction = probs.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "class_recall": v7.class_recall(labels, prediction, num_classes),
        "buckets": v7.bucket_metrics(labels, prediction, degree, coverage),
    }


def main():
    output_dir = SCRIPT_DIR / "output" / "run1"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = v7.resolve_data_root(None)
    adj, _, labels_all, train_idx, test_idx = v7.load_data(data_root)
    num_classes = int(labels_all[train_idx].max()) + 1
    directed, symmetric, _ = v7.prepare_graph_views(adj)
    _, degree = v7.build_structural_features(directed, symmetric)

    v7_formal = np.load(GNN_DIR / "V7" / "output" / "run1" / "artifacts_run1.npz")
    v20_formal = np.load(GNN_DIR / "V20" / "output" / "run1" / "artifacts_run1.npz")
    v23_formal = np.load(GNN_DIR / "V23" / "output" / "run1" / "artifacts_run1.npz")
    v26_formal = np.load(GNN_DIR / "V26" / "output" / "run1" / "artifacts_run1.npz")
    v27_formal = np.load(GNN_DIR / "V27" / "output" / "run1" / "artifacts_run1.npz")
    v28_formal = np.load(GNN_DIR / "V28" / "output" / "run1" / "artifacts_run1.npz")
    v30_formal = np.load(GNN_DIR / "V30" / "output" / "run1" / "artifacts_run1.npz")
    v31_formal = np.load(GNN_DIR / "V31" / "output" / "run1" / "artifacts_run1.npz")
    for source_name, source in (
        ("V20", v20_formal),
        ("V23", v23_formal),
        ("V26", v26_formal),
        ("V27", v27_formal),
        ("V28", v28_formal),
        ("V30", v30_formal),
        ("V31", v31_formal),
    ):
        if not np.array_equal(v7_formal["train_idx"], source["train_idx"]):
            raise RuntimeError(f"{source_name} formal train order mismatch")
        if not np.array_equal(v7_formal["test_idx"], source["test_idx"]):
            raise RuntimeError(f"{source_name} formal test order mismatch")
    coverage = v7_formal["oof_coverage"].astype(bool)
    if not (
        np.array_equal(coverage, v20_formal["oof_coverage"].astype(bool))
        and np.array_equal(coverage, v26_formal["oof_coverage"].astype(bool))
    ):
        raise RuntimeError("Formal OOF coverage mismatch")
    oof_probs = v20_formal["oof_probs"].copy()
    oof_probs[coverage] = blend_covered(
        v23_formal["oof_probs"][coverage],
        v26_formal["oof_probs"][coverage],
        v31_formal["oof_probs"][coverage],
    )
    test_coverage = np.asarray(
        symmetric[test_idx][:, train_idx].sum(axis=1)
    ).reshape(-1) > 0
    test_outer_probs = v20_formal["test_probs_fold_outer_context"].copy()
    test_outer_probs[test_coverage] = blend_covered(
        v23_formal["test_probs_fold_outer_context"][test_coverage],
        v26_formal["test_probs_fold_outer_context"][test_coverage],
        v31_formal["test_probs_fold_outer_context"][test_coverage],
    )
    test_full_probs = v20_formal["test_probs_fold_full_context"].copy()
    test_full_probs[test_coverage] = blend_covered(
        v23_formal["test_probs_fold_full_context"][test_coverage],
        v26_formal["test_probs_fold_full_context"][test_coverage],
        v31_formal["test_probs_fold_full_context"][test_coverage],
    )
    validate_probabilities("OOF", oof_probs, len(train_idx), num_classes)
    validate_probabilities("test outer", test_outer_probs, len(test_idx), num_classes)
    validate_probabilities("test full", test_full_probs, len(test_idx), num_classes)

    targets = labels_all[train_idx]
    v7_metrics = metric_bundle(
        targets,
        v7_formal["oof_probs"],
        degree[train_idx],
        coverage,
        num_classes,
    )
    v30_metrics = metric_bundle(
        targets, oof_probs, degree[train_idx], coverage, num_classes
    )
    v27_reference_metrics = metric_bundle(
        targets,
        v27_formal["oof_probs"],
        degree[train_idx],
        coverage,
        num_classes,
    )
    v28_reference_metrics = metric_bundle(
        targets,
        v28_formal["oof_probs"],
        degree[train_idx],
        coverage,
        num_classes,
    )
    v30_reference_metrics = metric_bundle(
        targets,
        v30_formal["oof_probs"],
        degree[train_idx],
        coverage,
        num_classes,
    )
    _, canonical_idx = train_test_split(
        train_idx,
        test_size=0.2,
        random_state=2026,
        stratify=targets,
    )
    canonical_loc = np.flatnonzero(np.isin(train_idx, canonical_idx))
    v7_canonical = float(accuracy_score(
        targets[canonical_loc],
        v7_formal["oof_probs"][canonical_loc].argmax(axis=1),
    ))
    v30_canonical = float(accuracy_score(
        targets[canonical_loc], oof_probs[canonical_loc].argmax(axis=1)
    ))
    v27_canonical = float(accuracy_score(
        targets[canonical_loc],
        v27_formal["oof_probs"][canonical_loc].argmax(axis=1),
    ))
    v28_canonical = float(accuracy_score(
        targets[canonical_loc],
        v28_formal["oof_probs"][canonical_loc].argmax(axis=1),
    ))
    v30_reference_canonical = float(accuracy_score(
        targets[canonical_loc],
        v30_formal["oof_probs"][canonical_loc].argmax(axis=1),
    ))

    v7_testlike = np.load(GNN_DIR / "V7" / "output" / "testlike_probs.npz")
    v20_testlike = np.load(GNN_DIR / "V20" / "output" / "testlike" / "probabilities.npz")
    v23_testlike = np.load(GNN_DIR / "V23" / "output" / "testlike" / "probabilities.npz")
    v26_testlike = np.load(GNN_DIR / "V26" / "output" / "testlike" / "probabilities.npz")
    v27_testlike = np.load(GNN_DIR / "V27" / "output" / "testlike" / "probabilities.npz")
    v28_testlike = np.load(GNN_DIR / "V28" / "output" / "testlike" / "probabilities.npz")
    v30_testlike = np.load(GNN_DIR / "V30" / "output" / "testlike" / "probabilities.npz")
    v31_testlike = np.load(GNN_DIR / "V31" / "output" / "testlike" / "probabilities.npz")
    testlike_idx = v7_testlike["testlike_idx"]
    if not (
        np.array_equal(testlike_idx, v20_testlike["testlike_idx"])
        and np.array_equal(testlike_idx, v23_testlike["testlike_idx"])
        and np.array_equal(testlike_idx, v26_testlike["testlike_idx"])
        and np.array_equal(testlike_idx, v27_testlike["testlike_idx"])
        and np.array_equal(testlike_idx, v28_testlike["testlike_idx"])
        and np.array_equal(testlike_idx, v30_testlike["testlike_idx"])
        and np.array_equal(testlike_idx, v31_testlike["testlike_idx"])
    ):
        raise RuntimeError("Test-like node order mismatch")
    development_idx = train_idx[~np.isin(train_idx, testlike_idx)]
    testlike_coverage = np.asarray(
        symmetric[testlike_idx][:, development_idx].sum(axis=1)
    ).reshape(-1) > 0
    testlike_probs = v20_testlike["v20_probs"].copy()
    testlike_probs[testlike_coverage] = blend_covered(
        v23_testlike["v23_probs"][testlike_coverage],
        v26_testlike["v26_probs"][testlike_coverage],
        v31_testlike["v31_probs"][testlike_coverage],
    )
    validate_probabilities(
        "test-like", testlike_probs, len(testlike_idx), num_classes
    )
    testlike_targets = v7_testlike["labels"]
    v7_testlike_metrics = metric_bundle(
        testlike_targets,
        v7_testlike["v7_probs"],
        degree[testlike_idx],
        testlike_coverage,
        num_classes,
    )
    v30_testlike_metrics = metric_bundle(
        testlike_targets,
        testlike_probs,
        degree[testlike_idx],
        testlike_coverage,
        num_classes,
    )
    v27_reference_testlike_metrics = metric_bundle(
        testlike_targets,
        v27_testlike["v27_probs"],
        degree[testlike_idx],
        testlike_coverage,
        num_classes,
    )
    v28_reference_testlike_metrics = metric_bundle(
        testlike_targets,
        v28_testlike["v28_probs"],
        degree[testlike_idx],
        testlike_coverage,
        num_classes,
    )
    v30_reference_testlike_metrics = metric_bundle(
        testlike_targets,
        v30_testlike["v30_probs"],
        degree[testlike_idx],
        testlike_coverage,
        num_classes,
    )
    gates = {
        "oof_improved": v30_metrics["accuracy"] > v7_metrics["accuracy"],
        "canonical_not_regressed": v30_canonical >= v7_canonical,
        "oof_improved_vs_v27": (
            v30_metrics["accuracy"] > v27_reference_metrics["accuracy"]
        ),
        "canonical_improved_vs_v27": v30_canonical > v27_canonical,
        "oof_improved_vs_v28": (
            v30_metrics["accuracy"] > v28_reference_metrics["accuracy"]
        ),
        "canonical_not_regressed_vs_v28": v30_canonical >= v28_canonical,
        "oof_improved_vs_v30": (
            v30_metrics["accuracy"] > v30_reference_metrics["accuracy"]
        ),
        "canonical_improved_vs_v30": (
            v30_canonical > v30_reference_canonical
        ),
        "oof_covered_not_regressed": (
            v30_metrics["buckets"]["covered"]["accuracy"]
            >= v7_metrics["buckets"]["covered"]["accuracy"]
        ),
        "oof_uncovered_not_regressed": (
            v30_metrics["buckets"]["uncovered"]["accuracy"]
            >= v7_metrics["buckets"]["uncovered"]["accuracy"]
        ),
        "testlike_improved": (
            v30_testlike_metrics["accuracy"] > v7_testlike_metrics["accuracy"]
        ),
        "testlike_improved_vs_v27": (
            v30_testlike_metrics["accuracy"]
            > v27_reference_testlike_metrics["accuracy"]
        ),
        "testlike_improved_vs_v28": (
            v30_testlike_metrics["accuracy"]
            > v28_reference_testlike_metrics["accuracy"]
        ),
        "testlike_not_regressed_vs_v30": (
            v30_testlike_metrics["accuracy"]
            >= v30_reference_testlike_metrics["accuracy"]
        ),
        "testlike_covered_not_regressed": (
            v30_testlike_metrics["buckets"]["covered"]["accuracy"]
            >= v7_testlike_metrics["buckets"]["covered"]["accuracy"]
        ),
        "testlike_uncovered_not_regressed": (
            v30_testlike_metrics["buckets"]["uncovered"]["accuracy"]
            >= v7_testlike_metrics["buckets"]["uncovered"]["accuracy"]
        ),
        "oof_covered_exact_three_expert_mean": bool(np.array_equal(
            oof_probs[coverage],
            blend_covered(
                v23_formal["oof_probs"][coverage],
                v26_formal["oof_probs"][coverage],
                v31_formal["oof_probs"][coverage],
            ),
        )),
        "oof_uncovered_exact_v20": bool(np.array_equal(
            oof_probs[~coverage], v20_formal["oof_probs"][~coverage]
        )),
        "testlike_covered_exact_three_expert_mean": bool(np.array_equal(
            testlike_probs[testlike_coverage],
            blend_covered(
                v23_testlike["v23_probs"][testlike_coverage],
                v26_testlike["v26_probs"][testlike_coverage],
                v31_testlike["v31_probs"][testlike_coverage],
            ),
        )),
        "testlike_uncovered_exact_v20": bool(np.array_equal(
            testlike_probs[~testlike_coverage],
            v20_testlike["v20_probs"][~testlike_coverage],
        )),
    }
    all_gates_passed = all(gates.values())
    artifact_path = output_dir / "artifacts_run1.npz"
    np.savez_compressed(
        artifact_path,
        train_idx=train_idx,
        test_idx=test_idx,
        oof_probs=oof_probs,
        test_probs_fold_outer_context=test_outer_probs,
        test_probs_fold_full_context=test_full_probs,
        oof_coverage=coverage,
        test_coverage=test_coverage,
    )
    submission_path = output_dir / "A1.csv"
    if all_gates_passed:
        v7.save_submission(
            data_root / "sample_submission.csv",
            submission_path,
            test_full_probs.argmax(axis=1),
        )
    summary = {
        "version": VERSION,
        "status": "all_gates_passed" if all_gates_passed else "gates_failed",
        "protocol": {
            "covered": "Equal probability mean of V23, V26, and seed-314 V31",
            "uncovered": "V20 protected distilled route",
            "routing_uses_labels": False,
        },
        "oof": {
            "v7": v7_metrics,
            "v32": v30_metrics,
            "v30_reference": v30_reference_metrics,
            "v28_reference": v28_reference_metrics,
            "v27_reference": v27_reference_metrics,
            "v7_canonical": v7_canonical,
            "v27_canonical": v27_canonical,
            "v28_canonical": v28_canonical,
            "v30_canonical": v30_reference_canonical,
            "v32_canonical": v30_canonical,
        },
        "testlike": {
            "v7": v7_testlike_metrics,
            "v32": v30_testlike_metrics,
            "v30_reference": v30_reference_testlike_metrics,
            "v28_reference": v28_reference_testlike_metrics,
            "v27_reference": v27_reference_testlike_metrics,
            "v33_reference_accuracy": V33_TESTLIKE_ACCURACY,
            "gap_to_v33": float(
                V33_TESTLIKE_ACCURACY - v30_testlike_metrics["accuracy"]
            ),
        },
        "gates": gates,
        "all_gates_passed": all_gates_passed,
        "artifacts": {
            "probabilities": str(artifact_path),
            "submission": str(submission_path) if all_gates_passed else None,
            "submission_sha256": v7.sha256(submission_path) if all_gates_passed else None,
        },
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    testlike_output_dir = SCRIPT_DIR / "output" / "testlike"
    testlike_output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        testlike_output_dir / "probabilities.npz",
        testlike_idx=testlike_idx,
        coverage=testlike_coverage,
        v7_probs=v7_testlike["v7_probs"],
        v32_probs=testlike_probs,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
