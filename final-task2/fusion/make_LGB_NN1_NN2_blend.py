"""Blend LGB, deeper NN (NN1), and medium NN (NN2) at 4:3:3."""
from __future__ import annotations

import gc
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/home/cyp/speedsci/AFAC2026")
BLEND_ROOT = ROOT / "task2_blend"
OUTPUT_DIR = BLEND_ROOT / "output"
DATA_ROOT = ROOT / "data" / "rec_data"
NN2_ROOT = ROOT / "task2_nn_medium" / "nn2_fivefold"
NN2_SOURCE = ROOT / "task2_nn_medium" / "[0.5080]train_task2_nn2_medium.py"
NN2_SCORES_PATH = OUTPUT_DIR / "test_candidate_scores_nn2.parquet"
WEIGHTS = {"lgb_score": 0.4, "nn1_score": 0.3, "nn2_score": 0.3}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_nn2_scores(nn2) -> pd.DataFrame:
    if NN2_SCORES_PATH.exists():
        print(f"Reusing NN2 scores: {NN2_SCORES_PATH}")
        return pd.read_parquet(NN2_SCORES_PATH)

    train = pd.read_csv(DATA_ROOT / "train.csv")
    test = pd.read_csv(DATA_ROOT / "test.csv")
    user = pd.read_csv(DATA_ROOT / "user.csv")
    train, test, cat_cols = nn2.prepare_frames(train, test, user)
    stats = nn2.RecallStats(train, cat_cols)
    ranker = nn2.NNRankerWrapper.load(NN2_ROOT / "model_nn_medium.pt")
    with (NN2_ROOT / "feature_cols.json").open(encoding="utf-8") as handle:
        cols = json.load(handle)

    parts = []
    batch_size = nn2.PRED_BATCH_SIZE
    for start in range(0, len(test), batch_size):
        stop = min(start + batch_size, len(test))
        rows = []
        for offset, (_, row) in enumerate(test.iloc[start:stop].iterrows()):
            qid = start + offset
            raw, dedup, cat_vals = nn2.prep_row(row, cat_cols)
            candidates, meta = stats.generate_candidates(row)
            for iid in candidates:
                feats = stats.pair_features(
                    raw,
                    dedup,
                    cat_vals,
                    iid,
                    meta.get(iid, nn2.CandidateMeta()),
                )
                feats["qid"] = qid
                feats["uid"] = row["uid"]
                feats["candidate_iid"] = iid
                rows.append(feats)
        frame = pd.DataFrame(rows)
        frame["nn2_score"] = ranker.predict(nn2.fm(frame, cols)).astype(np.float32)
        parts.append(frame[["qid", "uid", "candidate_iid", "nn2_score"]])
        print(f"NN2 test scores: {stop}/{len(test)}")
        del frame, rows
        gc.collect()

    scores = pd.concat(parts, ignore_index=True)
    scores.to_parquet(NN2_SCORES_PATH, index=False)
    print(f"Saved NN2 scores: {NN2_SCORES_PATH}")
    return scores


def normalized(frame: pd.DataFrame, score_col: str, method: str) -> pd.Series:
    group = frame.groupby("qid", sort=False)[score_col]
    if method == "rank":
        rank = group.rank(method="first", ascending=False, na_option="keep")
        size = group.transform("count").astype(np.float32)
        value = (size - rank) / (size - 1.0).clip(lower=1.0)
        return value.fillna(0.0).astype(np.float32)

    mean = group.transform("mean")
    std = group.transform("std")
    value = (frame[score_col] - mean) / std.clip(lower=1e-6)
    floor = value.groupby(frame["qid"], sort=False).transform("min").fillna(-1.0) - 0.5
    return value.fillna(floor).astype(np.float32)


def make_submission(
    scores: pd.DataFrame,
    sample: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    scored = scores.copy()
    scored["blend_score"] = 0.0
    for score_col, weight in WEIGHTS.items():
        scored["blend_score"] += weight * normalized(scored, score_col, method)

    predictions = {}
    for _, group in scored.groupby("qid", sort=False):
        ranked = group.sort_values(
            ["blend_score", "candidate_iid"],
            ascending=[False, True],
            kind="stable",
        )["candidate_iid"].drop_duplicates().head(10)
        predictions[str(group["uid"].iloc[0])] = ",".join(ranked)
    output = sample.copy()
    output["prediction"] = output["uid"].astype(str).map(predictions)
    return output


def main():
    os.environ["DATA_ROOT"] = str(DATA_ROOT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nn2 = load_module("nn2_medium_source", NN2_SOURCE)
    nn2_scores = build_nn2_scores(nn2)

    base_scores = pd.read_parquet(OUTPUT_DIR / "test_candidate_scores.parquet")
    base_scores = base_scores.rename(columns={"nn_score": "nn1_score"})
    keys = ["qid", "uid", "candidate_iid"]
    if base_scores.duplicated(keys).any() or nn2_scores.duplicated(keys).any():
        raise ValueError("Candidate score files contain duplicate keys")
    scores = base_scores.merge(nn2_scores, on=keys, how="outer", validate="one_to_one")
    scores.to_parquet(OUTPUT_DIR / "test_candidate_scores_three_models.parquet", index=False)

    sample = pd.read_csv(DATA_ROOT / "sample_submission.csv")
    outputs = {}
    for method in ("rank", "zscore"):
        path = OUTPUT_DIR / f"A2_blend_LGB40_NN1_30_NN2_30_{method}.csv"
        make_submission(scores, sample, method).to_csv(path, index=False)
        outputs[method] = str(path)
        print(f"Saved {method}: {path}")

    primary_path = OUTPUT_DIR / "A2_blend_LGB40_NN1_30_NN2_30.csv"
    make_submission(scores, sample, "zscore").to_csv(primary_path, index=False)
    overlap = {
        "base_candidates": int(base_scores[keys].shape[0]),
        "nn2_candidates": int(nn2_scores[keys].shape[0]),
        "union_candidates": int(scores.shape[0]),
        "shared_candidates": int(scores["lgb_score"].notna().mul(scores["nn2_score"].notna()).sum()),
    }
    (OUTPUT_DIR / "manifest_three_model_40_30_30.json").write_text(
        json.dumps(
            {
                "weights": WEIGHTS,
                "primary_method": "zscore",
                "primary": str(primary_path),
                "outputs": outputs,
                "candidate_overlap": overlap,
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Primary output: {primary_path}")
    print(json.dumps(overlap, indent=2))


if __name__ == "__main__":
    main()
