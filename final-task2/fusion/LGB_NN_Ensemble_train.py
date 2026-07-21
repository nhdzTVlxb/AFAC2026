"""T2 LGB + NN OOF fusion experiment.

This version keeps V37 and NN V1 untouched. It reuses their feature/candidate
builders, trains both rankers on the same OOF frame, searches blend weights on
the same user-level validation split, and writes one CSV per blend ratio.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path


# ============================================================
# Configuration: paths, seeds, folds, model and blend settings
# ============================================================
REQUIRED_PYTHONHASHSEED = "0"
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
if os.environ.get("PYTHONHASHSEED") != REQUIRED_PYTHONHASHSEED:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = REQUIRED_PYTHONHASHSEED
    result = subprocess.run(
        [sys.executable, os.path.abspath(__file__), *sys.argv[1:]],
        env=env,
        check=False,
    )
    raise SystemExit(result.returncode)

VERSION = "T2_LGB_NN_V1_OOF_FUSION"
VERSION_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(r"F:\CODE\competition\AFAC_\Dataset\A推荐\A推荐")
LGB_SOURCE = Path(r"F:\CODE\competition\AFAC_\Version\T2\LGB\V37\train.py")
NN_SOURCE = Path(r"F:\CODE\competition\AFAC_\Version\T2\nn\V1\train.py")
OUTPUT_DIR = VERSION_ROOT / "output"
OOF_WORK_DIR = VERSION_ROOT / "work" / "oof_folds"
SUBMISSION_NAME = "A2.csv"

RANDOM_STATE = 2025
TORCH_SEED = 2025
N_FOLDS = 5
MAX_CANDIDATES = 55
META_VALID_FRAC = 0.12
OOF_CHUNK_USERS = 2000
PRED_BATCH_SIZE = 300
LGB_MAX_ESTIMATORS = 2500
LGB_EARLY_STOPPING_ROUNDS = 120
LGB_FALLBACK_ITERATION = 600
NN_EPOCHS = 12
NN_BATCH_SIZE = 4096

# Search all tenth-step ratios. The report decides the primary output.
BLEND_RATIOS = tuple((i / 10.0, 1.0 - i / 10.0) for i in range(11))
BLEND_METHODS = ("rank", "zscore")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_environment() -> None:
    os.environ["DATA_ROOT"] = str(DATA_ROOT)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    random.seed(RANDOM_STATE)


def train_lgb(lgbmod, oof, cols):
    import lightgbm as lgb
    import numpy as np

    mt, mv = lgbmod.split_valid(oof, frac=META_VALID_FRAC)
    mve = lgbmod.natural_recall_subset(mv)
    xt = lgbmod.fm(mt, cols)
    yt = mt["label"].astype(np.int8).to_numpy()
    xve = lgbmod.fm(mve, cols)
    yve = mve["label"].astype(np.int8).to_numpy()
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "label_gain": [0, 1],
        "n_estimators": LGB_MAX_ESTIMATORS,
        "learning_rate": 0.025,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 45,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.15,
        "reg_lambda": 1.5,
        "max_bin": 127,
        "random_state": RANDOM_STATE,
        "n_jobs": 1,
        "verbosity": -1,
    }
    model = lgb.LGBMRanker(**params)
    model.fit(
        xt,
        yt,
        group=lgbmod.group_sizes(mt),
        eval_set=[(xve, yve)],
        eval_group=[lgbmod.group_sizes(mve)],
        eval_at=[10],
        callbacks=[
            lgb.early_stopping(LGB_EARLY_STOPPING_ROUNDS, verbose=True),
            lgb.log_evaluation(100),
        ],
    )
    best_iter = max(int(model.best_iteration_ or LGB_FALLBACK_ITERATION), 100)
    mv = mv.copy()
    mv["lgb_score"] = model.predict(
        lgbmod.fm(mv, cols), num_iteration=best_iter
    ).astype("float32")
    del mt, mve, xt, yt, xve, yve
    return model, best_iter, mv


def train_nn(nnmod, lgbmod, oof, cols):
    import numpy as np

    nnmod.configure_determinism()
    mt, mv = lgbmod.split_valid(oof, frac=META_VALID_FRAC)
    model = nnmod.NNRankerWrapper()
    model.fit(
        lgbmod.fm(mt, cols),
        mt["label"].astype(np.float32).to_numpy(),
        mt["qid"].to_numpy(),
        epochs=NN_EPOCHS,
        batch_size=NN_BATCH_SIZE,
    )
    mv = mv.copy()
    mv["nn_score"] = model.predict(lgbmod.fm(mv, cols)).astype("float32")
    del mt
    return model, mv


def refit_lgb(lgbmod, oof, cols, best_iter):
    import lightgbm as lgb
    import numpy as np

    ordered = oof.sort_values("qid", kind="stable").reset_index(drop=True)
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        label_gain=[0, 1],
        n_estimators=best_iter,
        learning_rate=0.025,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=45,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.15,
        reg_lambda=1.5,
        max_bin=127,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=-1,
    )
    model.fit(
        lgbmod.fm(ordered, cols),
        ordered["label"].astype(np.int8).to_numpy(),
        group=lgbmod.group_sizes(ordered),
    )
    return model


def refit_nn(nnmod, lgbmod, oof, cols):
    import numpy as np
    import torch

    # The source helper also sets inter-op threads, which PyTorch only allows
    # once per process. Reset the random streams here without touching threads.
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(TORCH_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(TORCH_SEED)
        torch.cuda.manual_seed_all(TORCH_SEED)
    model = nnmod.NNRankerWrapper()
    ordered = oof.sort_values("qid", kind="stable").reset_index(drop=True)
    model.fit(
        lgbmod.fm(ordered, cols),
        ordered["label"].astype(np.float32).to_numpy(),
        ordered["qid"].to_numpy(),
        epochs=NN_EPOCHS,
        batch_size=NN_BATCH_SIZE,
    )
    return model


def score_test(lgbmod, lgb_model, nn_model, cols, train, test, cat_cols):
    import gc
    import numpy as np
    import pandas as pd

    stats = lgbmod.RecallStats(train, cat_cols)
    all_parts = []
    for start in range(0, len(test), PRED_BATCH_SIZE):
        stop = min(start + PRED_BATCH_SIZE, len(test))
        rows = []
        qmap = {}
        for i, (_, row) in enumerate(test.iloc[start:stop].iterrows()):
            qid = start + i
            qmap[qid] = row["uid"]
            raw, dedup, cat_vals = lgbmod.prep_row(row, cat_cols)
            candidates, meta = stats.generate_candidates(row)
            for iid in candidates:
                feats = stats.pair_features(
                    raw, dedup, cat_vals, iid, meta.get(iid, lgbmod.CandidateMeta())
                )
                feats["qid"] = qid
                feats["uid"] = row["uid"]
                feats["candidate_iid"] = iid
                rows.append(feats)
        frame = pd.DataFrame(rows)
        frame["lgb_score"] = lgb_model.predict(
            lgbmod.fm(frame, cols)
        ).astype(np.float32)
        frame["nn_score"] = nn_model.predict(
            lgbmod.fm(frame, cols)
        ).astype(np.float32)
        all_parts.append(frame[["qid", "uid", "candidate_iid", "lgb_score", "nn_score"]])
        print(f"  test scores: {stop}/{len(test)}")
        del frame, rows
        gc.collect()
    return pd.concat(all_parts, ignore_index=True)


def normalized(frame, score_col: str, method: str):
    group = frame.groupby("qid", sort=False)[score_col]
    if method == "rank":
        rank = group.rank(method="first", ascending=False)
        size = group.transform("size").astype("float32")
        return ((size - rank) / (size - 1.0).clip(lower=1.0)).astype("float32")
    mean = group.transform("mean")
    std = group.transform("std").fillna(0.0)
    return ((frame[score_col] - mean) / std.clip(lower=1e-6)).astype("float32")


def blend_scores(frame, method: str, w_lgb: float):
    out = frame.copy()
    out["blend_score"] = (
        w_lgb * normalized(out, "lgb_score", method)
        + (1.0 - w_lgb) * normalized(out, "nn_score", method)
    )
    return out


def evaluate(frame, method: str, w_lgb: float):
    import numpy as np

    source = frame.loc[frame["augmented"].eq(0)].copy()
    source = blend_scores(source, method, w_lgb)
    ndcgs = []
    hits = []
    for _, group in source.groupby("qid", sort=False):
        if int(group["positive_retrieved"].iloc[0]) == 0:
            ndcgs.append(0.0)
            hits.append(0.0)
            continue
        ranked = group.sort_values(
            ["blend_score", "candidate_iid"], ascending=[False, True], kind="stable"
        ).head(10)
        hit_pos = np.flatnonzero(ranked["label"].to_numpy() == 1)
        if len(hit_pos):
            ndcgs.append(1.0 / math.log2(int(hit_pos[0]) + 2.0))
            hits.append(1.0)
        else:
            ndcgs.append(0.0)
            hits.append(0.0)
    return float(np.mean(ndcgs)), float(np.mean(hits))


def make_submission(scores, sample, method: str, w_lgb: float):
    import pandas as pd

    scored = blend_scores(scores, method, w_lgb)
    uid_to_pred = {}
    for qid, group in scored.groupby("qid", sort=False):
        ranked = group.sort_values(
            ["blend_score", "candidate_iid"], ascending=[False, True], kind="stable"
        )["candidate_iid"].drop_duplicates().tolist()
        uid_to_pred[str(group["uid"].iloc[0])] = ",".join(ranked[:10])
    out = sample.copy()
    out["prediction"] = out["uid"].astype(str).map(uid_to_pred)
    return out


def ratio_tag(w_lgb: float) -> str:
    return f"{int(round(w_lgb * 10)):02d}_{int(round((1.0 - w_lgb) * 10)):02d}"


def main():
    import numpy as np
    import pandas as pd

    configure_environment()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OOF_WORK_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"=== {VERSION} ===")
    print(f"PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')}")
    print(f"Data: {DATA_ROOT}")
    print(f"LGB source: {LGB_SOURCE}")
    print(f"NN source: {NN_SOURCE}")

    lgbmod = load_module("lgb_v37_source", LGB_SOURCE)
    nnmod = load_module("nn_v1_source", NN_SOURCE)
    os.environ["DATA_ROOT"] = str(DATA_ROOT)
    lgbmod.N_FOLDS = N_FOLDS
    lgbmod.MAX_CANDIDATES = MAX_CANDIDATES
    lgbmod.OOF_CHUNK_USERS = OOF_CHUNK_USERS

    train = pd.read_csv(DATA_ROOT / "train.csv")
    test = pd.read_csv(DATA_ROOT / "test.csv")
    user = pd.read_csv(DATA_ROOT / "user.csv")
    sample = pd.read_csv(DATA_ROOT / "sample_submission.csv")
    train, test, cat_cols = lgbmod.prepare_frames(train, test, user)

    print("\nBuilding shared 5-fold OOF frame...")
    oof, recalls = lgbmod.build_oof(
        train, cat_cols, base_dir=str(OOF_WORK_DIR.parent)
    )
    print(f"OOF recall mean={np.mean(recalls):.4f}, std={np.std(recalls):.4f}")
    cols = lgbmod.feature_cols(oof)

    print("\nTraining LGB on the shared OOF frame...")
    lgb_meta_model, best_iter, lgb_meta = train_lgb(lgbmod, oof, cols)
    print(f"LGB best iteration: {best_iter}")
    print("\nTraining NN on the same meta-training users...")
    nn_meta_model, nn_meta = train_nn(nnmod, lgbmod, oof, cols)
    meta = lgb_meta.copy()
    meta["nn_score"] = nn_meta["nn_score"].to_numpy()
    meta.to_parquet(OUTPUT_DIR / "meta_validation_scores.parquet", index=False)

    reports = []
    for method in BLEND_METHODS:
        for w_lgb, w_nn in BLEND_RATIOS:
            ndcg, hit = evaluate(meta, method, w_lgb)
            reports.append({
                "method": method,
                "w_lgb": w_lgb,
                "w_nn": w_nn,
                "ndcg10_original": ndcg,
                "hit10_original": hit,
            })
    report_df = pd.DataFrame(reports).sort_values(
        ["ndcg10_original", "hit10_original", "w_lgb"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    report_df.to_csv(OUTPUT_DIR / "fusion_search.csv", index=False)
    print("\nOOF blend search:")
    print(report_df.head(10).to_string(index=False))
    best = report_df.iloc[0].to_dict()

    print("\nRefitting LGB and NN on all shared OOF users...")
    final_lgb = refit_lgb(lgbmod, oof, cols, int(best_iter))
    final_nn = refit_nn(nnmod, lgbmod, oof, cols)
    del lgb_meta_model, nn_meta_model, lgb_meta, nn_meta, oof

    print("\nScoring all test candidates...")
    test_scores = score_test(lgbmod, final_lgb, final_nn, cols, train, test, cat_cols)
    test_scores.to_parquet(OUTPUT_DIR / "test_candidate_scores.parquet", index=False)

    generated = []
    for method in BLEND_METHODS:
        for w_lgb, _ in BLEND_RATIOS:
            out = make_submission(test_scores, sample, method, w_lgb)
            path = OUTPUT_DIR / f"A2_{method}_{ratio_tag(w_lgb)}.csv"
            out.to_csv(path, index=False)
            generated.append(str(path))

    best_path = OUTPUT_DIR / SUBMISSION_NAME
    best_out = make_submission(
        test_scores, sample, str(best["method"]), float(best["w_lgb"])
    )
    best_out.to_csv(best_path, index=False)
    manifest = {
        "version": VERSION,
        "best": best,
        "n_oof_rows": int(len(meta)),
        "n_test_score_rows": int(len(test_scores)),
        "generated_csv_count": len(generated),
        "paths": {
            "primary": str(best_path),
            "search": str(OUTPUT_DIR / "fusion_search.csv"),
            "meta_validation_scores": str(OUTPUT_DIR / "meta_validation_scores.parquet"),
            "test_candidate_scores": str(OUTPUT_DIR / "test_candidate_scores.parquet"),
        },
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    print(f"\nPrimary output: {best_path}")
    print(f"Generated CSVs: {len(generated)}")


if __name__ == "__main__":
    main()
