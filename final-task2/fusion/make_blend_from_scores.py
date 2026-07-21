"""Create a weighted LGB/deeper-NN submission from saved candidate scores."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path("/home/cyp/speedsci/AFAC2026")
BLEND_ROOT = ROOT / "task2_blend"
OUTPUT_DIR = BLEND_ROOT / "output"


def load_base():
    path = BLEND_ROOT / "LGB_NN_Ensemble_train.py"
    spec = importlib.util.spec_from_file_location("lgb_nn_ensemble_base", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load fusion base: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def weight_tag(weight: float) -> str:
    return f"{int(round(weight * 100)):02d}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--w-lgb", type=float, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.w_lgb <= 1.0:
        parser.error("--w-lgb must be between 0 and 1")

    w_lgb = args.w_lgb
    w_nn = 1.0 - w_lgb
    lgb_tag = weight_tag(w_lgb)
    nn_tag = weight_tag(w_nn)
    tag = f"LGB{lgb_tag}_NN{nn_tag}"

    base = load_base()
    meta = pd.read_parquet(OUTPUT_DIR / "meta_validation_scores.parquet")
    scores = pd.read_parquet(OUTPUT_DIR / "test_candidate_scores.parquet")
    sample = pd.read_csv(ROOT / "data" / "rec_data" / "sample_submission.csv")

    reports = []
    for method in ("rank", "zscore"):
        ndcg, hit = base.evaluate(meta, method, w_lgb)
        reports.append(
            {
                "method": method,
                "w_lgb": w_lgb,
                "w_nn": w_nn,
                "ndcg10_original": ndcg,
                "hit10_original": hit,
            }
        )
        base.make_submission(scores, sample, method, w_lgb).to_csv(
            OUTPUT_DIR / f"A2_blend_{tag}_{method}.csv", index=False
        )

    report = pd.DataFrame(reports).sort_values(
        ["ndcg10_original", "hit10_original"], ascending=[False, False]
    )
    report.to_csv(OUTPUT_DIR / f"fusion_{lgb_tag}_{nn_tag}_report.csv", index=False)
    best = report.iloc[0].to_dict()
    primary_path = OUTPUT_DIR / f"A2_blend_{tag}.csv"
    base.make_submission(scores, sample, str(best["method"]), w_lgb).to_csv(
        primary_path, index=False
    )
    (OUTPUT_DIR / f"manifest_{lgb_tag}_{nn_tag}.json").write_text(
        json.dumps(
            {"w_lgb": w_lgb, "w_nn": w_nn, "best": best, "primary": str(primary_path)},
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(report.to_string(index=False))
    print(f"Primary output: {primary_path}")


if __name__ == "__main__":
    main()
