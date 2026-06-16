"""V48: Linear fusion of existing model predictions

Strategy: heterogeneous models, rank-percentile normalization,
linear weighted blend.

Models available:
  1. V45 ensemble (LGB+XGB rank-fused)
  2. V45 XGB only
  3. V45 LGB only
  4. V27 heuristic

Fusion: score(uid, item) = Σ w_i × rank_pct(model_i, uid, item)
"""
import os, argparse
import numpy as np
import pandas as pd
from collections import defaultdict


def load_predictions():
    """Load all model predictions"""
    preds = {}
    base = r"D:\CODE\competition\AFAC\.Version"

    # V45 component models
    v45_path = os.path.join(base, "V45", "framework")
    for name, fname in [("v45_lgb", "A2_lgb.csv"), ("v45_xgb", "A2_xgb.csv"),
                         ("v45_ens", "A2_ensemble.csv")]:
        path = os.path.join(v45_path, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            preds[name] = dict(zip(df["uid"], df["prediction"].str.split(",")))

    # V27 heuristic
    for alt in [r"D:\CODE\competition\AFAC\.temp\A2_v27.csv",
                os.path.join(base, "V27", "framework", "A2.csv")]:
        if os.path.exists(alt):
            df = pd.read_csv(alt)
            preds["v27"] = dict(zip(df["uid"], df["prediction"].str.split(",")))
            break

    print(f"Loaded {len(preds)} models: {list(preds.keys())}")
    return preds


def rank_percentile_scores(preds):
    """Convert ranked lists to rank-percentile scores.
    For each user, items at rank k get score = 1 - (k-1)/(len_list-1)
    Items not in list get score = 0
    """
    scores = {}
    for model_name, uid_preds in preds.items():
        uid_scores = {}
        for uid, items in uid_preds.items():
            s = {}
            for rank, item in enumerate(items):
                s[item] = 1.0 - rank / max(len(items) - 1, 1)
            uid_scores[uid] = s
        scores[model_name] = uid_scores
    return scores


def blend(scores, weights, sample_sub, topk=10):
    """Blend models with given weights. Generate topk predictions."""
    predictions = {}
    all_uids = list(next(iter(scores.values())).keys())

    for uid in all_uids:
        blended = defaultdict(float)
        for model, w in weights.items():
            if w <= 0: continue
            uid_scores = scores[model].get(uid, {})
            for item, s in uid_scores.items():
                blended[item] += w * s

        ranked = sorted(blended.items(), key=lambda x: -x[1])
        top = [item for item, _ in ranked[:topk]]

        # Fill with global popularity if needed
        if len(top) < topk:
            all_items = list(blended.keys())
            for item in all_items:
                if item not in top:
                    top.append(item)
                if len(top) >= topk:
                    break
        predictions[uid] = top

    out = sample_sub.copy()
    out["prediction"] = out["uid"].map(
        lambda u: ",".join(predictions.get(u, [])[:topk]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--w_v45_ens", type=float, default=1.0)
    p.add_argument("--w_v45_xgb", type=float, default=0.0)
    p.add_argument("--w_v45_lgb", type=float, default=0.0)
    p.add_argument("--w_v27", type=float, default=0.0)
    p.add_argument("--save_to", type=str, default=None)
    args = p.parse_args()

    print("=== V48: Model Blending ===\n")
    preds = load_predictions()
    if len(preds) < 2:
        print("Need at least 2 models. Generate component A2s from V45 first.")
        return

    sample = pd.read_csv(r"D:\CODE\competition\AFAC\.Version\V45\framework\A2.csv")[["uid"]]
    scores = rank_percentile_scores(preds)

    weights = {
        "v45_ensemble": args.w_v45_ens,
        "v45_xgb": args.w_v45_xgb,
        "v45_lgb": args.w_v45_lgb,
        "v27": args.w_v27,
    }

    print(f"Weights: { {k:v for k,v in weights.items() if v>0} }")
    sub = blend(scores, weights, sample)
    save = args.save_to or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "A2.csv")
    sub.to_csv(save, index=False)
    print(f"Saved: {save}")
    print("Submit to check if blend > individual models")


if __name__ == "__main__":
    main()
