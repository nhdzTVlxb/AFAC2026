#!/usr/bin/env python3
"""Generate a 0.4941 training script with output657 source features added.

The 0.4941 version already has multi-recall + LGB/XGB. The best 0.4945 result
came from exposing source-hit/source-strength features to LightGBM. This patcher
adds that feature block to train_lgb_xgb.py and also makes DATA_ROOT configurable
for the server.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def patch_data_root(code: str) -> str:
    old = '''def find_data_path() -> str:
    code_dir = os.path.dirname(os.path.abspath(__file__))  # V45/framework/code
    afac_dir = os.path.dirname(os.path.dirname(os.path.dirname(code_dir)))  # AFAC/
    candidates = [
        os.path.join(afac_dir, "V0", "framework", "data", "rec_data"),
        r"D:\\CODE\\competition\\AFAC\\.temp",
    ]
    for p in candidates:
        if os.path.exists(os.path.join(p, "train.csv")):
            return p
    raise FileNotFoundError("Cannot find rec_data")
'''
    new = '''def find_data_path() -> str:
    env_path = os.environ.get("DATA_ROOT")
    candidates = []
    if env_path:
        candidates.append(env_path)
    code_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.extend([
        os.path.join(code_dir, "A推荐", "A推荐"),
        os.path.join(os.path.dirname(code_dir), "A推荐", "A推荐"),
        os.path.join(os.path.dirname(code_dir), "v1", "framework", "data", "rec_data"),
        r"D:\\CODE\\competition\\AFAC\\.temp",
    ])
    for p in candidates:
        if p and os.path.exists(os.path.join(p, "train.csv")):
            return p
    raise FileNotFoundError("Cannot find rec_data; set DATA_ROOT=/path/to/A推荐/A推荐")
'''
    if old not in code:
        raise ValueError("find_data_path anchor not found")
    return code.replace(old, new, 1)


def patch_runtime_knobs(code: str) -> str:
    replacements = {
        "N_FOLDS = 3\n": "N_FOLDS = int(os.environ.get('N_FOLDS', '3'))\n",
        "MAX_CANDIDATES = 100\n": "MAX_CANDIDATES = int(os.environ.get('MAX_CANDIDATES', '100'))\n",
        "N_JOBS = max(1, min(8, os.cpu_count() or 2))\n": (
            "N_JOBS = int(os.environ.get('N_JOBS', str(max(1, min(8, os.cpu_count() or 2)))))\n"
        ),
        "    batch = 300\n": "    batch = int(os.environ.get('PRED_BATCH_SIZE', '300'))\n",
    }
    for old, new in replacements.items():
        if old in code:
            code = code.replace(old, new, 1)
    return code


def patch_gpu_training(code: str) -> str:
    code = code.replace(
        "        reg_alpha=0.15, reg_lambda=1.5, max_bin=127,\n"
        "        random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=-1)",
        "        reg_alpha=0.15, reg_lambda=1.5, max_bin=127,\n"
        "        random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=-1,\n"
        "        device=os.environ.get('LGB_DEVICE', 'gpu'),\n"
        "        gpu_use_dp=False)",
    )
    code = code.replace(
        "        tree_method=\"hist\", max_bin=256, lambdarank_pair_method=\"topk\",\n",
        "        tree_method=os.environ.get('XGB_TREE_METHOD', 'hist'),\n"
        "        device=os.environ.get('XGB_DEVICE', 'cuda'), max_bin=256, lambdarank_pair_method=\"topk\",\n",
    )
    return code


def patch_source_features(code: str) -> str:
    old = '''        # Interactions
        f["repeat_x_recent"] = f["history_present"] * f["history_recency"]
        f["last1_x_history"] = f["last1_prob"] * (1.0 + f["history_logcount"])
        f["suffix_best_prob"] = max(f["last1_prob"], f["last2_prob"], f["last3_prob"])
        f["suffix_best_lift"] = max(f["last1_lift"], f["last2_lift"], f["last3_lift"])

        return f
'''
    new = '''        # Interactions
        f["repeat_x_recent"] = f["history_present"] * f["history_recency"]
        f["last1_x_history"] = f["last1_prob"] * (1.0 + f["history_logcount"])
        f["suffix_best_prob"] = max(f["last1_prob"], f["last2_prob"], f["last3_prob"])
        f["suffix_best_lift"] = max(f["last1_lift"], f["last2_lift"], f["last3_lift"])

        # output657 source-rank features: these were the stable online gain.
        f.update({
            "src_last1_hit": float(f["last1_logcnt"] > 0),
            "src_last2_hit": float(f["last2_logcnt"] > 0),
            "src_last3_hit": float(f["last3_logcnt"] > 0),
            "src_recent_hit": float(f["recent_hit_sources"] > 0),
            "src_transition_hit": float(f["trans_logcnt"] > 0),
            "src_profile_hit": float(f["prof_logcnt"] > 0),
            "src_length_hit": float(f["len_logcnt"] > 0),
            "src_segment_hit_sum": float(sum(1.0 for x in scs if x > 0)),
            "src_pair_hit_sum": float(sum(1.0 for x in pcs if x > 0)),
            "src_supervised_hit_sum": float(f["last1_logcnt"] > 0)
                + float(f["last2_logcnt"] > 0)
                + float(f["last3_logcnt"] > 0)
                + float(f["recent_hit_sources"] > 0)
                + float(f["trans_logcnt"] > 0),
            "src_profile_pair_strength": f["prof_prob"] + f["pair_prob_max"],
            "src_suffix_recent_strength": f["suffix_best_prob"] + f["recent_prob_max"],
            "src_rrf_x_best_rank": f["rrf"] * f["best_rank_recip"],
        })

        return f
'''
    if old not in code:
        raise ValueError("pair feature return anchor not found")
    return code.replace(old, new, 1)


def patch_weights(code: str, lgb_weight: float, xgb_weight: float) -> str:
    old = 'ms = add_rank_fusion(ms, {"lgb": "lgb_score", "xgb": "xgb_score"})'
    new = (
        'ms = add_rank_fusion(ms, {"lgb": "lgb_score", "xgb": "xgb_score"}, '
        f'weights={{"lgb": {lgb_weight}, "xgb": {xgb_weight}}})'
    )
    code = code.replace(old, new)
    old2 = 'cf = add_rank_fusion(cf, {"lgb": "lgb_score", "xgb": "xgb_score"})'
    new2 = (
        'cf = add_rank_fusion(cf, {"lgb": "lgb_score", "xgb": "xgb_score"}, '
        f'weights={{"lgb": {lgb_weight}, "xgb": {xgb_weight}}})'
    )
    return code.replace(old2, new2)


def patch_lgb_only_final(code: str) -> str:
    code = code.replace(
        "    # XGBoost\n"
        "    print(\"\\nTraining XGBoost rank:ndcg...\")\n"
        "    qtrain = mt[\"qid\"].to_numpy(dtype=np.int64); qve = mve[\"qid\"].to_numpy(dtype=np.int64)\n"
        "    xgbm = xgb.XGBRanker(objective=\"rank:ndcg\", eval_metric=\"ndcg@10\",\n",
        "    # XGBoost\n"
        "    if os.environ.get(\"LGB_ONLY_PIPELINE\", \"0\") == \"1\":\n"
        "        print(\"\\nLGB_ONLY_PIPELINE=1: skip XGBoost meta training to save memory.\")\n"
        "        xgbm = None\n"
        "        xb = 0\n"
        "    else:\n"
        "        print(\"\\nTraining XGBoost rank:ndcg...\")\n"
        "        qtrain = mt[\"qid\"].to_numpy(dtype=np.int64); qve = mve[\"qid\"].to_numpy(dtype=np.int64)\n"
        "        xgbm = xgb.XGBRanker(objective=\"rank:ndcg\", eval_metric=\"ndcg@10\",\n",
        1,
    )
    code = code.replace(
        "        n_estimators=2200, learning_rate=0.025, max_depth=7, min_child_weight=10.0,\n"
        "        subsample=0.85, colsample_bytree=0.85, reg_alpha=0.10, reg_lambda=2.0,\n"
        "        tree_method=os.environ.get('XGB_TREE_METHOD', 'hist'),\n"
        "        device=os.environ.get('XGB_DEVICE', 'cuda'), max_bin=256, lambdarank_pair_method=\"topk\",\n"
        "        lambdarank_num_pair_per_sample=16, early_stopping_rounds=120,\n"
        "        random_state=RANDOM_STATE+1, n_jobs=N_JOBS)\n"
        "    xgbm.fit(Xt, yt, qid=qtrain,\n"
        "             eval_set=[(Xve, yve)], eval_qid=[qve], verbose=100)\n"
        "    xb = max(int(getattr(xgbm, \"best_iteration\", 599)) + 1, 100)\n",
        "            n_estimators=2200, learning_rate=0.025, max_depth=7, min_child_weight=10.0,\n"
        "            subsample=0.85, colsample_bytree=0.85, reg_alpha=0.10, reg_lambda=2.0,\n"
        "            tree_method=os.environ.get('XGB_TREE_METHOD', 'hist'),\n"
        "            device=os.environ.get('XGB_DEVICE', 'cuda'), max_bin=256, lambdarank_pair_method=\"topk\",\n"
        "            lambdarank_num_pair_per_sample=16, early_stopping_rounds=120,\n"
        "            random_state=RANDOM_STATE+1, n_jobs=N_JOBS)\n"
        "        xgbm.fit(Xt, yt, qid=qtrain,\n"
        "                 eval_set=[(Xve, yve)], eval_qid=[qve], verbose=100)\n"
        "        xb = max(int(getattr(xgbm, \"best_iteration\", 599)) + 1, 100)\n",
        1,
    )
    code = code.replace(
        "    ms[\"xgb_score\"] = xgbm.predict(Xv).astype(np.float32)\n",
        "    if xgbm is None:\n"
        "        ms[\"xgb_score\"] = ms[\"lgb_score\"].astype(np.float32)\n"
        "    else:\n"
        "        ms[\"xgb_score\"] = xgbm.predict(Xv).astype(np.float32)\n",
        1,
    )
    code = code.replace(
        "    del lgbm, xgbm, Xt, Xv, Xve, yt, yve, qtrain, qve; gc.collect()\n",
        "    for _name in ['lgbm', 'xgbm', 'Xt', 'Xv', 'Xve', 'yt', 'yve', 'qtrain', 'qve', 'mt', 'mv', 'mve', 'ms']:\n"
        "        if _name in locals():\n"
        "            del locals()[_name]\n"
        "    gc.collect()\n",
        1,
    )
    code = code.replace(
        "    print(f\"  Selected: LGB={lb}, XGB={xb}\")\n\n"
        "    # Refit on all OOF\n",
        "    print(f\"  Selected: LGB={lb}, XGB={xb}\")\n\n"
        "    if os.environ.get(\"SKIP_FINAL_REFIT\", \"0\") == \"1\":\n"
        "        print(\"SKIP_FINAL_REFIT=1: use meta-trained LightGBM directly for test prediction.\")\n"
        "        return lgbm, xgbm, cols, {\"lgb\": lb, \"xgb\": xb}\n\n"
        "    # Refit on all OOF\n",
        1,
    )
    old = '''    fxgb = xgb.XGBRanker(objective="rank:ndcg", eval_metric="ndcg@10",
        n_estimators=xb, learning_rate=0.025, max_depth=7, min_child_weight=10.0,
        subsample=0.85, colsample_bytree=0.85, reg_alpha=0.10, reg_lambda=2.0,
        tree_method=os.environ.get('XGB_TREE_METHOD', 'hist'),
        device=os.environ.get('XGB_DEVICE', 'cuda'), max_bin=256, lambdarank_pair_method="topk",
        lambdarank_num_pair_per_sample=16,
        random_state=RANDOM_STATE+1, n_jobs=N_JOBS)
    fxgb.fit(Xa, ya, qid=oof_s["qid"].to_numpy(dtype=np.int64), verbose=False)

    return flgb, fxgb, cols, {"lgb": lb, "xgb": xb}
'''
    new = '''    if os.environ.get("LGB_ONLY_FINAL", "0") == "1":
        print("LGB_ONLY_FINAL=1: skip final XGBoost refit to save memory.")
        del Xa, ya, oof_s
        gc.collect()
        return flgb, None, cols, {"lgb": lb, "xgb": 0}

    fxgb = xgb.XGBRanker(objective="rank:ndcg", eval_metric="ndcg@10",
        n_estimators=xb, learning_rate=0.025, max_depth=7, min_child_weight=10.0,
        subsample=0.85, colsample_bytree=0.85, reg_alpha=0.10, reg_lambda=2.0,
        tree_method=os.environ.get('XGB_TREE_METHOD', 'hist'),
        device=os.environ.get('XGB_DEVICE', 'cuda'), max_bin=256, lambdarank_pair_method="topk",
        lambdarank_num_pair_per_sample=16,
        random_state=RANDOM_STATE+1, n_jobs=N_JOBS)
    fxgb.fit(Xa, ya, qid=oof_s["qid"].to_numpy(dtype=np.int64), verbose=False)

    return flgb, fxgb, cols, {"lgb": lb, "xgb": xb}
'''
    if old not in code:
        raise ValueError("final xgb refit anchor not found")
    code = code.replace(old, new, 1)
    code = code.replace(
        "        cf[\"xgb_score\"] = fxgb.predict(X).astype(np.float32)\n",
        "        if fxgb is None:\n"
        "            cf[\"xgb_score\"] = cf[\"lgb_score\"].astype(np.float32)\n"
        "        else:\n"
        "            cf[\"xgb_score\"] = fxgb.predict(X).astype(np.float32)\n",
        1,
    )
    return code


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="train_lgb_xgb.py")
    p.add_argument("--out", default="train_lgb_xgb_source657.py")
    p.add_argument("--lgb_weight", type=float, default=1.0)
    p.add_argument("--xgb_weight", type=float, default=2.0)
    args = p.parse_args()

    src = Path(args.src)
    code = src.read_text(encoding="utf-8")
    code = patch_data_root(code)
    code = patch_runtime_knobs(code)
    code = patch_gpu_training(code)
    code = patch_source_features(code)
    code = patch_weights(code, args.lgb_weight, args.xgb_weight)
    code = patch_lgb_only_final(code)
    Path(args.out).write_text(code, encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
