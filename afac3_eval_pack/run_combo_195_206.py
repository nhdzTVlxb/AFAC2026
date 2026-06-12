#!/usr/bin/env python3
import argparse
import json
import os
import runpy
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from run_afac_task1_181_194 import load_data, run_one


def base_task1_config():
    return {
        "model_type": "gcn",
        "hidden_dim": 256,
        "num_layers": 3,
        "dropout": 0.5,
        "lr": 0.01,
        "weight_decay": 5e-4,
        "epochs": 300,
        "patience": 50,
        "feature_norm": "standard",
        "symmetrize": True,
        "norm_mode": "symmetric",
        "val_ratio": 0.05,
        "seed": 42,
    }


def make_task1_config(kind):
    cfg = base_task1_config()
    if kind == "original":
        return cfg
    if kind == "wd1e3":
        cfg["weight_decay"] = 1e-3
    elif kind == "binary":
        cfg["feature_norm"] = "binary"
    elif kind == "binary_wd1e3":
        cfg["feature_norm"] = "binary"
        cfg["weight_decay"] = 1e-3
    elif kind == "val007":
        cfg["val_ratio"] = 0.07
    elif kind == "val008":
        cfg["val_ratio"] = 0.08
    elif kind == "val010":
        cfg["val_ratio"] = 0.10
    elif kind == "gcnii_binary":
        cfg.update({
            "model_type": "gcnii",
            "alpha": 0.15,
            "num_layers": 8,
            "dropout": 0.3,
            "weight_decay": 1e-3,
            "feature_norm": "binary",
        })
    else:
        raise ValueError(f"unknown task1 kind: {kind}")
    return cfg


def ensure_afac_a2(root):
    path = Path(root) / "A2.csv"
    if path.exists():
        return path
    print("AFAC base A2 not found, running train_task_b_final.py once...")
    runpy.run_path(str(Path(root) / "train_task_b_final.py"), run_name="__main__")
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def parse_seq(s):
    if pd.isna(s) or str(s).strip() in ("", "nan"):
        return []
    return [x for x in str(s).split(",") if x]


def load_rec(root):
    base = Path(root) / "A推荐" / "A推荐"
    train = pd.read_csv(base / "train.csv")
    test = pd.read_csv(base / "test.csv")
    user = pd.read_csv(base / "user.csv")
    sample = pd.read_csv(base / "sample_submission.csv")
    return train, test, user, sample


def build_stats(root):
    train, test, user, sample = load_rec(root)
    target_counts = Counter(train["target_iid"].astype(str))
    raw_counts = Counter()
    for seq in train["item_seq_raw"].dropna():
        raw_counts.update(parse_seq(seq))
    global_rank = [iid for iid, _ in (target_counts + raw_counts).most_common()]

    user_cols = [c for c in user.columns if c != "uid"]
    group_cols = user_cols[:4]
    merged = train[["uid", "target_iid"]].merge(user, on="uid", how="left")
    group_rank = {}
    for key, g in merged.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        group_rank[key] = [iid for iid, _ in Counter(g["target_iid"].astype(str)).most_common()]
    user_lookup = user.set_index("uid") if "uid" in user.columns else pd.DataFrame()
    test_lookup = test.set_index("uid")
    return {
        "train": train,
        "test": test,
        "sample": sample,
        "global_rank": global_rank,
        "group_rank": group_rank,
        "group_cols": group_cols,
        "user_lookup": user_lookup,
        "test_lookup": test_lookup,
    }


def read_a2(path):
    df = pd.read_csv(path)
    return {row["uid"]: parse_seq(row["prediction"]) for _, row in df.iterrows()}


def write_a2(preds, sample, out_path):
    rows = []
    for _, row in sample.iterrows():
        uid = row["uid"]
        pred = preds.get(uid, [])
        rows.append({"uid": uid, "prediction": ",".join(pred[:10])})
    pd.DataFrame(rows).to_csv(out_path, index=False)


def fill_unique(items, uid, stats, max_len=10, avoid_history=False):
    test_lookup = stats["test_lookup"]
    hist = set()
    if uid in test_lookup.index:
        hist = set(parse_seq(test_lookup.loc[uid, "item_seq_raw"]))
    result = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        if avoid_history and item in hist:
            continue
        result.append(item)
        seen.add(item)
        if len(result) >= max_len:
            return result
    for item in stats["global_rank"]:
        if item in seen:
            continue
        if avoid_history and item in hist:
            continue
        result.append(item)
        seen.add(item)
        if len(result) >= max_len:
            break
    return result


def cap_history(preds, stats, max_hist=3, prefer_group=False):
    out = {}
    user_lookup = stats["user_lookup"]
    for uid, items in preds.items():
        hist = set()
        if uid in stats["test_lookup"].index:
            hist = set(parse_seq(stats["test_lookup"].loc[uid, "item_seq_raw"]))
        kept = []
        hist_count = 0
        for item in items:
            if item in hist:
                if hist_count >= max_hist:
                    continue
                hist_count += 1
            kept.append(item)

        fillers = []
        if prefer_group and uid in user_lookup.index and stats["group_cols"]:
            key = tuple(user_lookup.loc[uid, stats["group_cols"]].tolist())
            fillers.extend(stats["group_rank"].get(key, []))
        fillers.extend(stats["global_rank"])
        out[uid] = fill_unique(kept + fillers, uid, stats, avoid_history=False)
    return out


def user_global_boost(preds, stats):
    out = {}
    user_lookup = stats["user_lookup"]
    for uid, items in preds.items():
        boost = []
        if uid in user_lookup.index and stats["group_cols"]:
            key = tuple(user_lookup.loc[uid, stats["group_cols"]].tolist())
            boost.extend(stats["group_rank"].get(key, [])[:5])
        boost.extend(stats["global_rank"][:10])
        mixed = []
        if items:
            mixed.extend(items[:5])
        mixed.extend(boost)
        mixed.extend(items[5:])
        out[uid] = fill_unique(mixed, uid, stats)
    return out


def rank_fusion(pred_a, pred_b, stats, w_a=1.0, w_b=0.65):
    uids = set(pred_a) | set(pred_b)
    out = {}
    for uid in uids:
        scores = defaultdict(float)
        for rank, item in enumerate(pred_a.get(uid, []), start=1):
            scores[item] += w_a / rank
        for rank, item in enumerate(pred_b.get(uid, []), start=1):
            scores[item] += w_b / rank
        items = [item for item, _ in sorted(scores.items(), key=lambda x: -x[1])]
        out[uid] = fill_unique(items, uid, stats)
    return out


def build_task2(root, kind, out_path, stats, output122_a2):
    base_a2 = ensure_afac_a2(root)
    base = read_a2(base_a2)
    if kind == "afac":
        preds = base
    elif kind == "novel_cap3":
        preds = cap_history(base, stats, max_hist=3, prefer_group=True)
    elif kind == "safe_cap2":
        preds = cap_history(base, stats, max_hist=2, prefer_group=False)
    elif kind == "user_global":
        preds = user_global_boost(base, stats)
    elif kind == "gru_fusion":
        if output122_a2 and Path(output122_a2).exists():
            preds = rank_fusion(base, read_a2(output122_a2), stats, w_a=1.0, w_b=0.55)
        else:
            print(f"WARNING: missing output122 A2: {output122_a2}, fallback to AFAC A2")
            preds = base
    elif kind == "gru_fusion_strong":
        if output122_a2 and Path(output122_a2).exists():
            preds = rank_fusion(base, read_a2(output122_a2), stats, w_a=1.0, w_b=0.85)
        else:
            print(f"WARNING: missing output122 A2: {output122_a2}, fallback to AFAC A2")
            preds = base
    elif kind in ("ndcg", "v6"):
        script = "train_task_b_ndcg.py" if kind == "ndcg" else "train_task_b_v6.py"
        print(f"Running {script} for Task2 variant...")
        runpy.run_path(str(Path(root) / script), run_name="__main__")
        preds = read_a2(Path(root) / "A2.csv")
    else:
        raise ValueError(f"unknown task2 kind: {kind}")
    write_a2(preds, stats["sample"], out_path)


def combo_specs():
    return [
        (195, "output190 original + AFAC A2 original", "original", "afac"),
        (196, "output190 wd1e-3 + AFAC A2 novel cap3", "wd1e3", "novel_cap3"),
        (197, "output190 binary + AFAC A2 safe cap2", "binary", "safe_cap2"),
        (198, "output190 binary wd1e-3 + AFAC A2 user/global", "binary_wd1e3", "user_global"),
        (199, "output190 val_ratio007 + AFAC/GRU fusion", "val007", "gru_fusion"),
        (200, "output190 val_ratio008 + AFAC/GRU fusion strong", "val008", "gru_fusion_strong"),
        (201, "output190 val_ratio010 + AFAC task_b_ndcg", "val010", "ndcg"),
        (202, "output190 original + AFAC task_b_v6", "original", "v6"),
        (203, "GCNII binary LP + AFAC A2 original", "gcnii_binary", "afac"),
        (204, "GCNII binary LP + AFAC A2 novel cap3", "gcnii_binary", "novel_cap3"),
        (205, "output190 original + AFAC/GRU rank fusion", "original", "gru_fusion"),
        (206, "output190 wd1e-3 + AFAC/GRU rank fusion", "wd1e3", "gru_fusion"),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out_root", default="../v1/framework")
    ap.add_argument("--output122_a2", default="../v1/framework/output122/submission/A2.csv")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    out_root = Path(args.out_root).resolve()
    data = load_data(root)
    stats = build_stats(root)
    rows = []
    for num, name, t1_kind, t2_kind in combo_specs():
        cfg = make_task1_config(t1_kind)
        metrics = run_one(root, out_root, num, name, cfg, data, None)
        out_dir = out_root / f"output{num}"
        sub_dir = out_dir / "submission"
        build_task2(root, t2_kind, sub_dir / "A2.csv", stats, args.output122_a2)
        with zipfile.ZipFile(out_dir / "prediction.zip", "w", zipfile.ZIP_DEFLATED) as z:
            z.write(sub_dir / "A1.csv", "A1.csv")
            z.write(sub_dir / "A2.csv", "A2.csv")
        metrics["task2_variant"] = t2_kind
        metrics["zip"] = True
        (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append(metrics)

    summary = out_root / "ablation_195_206_summary.md"
    lines = [
        "# Combo Ablation 195-206 Summary",
        "",
        "| output | experiment | task1_base_acc | task1_fusion_acc | task1_strategy | task2_variant | zip |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['output']} | {r['experiment']} | {r['base_model_acc']:.6f} | "
            f"{r['fusion_acc']:.6f} | {r['strategy']} | {r['task2_variant']} | {'yes' if r['zip'] else 'no'} |"
        )
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.read_text())


if __name__ == "__main__":
    main()
