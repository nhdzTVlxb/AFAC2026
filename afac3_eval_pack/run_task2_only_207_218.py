#!/usr/bin/env python3
import argparse
import json
import runpy
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from run_combo_195_206 import (
    build_stats, ensure_afac_a2, read_a2, write_a2,
    cap_history, user_global_boost, rank_fusion, fill_unique,
)


def parse_seq(s):
    if pd.isna(s) or str(s).strip() in ("", "nan"):
        return []
    return [x for x in str(s).split(",") if x]


def run_script_a2(root, script_name):
    print(f"Running {script_name}...")
    runpy.run_path(str(Path(root) / script_name), run_name="__main__")
    path = Path(root) / "A2.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return read_a2(path)


def novel_only_rerank(base_preds, stats, max_hist=1):
    return cap_history(base_preds, stats, max_hist=max_hist, prefer_group=True)


def balanced_cap(base_preds, stats):
    out = {}
    for uid, items in base_preds.items():
        hist = set()
        if uid in stats["test_lookup"].index:
            hist = set(parse_seq(stats["test_lookup"].loc[uid, "item_seq_raw"]))
        hist_items = [x for x in items if x in hist]
        novel_items = [x for x in items if x not in hist]
        mixed = []
        if novel_items:
            mixed.extend(novel_items[:5])
        if hist_items:
            mixed.extend(hist_items[:3])
        mixed.extend(novel_items[5:])
        mixed.extend(hist_items[3:])
        mixed.extend(stats["global_rank"])
        out[uid] = fill_unique(mixed, uid, stats)
    return out


def cold_fallback_stronger(base_preds, stats):
    out = dict(base_preds)
    user_lookup = stats["user_lookup"]
    for uid in list(out.keys()):
        hist_len = 0
        if uid in stats["test_lookup"].index:
            hist_len = len(parse_seq(stats["test_lookup"].loc[uid, "item_seq_raw"]))
        if hist_len > 2:
            continue
        group_items = []
        if uid in user_lookup.index and stats["group_cols"]:
            key = tuple(user_lookup.loc[uid, stats["group_cols"]].tolist())
            group_items = stats["group_rank"].get(key, [])
        out[uid] = fill_unique(group_items + stats["global_rank"] + out.get(uid, []), uid, stats)
    return out


def save_output(out_root, num, name, a1_path, preds, stats):
    out_dir = Path(out_root) / f"output{num}"
    sub_dir = out_dir / "submission"
    sub_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(a1_path, sub_dir / "A1.csv")
    write_a2(preds, stats["sample"], sub_dir / "A2.csv")
    with zipfile.ZipFile(out_dir / "prediction.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(sub_dir / "A1.csv", "A1.csv")
        z.write(sub_dir / "A2.csv", "A2.csv")
    metrics = {
        "output": f"output{num}",
        "experiment": name,
        "zip": True,
        "a1_source": str(a1_path),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def specs():
    return [
        (207, "AFAC A2 original", "afac"),
        (208, "AFAC A2 history cap3", "cap3"),
        (209, "AFAC A2 history cap2", "cap2"),
        (210, "AFAC A2 user_group/global boost", "user_global"),
        (211, "AFAC A2 + output122 GRU rank fusion weak", "gru_weak"),
        (212, "AFAC A2 + output122 GRU rank fusion strong", "gru_strong"),
        (213, "AFAC train_task_b_v6.py", "v6"),
        (214, "AFAC train_task_b_ndcg.py", "ndcg"),
        (215, "AFAC train_task_b_v5.py", "v5"),
        (216, "AFAC A2 novel-only cap1", "novel_only"),
        (217, "AFAC A2 repeat/novel balanced cap", "balanced"),
        (218, "AFAC A2 cold-start fallback stronger", "cold"),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out_root", default="../v1/framework")
    ap.add_argument("--a1_path", default="../v1/framework/output203/submission/A1.csv")
    ap.add_argument("--output122_a2", default="../v1/framework/output122/submission/A2.csv")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_root = Path(args.out_root).resolve()
    a1_path = Path(args.a1_path).resolve()
    if not a1_path.exists():
        raise FileNotFoundError(f"missing fixed A1: {a1_path}")

    stats = build_stats(root)
    base_path = ensure_afac_a2(root)
    base = read_a2(base_path)
    rows = []
    cache = {"afac": base}

    for num, name, kind in specs():
        if kind == "afac":
            preds = base
        elif kind == "cap3":
            preds = cap_history(base, stats, max_hist=3, prefer_group=True)
        elif kind == "cap2":
            preds = cap_history(base, stats, max_hist=2, prefer_group=False)
        elif kind == "user_global":
            preds = user_global_boost(base, stats)
        elif kind == "gru_weak":
            if Path(args.output122_a2).exists():
                preds = rank_fusion(base, read_a2(args.output122_a2), stats, w_a=1.0, w_b=0.55)
            else:
                print(f"WARNING missing {args.output122_a2}, fallback to AFAC")
                preds = base
        elif kind == "gru_strong":
            if Path(args.output122_a2).exists():
                preds = rank_fusion(base, read_a2(args.output122_a2), stats, w_a=1.0, w_b=0.85)
            else:
                print(f"WARNING missing {args.output122_a2}, fallback to AFAC")
                preds = base
        elif kind == "v6":
            preds = run_script_a2(root, "train_task_b_v6.py")
        elif kind == "ndcg":
            preds = run_script_a2(root, "train_task_b_ndcg.py")
        elif kind == "v5":
            preds = run_script_a2(root, "train_task_b_v5.py")
        elif kind == "novel_only":
            preds = novel_only_rerank(base, stats, max_hist=1)
        elif kind == "balanced":
            preds = balanced_cap(base, stats)
        elif kind == "cold":
            preds = cold_fallback_stronger(base, stats)
        else:
            raise ValueError(kind)
        rows.append(save_output(out_root, num, name, a1_path, preds, stats))

    summary = out_root / "ablation_207_218_summary.md"
    lines = [
        "# Task2-only Ablation 207-218 Summary",
        "",
        f"Fixed A1: {a1_path}",
        "",
        "| output | experiment | zip |",
        "|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['output']} | {r['experiment']} | {'yes' if r['zip'] else 'no'} |")
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.read_text())


if __name__ == "__main__":
    main()
