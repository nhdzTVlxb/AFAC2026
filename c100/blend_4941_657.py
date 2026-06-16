#!/usr/bin/env python3
"""Rank-percentile blend for the 0.4941 A2 and the 0.4945/657 A2.

This is a result-level probe: it does not retrain models. It keeps the two
submission files as ranked lists, converts each list into per-user rank-percentile
scores, then writes weighted fused A2 files.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd


def read_a2(path: str) -> dict[str, list[str]]:
    df = pd.read_csv(path)
    if "uid" not in df.columns or "prediction" not in df.columns:
        raise ValueError(f"{path} must contain uid,prediction columns")
    return {
        str(uid): [x for x in str(pred).split(",") if x]
        for uid, pred in zip(df["uid"], df["prediction"])
    }


def rank_scores(items: list[str]) -> dict[str, float]:
    denom = max(len(items) - 1, 1)
    return {item: 1.0 - i / denom for i, item in enumerate(items)}


def blend_one(
    a: dict[str, list[str]],
    b: dict[str, list[str]],
    wa: float,
    wb: float,
    keep_a_top: int,
    topk: int = 10,
) -> pd.DataFrame:
    rows = []
    for uid in sorted(set(a) | set(b)):
        a_items = a.get(uid, [])
        b_items = b.get(uid, [])
        score = defaultdict(float)
        for item, s in rank_scores(a_items).items():
            score[item] += wa * s
        for item, s in rank_scores(b_items).items():
            score[item] += wb * s
        ranked = [item for item, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))]
        if keep_a_top > 0:
            fixed = a_items[:keep_a_top]
            tail = [x for x in ranked if x not in fixed]
            ranked = fixed + tail
        rows.append({"uid": uid, "prediction": ",".join(ranked[:topk])})
    return pd.DataFrame(rows)


def overlap(a: dict[str, list[str]], b: dict[str, list[str]]) -> tuple[float, float]:
    n = 0
    total = 0
    top1_same = 0
    for uid in sorted(set(a) & set(b)):
        aa = a[uid][:10]
        bb = b[uid][:10]
        total += len(set(aa) & set(bb))
        top1_same += int(bool(aa and bb and aa[0] == bb[0]))
        n += 1
    return total / max(n, 1), top1_same / max(n, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--a2_4941", required=True, help="A2 from the 0.4941 version")
    p.add_argument("--a2_657", required=True, help="A2 from output657 / 0.4945")
    p.add_argument("--out_dir", default="blend_4941_657_outputs")
    p.add_argument(
        "--weights",
        default="0.7,0.3;0.6,0.4;0.5,0.5;0.4,0.6;0.3,0.7",
        help="semicolon-separated wa,wb pairs; wa is 4941 weight",
    )
    p.add_argument("--keep_4941_top", type=int, default=0)
    args = p.parse_args()

    a = read_a2(args.a2_4941)
    b = read_a2(args.a2_657)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_overlap, base_top1 = overlap(a, b)
    summary = []
    for spec in args.weights.split(";"):
        wa, wb = [float(x) for x in spec.split(",")]
        tag = f"w4941_{int(wa * 100):02d}_w657_{int(wb * 100):02d}"
        if args.keep_4941_top:
            tag += f"_keep4941top{args.keep_4941_top}"
        df = blend_one(a, b, wa, wb, args.keep_4941_top)
        path = out_dir / f"A2_{tag}.csv"
        df.to_csv(path, index=False)
        fused = read_a2(str(path))
        ov_a, top1_a = overlap(a, fused)
        ov_b, top1_b = overlap(b, fused)
        summary.append(
            {
                "file": path.name,
                "w4941": wa,
                "w657": wb,
                "keep_4941_top": args.keep_4941_top,
                "4941_657_overlap10": base_overlap,
                "4941_657_top1_same": base_top1,
                "fused_overlap10_vs_4941": ov_a,
                "fused_top1_same_vs_4941": top1_a,
                "fused_overlap10_vs_657": ov_b,
                "fused_top1_same_vs_657": top1_b,
            }
        )
        print(f"saved {path}")

    sm = pd.DataFrame(summary)
    sm.to_csv(out_dir / "summary.csv", index=False)
    print(sm.to_string(index=False))


if __name__ == "__main__":
    main()
