"""V19: V17 + 多截断模式训练增强 (队友SASRec: testmix+last3+empty)

V17 baseline: testmix单截断, LGB单模
V19 改动: 每个用户生成4份特征行(原始+3种截断)
  - 原始完整序列
  - testmix截断 (匹配测试分布)
  - last3截断   (随机最后1~3个物品)
  - empty截断   (强制冷启动, 空序列)
  3种截断覆盖全长度谱: 冷启动→极短→短→完整
"""
from __future__ import annotations
import gc, math, os, sys, random, argparse, json
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb

# ============================================================
# Config
# ============================================================
N_FOLDS = int(os.environ.get('N_FOLDS', '3'))
RANDOM_STATE = 2025
MAX_CANDIDATES = int(os.environ.get('MAX_CANDIDATES', '55'))
RECENT_ITEMS = 6
MAX_PAIR_FEATURE_COLS = 6
N_JOBS = int(os.environ.get('N_JOBS', str(max(1, min(8, os.cpu_count() or 2)))))

TOP_GLOBAL = 80
TOP_LAST1 = 60
TOP_LAST2 = 50
TOP_LAST3 = 40
TOP_RECENT_EACH = 24
TOP_TRANSITION = 35
TOP_PROFILE = 40
TOP_SEGMENT = 25
TOP_SEGMENT_PAIR = 25
TOP_LENGTH_BUCKET = 25
TOP_GROUP_FINE = 30
TOP_GROUP_MID = 20
TOP_GROUP_COARSE = 15

# ============================================================
# V16: testmix 序列截断 (匹配测试集长度分布)
# ============================================================
def testmix_truncate(seq_str: str, rng: np.random.Generator) -> str:
    """按测试集长度分布随机截断序列"""
    return _truncate_seq(seq_str, rng, mode="testmix")

def last3_truncate(seq_str: str, rng: np.random.Generator) -> str:
    """随机取最后1~3个物品 — 模拟极短历史用户"""
    return _truncate_seq(seq_str, rng, mode="last3")

def empty_truncate(seq_str: str, rng: np.random.Generator) -> str:
    """强制空序列 — 模拟纯冷启动用户"""
    return ""

def _truncate_seq(seq_str: str, rng: np.random.Generator, mode: str) -> str:
    """按指定模式截断序列"""
    items = [x.strip() for x in str(seq_str).split(",") if x.strip()] if seq_str else []
    n = len(items)
    if n == 0:
        return ""
    if mode == "empty":
        return ""
    elif mode == "last3":
        k = min(int(rng.integers(1, 4)), n)
    elif mode == "testmix":
        r = rng.random()
        if r < 0.3515:        k = 0
        elif r < 0.4518:      k = 1
        elif r < 0.8992:      k = min(int(rng.integers(2, 4)), n)
        elif r < 0.9006:      k = min(int(rng.integers(4, 6)), n)
        elif r < 0.9053:      k = min(int(rng.integers(6, 11)), n)
        elif r < 0.9202:      k = min(int(rng.integers(11, 31)), n)
        else:
            if n <= 31: k = n
            else:       k = int(rng.integers(31, min(n, 200) + 1))
    else:
        return seq_str
    items = items[-min(k, n):] if k > 0 else []
    return ",".join(items) if items else ""

def build_truncated_row(row, rng, mode="testmix"):
    """构建截断版本的 row 副本(只修改序列相关列)"""
    val_row = row.copy()
    original_raw = str(row.get("item_seq_raw", ""))
    truncated_raw = _truncate_seq(original_raw, rng, mode)
    val_row["item_seq_raw"] = truncated_raw
    raw_items = [x.strip() for x in truncated_raw.split(",") if x.strip()] if truncated_raw else []
    dedup_items = []
    seen = set()
    for it in raw_items:
        if it not in seen:
            seen.add(it)
            dedup_items.append(it)
    val_row["item_seq_dedup"] = ",".join(dedup_items) if dedup_items else ""
    return val_row

# ============================================================
# Data helpers
# ============================================================
def find_data_path() -> str:
    env_path = os.environ.get("DATA_ROOT")
    if env_path and os.path.exists(os.path.join(env_path, "train.csv")):
        return env_path
    code_dir = os.path.dirname(os.path.abspath(__file__))
    # 优先读取同目录下的 data/ 文件夹
    candidates = [
        os.path.join(code_dir, "data"),
        os.path.join(code_dir, "..", "c55_lgb", "data"),
    ]
    for p in candidates:
        if p and os.path.exists(os.path.join(p, "train.csv")):
            return p
    raise FileNotFoundError(f"{code_dir}/data/ 下未找到数据文件")

def load_data():
    base = find_data_path()
    print(f"Data: {base}")
    return (pd.read_csv(os.path.join(base, "train.csv")),
            pd.read_csv(os.path.join(base, "test.csv")),
            pd.read_csv(os.path.join(base, "user.csv")),
            pd.read_csv(os.path.join(base, "sample_submission.csv")))

def clean_id(x):
    if pd.isna(x): return ""
    if isinstance(x, (np.integer, int)): return str(int(x))
    if isinstance(x, (np.floating, float)) and float(x).is_integer(): return str(int(x))
    s = str(x).strip()
    return "" if s.lower() == "nan" else s

def clean_scalar(x):
    if pd.isna(x): return "__NA__"
    s = str(x).strip()
    return s if s and s.lower() != "nan" else "__NA__"

def split_items(x):
    if pd.isna(x): return []
    s = str(x).strip()
    if not s or s.lower() == "nan": return []
    return [clean_id(v) for v in s.split(",") if clean_id(v)]

def dedup_preserve_order(items):
    seen = set(); out = []
    for item in items:
        if item not in seen: seen.add(item); out.append(item)
    return out

def length_bucket(n):
    if n <= 0: return "0"
    if n == 1: return "1"
    if n == 2: return "2"
    if n <= 4: return "3-4"
    if n <= 7: return "5-7"
    if n <= 15: return "8-15"
    if n <= 30: return "16-30"
    return "31+"

def prepare_frames(train, test, user):
    for df in (train, test, user):
        df["uid"] = df["uid"].map(clean_id)
    train["target_iid"] = train["target_iid"].map(clean_id)
    cat_cols = [c for c in user.columns if c.startswith("u_cat_")]
    for col in cat_cols: user[col] = user[col].map(clean_scalar)
    train = train.merge(user[["uid"] + cat_cols], on="uid", how="left", validate="many_to_one")
    test = test.merge(user[["uid"] + cat_cols], on="uid", how="left", validate="many_to_one")
    for df in (train, test):
        for col in cat_cols: df[col] = df[col].map(clean_scalar)
    return train, test, cat_cols

# ============================================================
# Recall Statistics
# ============================================================
FloatCounter = Counter

class CandidateMeta:
    __slots__ = ("rrf", "source_count", "best_rank")
    def __init__(self, rrf=0.0, source_count=0, best_rank=10000):
        self.rrf = rrf; self.source_count = source_count; self.best_rank = best_rank

class RecallStats:
    def __init__(self, df, cat_cols):
        self.cat_cols = list(cat_cols)
        self.pair_cols = list(combinations(self.cat_cols[:MAX_PAIR_FEATURE_COLS], 2))
        self.n_rows = max(len(df), 1)
        self.target_counts = Counter()
        self.sequence_item_counts = Counter()
        self.last1_to_target = defaultdict(Counter)
        self.last2_to_target = defaultdict(Counter)
        self.last3_to_target = defaultdict(Counter)
        self.recent_item_to_target = defaultdict(Counter)
        self.transition = defaultdict(Counter)
        self.segment_to_target = {c: defaultdict(Counter) for c in self.cat_cols}
        self.segment_pair_to_target = {pair: defaultdict(Counter) for pair in self.pair_cols}
        self.profile_to_target = defaultdict(Counter)
        self.length_to_target = defaultdict(Counter)
        # NEW: group-level item-pop for coarse generalization (V43 finding)
        self.group_fine_to_target = defaultdict(Counter)     # u_cat 01-04
        self.group_mid_to_target = defaultdict(Counter)      # u_cat 01-02
        self.group_coarse_to_target = defaultdict(Counter)   # u_cat 01 only
        self._build(df)
        self.target_total = float(sum(self.target_counts.values())) or 1.0
        self.seq_item_total = float(sum(self.sequence_item_counts.values())) or 1.0
        self.n_targets = max(len(self.target_counts), 1)
        self.global_top = [iid for iid, _ in self.target_counts.most_common(TOP_GLOBAL)]
        self.target_rank = {iid: r for r, (iid, _) in enumerate(self.target_counts.most_common(), start=1)}

    def _build(self, df):
        cols = list(df.columns)
        for tup in df.itertuples(index=False, name=None):
            row = dict(zip(cols, tup))
            target = clean_id(row["target_iid"])
            if not target: continue
            raw = split_items(row.get("item_seq_raw"))
            dedup = split_items(row.get("item_seq_dedup"))
            if not dedup: dedup = dedup_preserve_order(raw)
            self.target_counts[target] += 1.0
            self.sequence_item_counts.update(raw)
            if dedup:
                self.last1_to_target[dedup[-1]][target] += 1.0
                if len(dedup) >= 2: self.last2_to_target[tuple(dedup[-2:])][target] += 1.0
                if len(dedup) >= 3: self.last3_to_target[tuple(dedup[-3:])][target] += 1.0
                for distance, item in enumerate(reversed(dedup[-RECENT_ITEMS:])):
                    self.recent_item_to_target[item][target] += 0.82 ** distance
            for left, right in zip(raw[:-1], raw[1:]):
                self.transition[left][right] += 1.0
            for col in self.cat_cols:
                self.segment_to_target[col][clean_scalar(row.get(col))][target] += 1.0
            for c1, c2 in self.pair_cols:
                key = (clean_scalar(row.get(c1)), clean_scalar(row.get(c2)))
                self.segment_pair_to_target[(c1, c2)][key][target] += 1.0
            profile = tuple(clean_scalar(row.get(c)) for c in self.cat_cols)
            self.profile_to_target[profile][target] += 1.0
            self.length_to_target[length_bucket(len(raw))][target] += 1.0
            # NEW: multi-granularity group item popularity (V43 validated)
            grp_feats = [clean_scalar(row.get(c)) for c in self.cat_cols[:4]]
            self.group_fine_to_target[tuple(grp_feats)][target] += 1.0
            self.group_mid_to_target[tuple(grp_feats[:2])][target] += 1.0
            self.group_coarse_to_target[grp_feats[0]][target] += 1.0

    def _add_source(self, meta, counter, topn, weight):
        if not counter: return
        ranked = (counter.most_common(topn) if isinstance(counter, Counter)
                  else sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:topn])
        for rank, (iid, _) in enumerate(ranked, start=1):
            m = meta.setdefault(iid, CandidateMeta())
            m.rrf += weight / (10.0 + rank)
            m.source_count += 1
            m.best_rank = min(m.best_rank, rank)

    def generate_candidates(self, row):
        raw = split_items(row.get("item_seq_raw"))
        dedup = split_items(row.get("item_seq_dedup"))
        if not dedup: dedup = dedup_preserve_order(raw)
        meta: Dict[str, CandidateMeta] = {}

        # History/repeat
        hs = Counter()
        rc = Counter(raw)
        for iid, cnt in rc.items(): hs[iid] += 2.0 * math.log1p(cnt)
        for d, iid in enumerate(reversed(raw)): hs[iid] += 3.0 / (1.0 + d)
        self._add_source(meta, hs, min(len(hs), 60), 7.0)

        if dedup:
            self._add_source(meta, self.last1_to_target.get(dedup[-1], {}), TOP_LAST1, 6.0)
            self._add_source(meta, self.transition.get(dedup[-1], {}), TOP_TRANSITION, 2.0)
            if len(dedup) >= 2:
                self._add_source(meta, self.last2_to_target.get(tuple(dedup[-2:]), {}), TOP_LAST2, 8.0)
            if len(dedup) >= 3:
                self._add_source(meta, self.last3_to_target.get(tuple(dedup[-3:]), {}), TOP_LAST3, 10.0)
            for d, item in enumerate(reversed(dedup[-RECENT_ITEMS:])):
                self._add_source(meta, self.recent_item_to_target.get(item, {}), TOP_RECENT_EACH, 4.0 * (0.82 ** d))

        profile = tuple(clean_scalar(row.get(c)) for c in self.cat_cols)
        self._add_source(meta, self.profile_to_target.get(profile, {}), TOP_PROFILE, 5.0)
        for col in self.cat_cols:
            self._add_source(meta, self.segment_to_target[col].get(clean_scalar(row.get(col)), {}), TOP_SEGMENT, 2.0)
        for c1, c2 in self.pair_cols:
            key = (clean_scalar(row.get(c1)), clean_scalar(row.get(c2)))
            self._add_source(meta, self.segment_pair_to_target[(c1, c2)].get(key, {}), TOP_SEGMENT_PAIR, 3.0)
        self._add_source(meta, self.length_to_target.get(length_bucket(len(raw)), {}), TOP_LENGTH_BUCKET, 1.0)
        # NEW: group-level recalls (V43 — our validated signals)
        grp_key_fine = tuple(clean_scalar(row.get(c)) for c in self.cat_cols[:4])
        grp_key_mid = tuple(grp_key_fine[:2])
        grp_key_coarse = grp_key_fine[0]
        self._add_source(meta, self.group_fine_to_target.get(grp_key_fine, {}), TOP_GROUP_FINE, 4.0)
        self._add_source(meta, self.group_mid_to_target.get(grp_key_mid, {}), TOP_GROUP_MID, 1.5)
        self._add_source(meta, self.group_coarse_to_target.get(grp_key_coarse, {}), TOP_GROUP_COARSE, 0.8)
        self._add_source(meta, self.target_counts, TOP_GLOBAL, 0.8)

        ranked = sorted(meta, key=lambda iid: (-meta[iid].rrf, -meta[iid].source_count, meta[iid].best_rank, iid))
        return ranked[:MAX_CANDIDATES], meta

    def target_prior(self, iid):
        return (self.target_counts.get(iid, 0.0) + 0.5) / (self.target_total + 0.5 * self.n_targets)

    def seq_prior(self, iid):
        return (self.sequence_item_counts.get(iid, 0.0) + 0.5) / (self.seq_item_total + 0.5 * max(len(self.sequence_item_counts), 1))

    def _cond_feat(self, counter, iid, alpha, prior=None):
        prior = self.target_prior(iid) if prior is None else prior
        cnt = float(counter.get(iid, 0.0)) if counter else 0.0
        total = float(sum(counter.values())) if counter else 0.0
        prob = (cnt + alpha * prior) / (total + alpha) if total + alpha > 0 else prior
        lift = math.log((prob + 1e-12) / (prior + 1e-12))
        return math.log1p(cnt), math.log1p(total), prob, lift, cnt / (total + 1e-12)

    def pair_features(self, row, iid, meta):
        raw = split_items(row.get("item_seq_raw"))
        dedup = split_items(row.get("item_seq_dedup"))
        if not dedup: dedup = dedup_preserve_order(raw)
        rc = Counter(raw)
        rp = {}
        for d, item in enumerate(reversed(raw)): rp.setdefault(item, d)
        sl = len(raw); dl = len(dedup)
        prior = self.target_prior(iid); sprior = self.seq_prior(iid)

        f = {
            "rrf": meta.rrf, "source_count": meta.source_count,
            "best_rank_recip": 0.0 if meta.best_rank >= 10000 else 1.0 / meta.best_rank,
            "seq_len": sl, "log_seq_len": math.log1p(sl), "dedup_len": dl,
            "unique_ratio": dl / max(sl, 1), "is_cold_0": float(sl == 0),
            "is_cold_le2": float(sl <= 2),
            "global_target_logcnt": math.log1p(self.target_counts.get(iid, 0.0)),
            "global_target_prior": prior, "global_target_logprior": math.log(prior + 1e-12),
            "global_target_rank_recip": 1.0 / self.target_rank.get(iid, self.n_targets + 1),
            "seq_item_logcnt": math.log1p(self.sequence_item_counts.get(iid, 0.0)),
            "seq_item_prior": sprior, "history_present": float(iid in rc),
            "history_count": rc.get(iid, 0), "history_logcount": math.log1p(rc.get(iid, 0)),
            "history_freq_ratio": rc.get(iid, 0) / max(sl, 1),
            "history_recency": 0.0 if iid not in rp else 1.0 / (1.0 + rp[iid]),
            "is_last_raw": float(bool(raw) and iid == raw[-1]),
            "is_last_dedup": float(bool(dedup) and iid == dedup[-1]),
            "is_second_last": float(len(dedup) >= 2 and iid == dedup[-2]),
            "is_third_last": float(len(dedup) >= 3 and iid == dedup[-3]),
        }

        # Suffix features
        specs = []
        if dedup: specs.append(("last1", self.last1_to_target.get(dedup[-1], {}), 15.0))
        else: specs.append(("last1", {}, 15.0))
        if len(dedup) >= 2: specs.append(("last2", self.last2_to_target.get(tuple(dedup[-2:]), {}), 8.0))
        else: specs.append(("last2", {}, 8.0))
        if len(dedup) >= 3: specs.append(("last3", self.last3_to_target.get(tuple(dedup[-3:]), {}), 4.0))
        else: specs.append(("last3", {}, 4.0))
        for pr, ct, a in specs:
            c, s, p, l, m = self._cond_feat(ct, iid, a)
            f.update({f"{pr}_logcnt": c, f"{pr}_logsupport": s, f"{pr}_prob": p,
                       f"{pr}_lift": l, f"{pr}_mle": m})

        # Recent-item aggregate
        rps, rpm, rls, rlm, rcs, rhs = 0.0, 0.0, 0.0, -50.0, 0.0, 0.0
        for d, item in enumerate(reversed(dedup[-RECENT_ITEMS:])):
            ct = self.recent_item_to_target.get(item, {})
            c, _, p, l, _ = self._cond_feat(ct, iid, 12.0)
            w = 0.82 ** d
            rps += w * p; rpm = max(rpm, p); rls += w * l
            rlm = max(rlm, l); rcs += w * c; rhs += float(ct.get(iid, 0.0) > 0)
        f.update({"recent_prob_sum": rps, "recent_prob_max": rpm, "recent_lift_sum": rls,
                   "recent_lift_max": rlm if dedup else 0.0, "recent_logcount_sum": rcs,
                   "recent_hit_sources": rhs})

        # Transition
        tc = self.transition.get(dedup[-1], {}) if dedup else {}
        c, s, p, l, m = self._cond_feat(tc, iid, 15.0, prior=sprior)
        f.update({"trans_logcnt": c, "trans_logsupport": s, "trans_prob": p,
                   "trans_lift": l, "trans_mle": m})

        # Profile (all 8 u_cat: fine grain, reference code)
        prof = tuple(clean_scalar(row.get(c)) for c in self.cat_cols)
        c, s, p, l, m = self._cond_feat(self.profile_to_target.get(prof, {}), iid, 10.0)
        f.update({"prof_logcnt": c, "prof_logsupport": s, "prof_prob": p,
                   "prof_lift": l, "prof_mle": m})

        # NEW: coarse-profile features (V43 finding: better generalization)
        grp_feats = [clean_scalar(row.get(c)) for c in self.cat_cols[:4]]
        for name, grp_counter in [
            ("grp_fine", self.group_fine_to_target.get(tuple(grp_feats), {})),
            ("grp_mid", self.group_mid_to_target.get(tuple(grp_feats[:2]), {})),
            ("grp_coarse", self.group_coarse_to_target.get(grp_feats[0], {})),
        ]:
            c, s, p, l, m = self._cond_feat(grp_counter, iid, 8.0)
            f.update({f"{name}_logcnt": c, f"{name}_logsupport": s, f"{name}_prob": p,
                       f"{name}_lift": l, f"{name}_mle": m})

        # Length bucket
        c, s, p, l, m = self._cond_feat(self.length_to_target.get(length_bucket(sl), {}), iid, 30.0)
        f.update({"len_logcnt": c, "len_logsupport": s, "len_prob": p,
                   "len_lift": l, "len_mle": m})

        # Segment features
        sps, sls, scs = [], [], []
        for col in self.cat_cols:
            c, _, p, l, _ = self._cond_feat(
                self.segment_to_target[col].get(clean_scalar(row.get(col)), {}), iid, 25.0)
            f[f"seg_{col}_logcnt"] = c; f[f"seg_{col}_prob"] = p; f[f"seg_{col}_lift"] = l
            sps.append(p); sls.append(l); scs.append(c)

        # Segment-pair features
        pps, pls, pcs = [], [], []
        for c1, c2 in self.pair_cols:
            key = (clean_scalar(row.get(c1)), clean_scalar(row.get(c2)))
            c, _, p, l, _ = self._cond_feat(
                self.segment_pair_to_target[(c1, c2)].get(key, {}), iid, 15.0)
            pps.append(p); pls.append(l); pcs.append(c)

        f.update({
            "seg_prob_max": max(sps, default=prior), "seg_prob_mean": float(np.mean(sps)) if sps else prior,
            "seg_lift_max": max(sls, default=0.0), "seg_lift_mean": float(np.mean(sls)) if sls else 0.0,
            "seg_logcnt_sum": sum(scs),
            "pair_prob_max": max(pps, default=prior), "pair_prob_mean": float(np.mean(pps)) if pps else prior,
            "pair_lift_max": max(pls, default=0.0), "pair_lift_mean": float(np.mean(pls)) if pls else 0.0,
            "pair_logcnt_sum": sum(pcs),
        })

        # Interactions
        f["repeat_x_recent"] = f["history_present"] * f["history_recency"]
        f["last1_x_history"] = f["last1_prob"] * (1.0 + f["history_logcount"])
        f["suffix_best_prob"] = max(f["last1_prob"], f["last2_prob"], f["last3_prob"])
        f["suffix_best_lift"] = max(f["last1_lift"], f["last2_lift"], f["last3_lift"])

        # source-hit features (proven stable gain, from output657)
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


# ============================================================
# OOF and Training
# ============================================================
def make_group_folds(groups, n_splits, seed):
    ug = np.array(pd.unique(pd.Series(groups)), dtype=object)
    rng = np.random.default_rng(seed); rng.shuffle(ug)
    n = min(n_splits, len(ug))
    g2f = {g: i % n for i, g in enumerate(ug)}
    return np.array([g2f[g] for g in groups], dtype=np.int16)

def build_oof(train, cat_cols, base_dir):
    """Build OOF, saving each fold to disk to keep RAM low.
    Output is identical to original in-memory version."""
    fold_ids = make_group_folds(train["uid"].tolist(), N_FOLDS, RANDOM_STATE)
    recalls = []; qid = 0
    oof_dir = os.path.join(base_dir, "oof_folds")
    os.makedirs(oof_dir, exist_ok=True)
    fold_nums = sorted(np.unique(fold_ids))

    for fold in fold_nums:
        fit_df = train.loc[fold_ids != fold].reset_index(drop=True)
        val_df = train.loc[fold_ids == fold].reset_index(drop=True)
        print(f"  Fold {fold+1}: fit={len(fit_df)}, valid={len(val_df)}")
        stats = RecallStats(fit_df, cat_cols)
        rows = []
        pos = 0
        aug_count = 0
        for idx, (_, row) in enumerate(val_df.iterrows()):
            tgt = clean_id(row["target_iid"])

            # ---- 原始完整序列行 ----
            cands, meta = stats.generate_candidates(row)
            ret = int(tgt and tgt in cands)
            if tgt and tgt not in cands:
                cands.append(tgt); meta[tgt] = CandidateMeta()
            if len(cands) > MAX_CANDIDATES:
                if tgt in cands[MAX_CANDIDATES:]: cands = cands[:MAX_CANDIDATES-1] + [tgt]
                else: cands = cands[:MAX_CANDIDATES]
            for iid in cands:
                feats = stats.pair_features(row, iid, meta.get(iid, CandidateMeta()))
                feats["qid"] = qid; feats["uid"] = row["uid"]
                feats["candidate_iid"] = iid
                feats["label"] = int(iid == tgt); feats["positive_retrieved"] = int(ret)
                feats["augmented"] = 0
                rows.append(feats)

            # ---- V19: 3种截断增强行 (共享 qid 和 label) ----
            for aug_mode in ["testmix", "last3", "empty"]:
                aug_seed = RANDOM_STATE + 1900 + idx * 313 + hash(aug_mode) % 10000
                aug_rng = np.random.default_rng(aug_seed)
                val_row = build_truncated_row(row, aug_rng, mode=aug_mode)
                cands_a, meta_a = stats.generate_candidates(val_row)
                ret_a = int(tgt and tgt in cands_a)
                if tgt and tgt not in cands_a:
                    cands_a.append(tgt); meta_a[tgt] = CandidateMeta()
                if len(cands_a) > MAX_CANDIDATES:
                    if tgt in cands_a[MAX_CANDIDATES:]: cands_a = cands_a[:MAX_CANDIDATES-1] + [tgt]
                    else: cands_a = cands_a[:MAX_CANDIDATES]
                for iid in cands_a:
                    feats = stats.pair_features(val_row, iid, meta_a.get(iid, CandidateMeta()))
                    feats["qid"] = qid; feats["uid"] = row["uid"]
                    feats["candidate_iid"] = iid
                    feats["label"] = int(iid == tgt); feats["positive_retrieved"] = int(ret_a)
                    feats["augmented"] = 1
                    rows.append(feats)
            aug_count += 3

            pos += ret
            qid += 1
        recall = pos / max(len(val_df), 1)
        print(f"    recall@{MAX_CANDIDATES}: {recall:.4f}")
        recalls.append(recall)
        df_fold = pd.DataFrame(rows)
        fpath = os.path.join(oof_dir, f"fold_{int(fold)}.parquet")
        df_fold.to_parquet(fpath, index=False)
        print(f"    saved {len(df_fold):,} rows")
        del stats, rows, df_fold; gc.collect()

    # Reload all folds (data identical to original)
    dfs = [pd.read_parquet(os.path.join(oof_dir, f"fold_{int(f)}.parquet"))
           for f in fold_nums]
    oof = pd.concat(dfs, ignore_index=True)
    # Clean up disk
    del dfs; gc.collect()
    for f in fold_nums:
        os.remove(os.path.join(oof_dir, f"fold_{int(f)}.parquet"))
    os.rmdir(oof_dir)
    return oof, recalls

def feature_cols(frame):
    ignore = {"qid", "uid", "candidate_iid", "label", "positive_retrieved", "augmented"}
    cols = [c for c in frame.columns if c not in ignore]
    # V10: 剔除冗余特征
    cols = [c for c in cols if not c.endswith("_mle")]
    cols = [c for c in cols if not c.endswith("_logsupport")]
    return cols

def fm(frame, cols):
    return frame.reindex(columns=cols, fill_value=0.0).replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)

def split_valid(frame, frac=0.12):
    qids = frame["qid"].drop_duplicates().to_numpy().copy()
    rng = np.random.default_rng(RANDOM_STATE + 19); rng.shuffle(qids)
    nv = max(1, int(round(len(qids) * frac)))
    vq = set(qids[:nv])
    t = frame.loc[~frame["qid"].isin(vq)].sort_values("qid", kind="stable").reset_index(drop=True)
    v = frame.loc[frame["qid"].isin(vq)].sort_values("qid", kind="stable").reset_index(drop=True)
    return t, v

def natural_recall_subset(frame):
    gq = frame.loc[frame["positive_retrieved"].eq(1), "qid"].drop_duplicates()
    return frame.loc[frame["qid"].isin(gq)].sort_values("qid", kind="stable").reset_index(drop=True)

def ndcg_hit_from_frame(frame, score_col):
    ndcgs, hits = [], []
    for _, g in frame.groupby("qid", sort=False):
        if int(g["positive_retrieved"].iloc[0]) == 0:
            ndcgs.append(0.0); hits.append(0.0); continue
        r = g.sort_values(score_col, ascending=False, kind="stable").head(10)
        hp = np.flatnonzero(r["label"].to_numpy() == 1)
        if len(hp):
            ndcgs.append(1.0 / math.log2(int(hp[0]) + 2.0)); hits.append(1.0)
        else: ndcgs.append(0.0); hits.append(0.0)
    return float(np.mean(ndcgs)), float(np.mean(hits))

def group_sizes(frame):
    return frame.groupby("qid", sort=False).size().astype(int).tolist()

def train_rankers(oof):
    """V17: 纯 LGB LambdaMART 单模 (XGB 已完全移除)"""
    cols = feature_cols(oof)
    mt, mv = split_valid(oof)
    mve = natural_recall_subset(mv)

    Xt = fm(mt, cols); yt = mt["label"].astype(np.int8).to_numpy()
    gtrain = group_sizes(mt)
    Xve = fm(mve, cols); yve = mve["label"].astype(np.int8).to_numpy()
    give = group_sizes(mve)

    print("\nTraining LightGBM LambdaMART...")
    lgbm = lgb.LGBMRanker(objective="lambdarank", metric="ndcg", label_gain=[0,1],
        n_estimators=2500, learning_rate=0.025, num_leaves=31, max_depth=-1,
        min_child_samples=45, subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.15, reg_lambda=1.5, max_bin=127,
        random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=-1)
    lgbm.fit(Xt, yt, group=gtrain,
             eval_set=[(Xve, yve)], eval_group=[give], eval_at=[10],
             callbacks=[lgb.early_stopping(120, verbose=True), lgb.log_evaluation(100)])
    lb = max(int(lgbm.best_iteration_ or 600), 100)

    # Meta-validation (LGB only)
    Xv = fm(mv, cols)
    ms = mv.copy()
    ms["lgb_score"] = lgbm.predict(Xv, num_iteration=lb).astype(np.float32)
    n, h = ndcg_hit_from_frame(ms, "lgb_score")
    print(f"\nMeta-validation LGB: Hit@10={h:.4f}  NDCG@10={n:.4f}  best_iter={lb}")

    if os.environ.get("SKIP_FINAL_REFIT", "0") == "1":
        print("SKIP_FINAL_REFIT=1: use meta-trained LGB directly")
        return lgbm, cols, lb

    # Refit on all OOF
    del lgbm, Xt, Xv, Xve, yt, yve, mv; gc.collect()
    oof_s = oof.sort_values("qid", kind="stable").reset_index(drop=True)
    Xa = fm(oof_s, cols); ya = oof_s["label"].astype(np.int8).to_numpy()
    gsz = group_sizes(oof_s)

    print("\nRefitting LightGBM...")
    flgb = lgb.LGBMRanker(objective="lambdarank", metric="ndcg", label_gain=[0,1],
        n_estimators=lb, learning_rate=0.025, num_leaves=31, max_depth=-1,
        min_child_samples=45, subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.15, reg_lambda=1.5, max_bin=127,
        random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=-1)
    flgb.fit(Xa, ya, group=gsz)
    save_dir = os.path.dirname(os.path.abspath(__file__))
    flgb.booster_.save_model(os.path.join(save_dir, "output", "model_lgb.txt"))
    with open(os.path.join(save_dir, "output", "feature_cols.json"), "w") as f:
        json.dump(cols, f)
    print("Model saved: output/model_lgb.txt + feature_cols.json")
    del Xa, ya, oof_s, gsz; gc.collect()

    return flgb, cols, lb


# ============================================================
# Test Prediction
# ============================================================
def predict_test(flgb, cols, train, test, cat_cols, sample):
    """V17: 纯 LGB 单模预测"""
    print("\nBuilding test statistics...")
    stats = RecallStats(train, cat_cols)

    print("Predicting test (batch)...")
    preds = {}  # uid → top10 list
    batch = int(os.environ.get('PRED_BATCH_SIZE', '300'))
    for start in range(0, len(test), batch):
        stop = min(start + batch, len(test))
        bt = test.iloc[start:stop]
        rows = []; qmap = {}
        for i, (_, row) in enumerate(bt.iterrows()):
            q = start + i; qmap[q] = row["uid"]
            cands, meta = stats.generate_candidates(row)
            for iid in cands:
                feats = stats.pair_features(row, iid, meta.get(iid, CandidateMeta()))
                feats["qid"] = q; feats["candidate_iid"] = iid; rows.append(feats)
        cf = pd.DataFrame(rows)
        X = fm(cf, cols)
        cf["lgb_score"] = flgb.predict(X).astype(np.float32)
        for q, g in cf.groupby("qid", sort=False):
            uid = qmap[int(q)]
            ranked = g.sort_values("lgb_score", ascending=False, kind="stable")["candidate_iid"].drop_duplicates().tolist()
            top = ranked[:10]
            if len(top) < 10:
                for iid in stats.global_top:
                    if iid not in top: top.append(iid)
                    if len(top) >= 10: break
            preds[uid] = top
        print(f"  {stop}/{len(test)}")
        del cf, X, rows; gc.collect()

    return preds, stats


# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save_to", type=str, default=None)
    p.add_argument("--infer_only", action="store_true", help="跳过训练, 从已保存模型直接推理")
    args = p.parse_args()

    random.seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)
    print("=== V17: LGB LambdaMART (testmix augmented) ===\n")

    print("Loading data...")
    train, test, user, sample = load_data()
    train, test, cat_cols = prepare_frames(train, test, user)
    print(f"  Train={len(train)}, Test={len(test)}, Cats={len(cat_cols)}")

    if args.infer_only:
        # 从已保存模型直接推理
        code_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(code_dir, "output", "model_lgb.txt")
        cols_path = os.path.join(code_dir, "output", "feature_cols.json")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件未找到: {model_path}\n请先运行训练: python train.py --save_to output")
        print(f"\n加载模型: {model_path}")
        flgb = lgb.Booster(model_file=model_path)
        with open(cols_path) as f:
            cols = json.load(f)
        print(f"  特征数: {len(cols)}")
        preds, fstats = predict_test(flgb, cols, train, test, cat_cols, sample)
    else:
        print(f"\nBuilding OOF ranking data ({N_FOLDS}-fold)...")
        oof_dir = os.path.dirname(os.path.abspath(__file__))
        oof, recalls = build_oof(train, cat_cols, oof_dir)
        print(f"  OOF recall: {np.mean(recalls):.4f} +- {np.std(recalls):.4f}")
        print(f"  OOF rows: {len(oof):,}")

        flgb, cols, lb = train_rankers(oof)
        preds, fstats = predict_test(flgb, cols, train, test, cat_cols, sample)

    # Output: LGB single model only
    base_dir = args.save_to or os.path.dirname(os.path.abspath(__file__))
    out = sample.copy()
    out["uid"] = out["uid"].map(clean_id)
    out["prediction"] = out["uid"].map(
        lambda uid: ",".join(preds.get(uid, fstats.global_top[:10])[:10]))
    path = os.path.join(base_dir, "A2.csv")
    out.to_csv(path, index=False)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
