"""V40: 可维护性增强版。

模型、特征、五折OOF、视图级qid隔离和candidate=55均继承V37。
本版本只集中将参数配置置于文件头部，无其他变化。
"""
from __future__ import annotations

import os
import subprocess
import sys


REQUIRED_PYTHONHASHSEED = "0"
if os.environ.get("PYTHONHASHSEED") != REQUIRED_PYTHONHASHSEED:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = REQUIRED_PYTHONHASHSEED
    completed = subprocess.run(
        [sys.executable, os.path.abspath(__file__), *sys.argv[1:]],
        env=env,
        check=False,
    )
    raise SystemExit(completed.returncode)

os.environ["PYTHONHASHSEED"] = REQUIRED_PYTHONHASHSEED
for thread_env_key in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[thread_env_key] = "1"

import argparse
import gc
import json
import math
import random
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb

# ============================================================
# Config: paths, data generation, recall and model parameters
# ============================================================
VERSION = "T2_LGB_V40_MAINTAINABLE_DETERMINISTIC_V37"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths. DATA_ROOT can override the fallback search list.
DATA_ROOT_ENV_VAR = "DATA_ROOT"
DATA_DIR_CANDIDATES = (
    os.path.join(SCRIPT_DIR, "data"),
    os.path.abspath(os.path.join(SCRIPT_DIR, "..", "c55_lgb", "data")),
)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
OOF_WORK_DIR = os.path.join(SCRIPT_DIR, "oof_folds")
MODEL_PATH = os.path.join(OUTPUT_DIR, "model_lgb.txt")
FEATURE_COLS_PATH = os.path.join(OUTPUT_DIR, "feature_cols.json")
SUBMISSION_FILENAME = "A2.csv"

# Reproducibility and runtime.
RANDOM_STATE = 2025
N_JOBS = 1
N_FOLDS = int(os.environ.get("N_FOLDS", "5"))
OOF_CHUNK_USERS = int(os.environ.get("OOF_CHUNK_USERS", "2000"))
PRED_BATCH_SIZE = int(os.environ.get("PRED_BATCH_SIZE", "300"))
SKIP_FINAL_REFIT = os.environ.get("SKIP_FINAL_REFIT", "0") == "1"
META_VALID_FRAC = 0.12
FALLBACK_BEST_ITERATION = 600
MIN_FINAL_ESTIMATORS = 100

# Candidate and feature controls.
MAX_CANDIDATES = int(os.environ.get("MAX_CANDIDATES", "55"))
RECENT_ITEMS = 6
RECENCY_DECAY = 0.82
MAX_PAIR_FEATURE_COLS = 6
AUGMENTATION_MODES = ("testmix", "last3", "empty")
AUGMENTATION_SEED_BASE_OFFSET = 1900
AUGMENTATION_ROW_SEED_STRIDE = 313
# Equal to hash(mode) % 10000 under PYTHONHASHSEED=0 in V37. Explicit
# offsets preserve V37 behavior without making augmentation depend on hash().
AUGMENTATION_SEED_OFFSETS = {
    "testmix": 2510,
    "last3": 9097,
    "empty": 2636,
}

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

# LightGBM. Initial training adds LGB_MAX_ESTIMATORS; final refit uses the
# selected best_iteration instead.
LGB_MAX_ESTIMATORS = 2500
LGB_EARLY_STOPPING_ROUNDS = 120
LGB_LOG_EVALUATION_PERIOD = 100
LGB_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "label_gain": [0, 1],
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
    "n_jobs": N_JOBS,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
}

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
    env_path = os.environ.get(DATA_ROOT_ENV_VAR)
    if env_path and os.path.exists(os.path.join(env_path, "train.csv")):
        return env_path
    for p in DATA_DIR_CANDIDATES:
        if p and os.path.exists(os.path.join(p, "train.csv")):
            return p
    raise FileNotFoundError(
        "未找到train.csv；请修改顶部DATA_DIR_CANDIDATES或设置DATA_ROOT"
    )

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
        # V21: _cond_feat total cache (performance)
        self._total_cache = {}                  # id(counter) -> total for _cond_feat
        # V25: I2I co-occurrence
        self.item_cooc = defaultdict(Counter)   # {item: Counter({item: count})}
        self.item_occur = Counter()             # {item: user_count}
        self._build(df)
        self._build_jaccard_cache()             # pre-compute Jaccard lookup
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
                    self.recent_item_to_target[item][target] += RECENCY_DECAY ** distance
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
            # V25: I2I co-occurrence (dedup items within same user)
            if len(dedup) >= 2:
                self.item_occur.update(set(dedup))
                for a, b in combinations(dedup, 2):
                    if a < b: self.item_cooc[a][b] += 1.0
                    else:     self.item_cooc[b][a] += 1.0
            elif dedup:
                self.item_occur[dedup[0]] += 1.0

    def _add_source(self, meta, counter, topn, weight):
        if not counter: return
        ranked = (counter.most_common(topn) if isinstance(counter, Counter)
                  else sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:topn])
        for rank, (iid, _) in enumerate(ranked, start=1):
            m = meta.setdefault(iid, CandidateMeta())
            m.rrf += weight / (10.0 + rank)
            m.source_count += 1
            m.best_rank = min(m.best_rank, rank)


    # V25: I2I Jaccard cache + O(1) lookup
    def _build_jaccard_cache(self):
        """Pre-compute Jaccard for all co-occurring pairs once."""
        cache = {}
        for a, coocs in self.item_cooc.items():
            occ_a = self.item_occur.get(a, 0.0)
            for b, cooc in coocs.items():
                occ_b = self.item_occur.get(b, 0.0)
                denom = occ_a + occ_b - cooc
                cache[(a, b)] = cooc / denom if denom > 0 else 0.0
        self._jaccard_cache = cache

    def _i2i_jaccard(self, item_a, item_b):
        """O(1) Jaccard lookup."""
        if item_a == item_b:
            return 1.0
        a, b = (item_a, item_b) if item_a < item_b else (item_b, item_a)
        return self._jaccard_cache.get((a, b), 0.0)

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
                self._add_source(
                    meta,
                    self.recent_item_to_target.get(item, {}),
                    TOP_RECENT_EACH,
                    4.0 * (RECENCY_DECAY ** d),
                )

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

    def _cond_feat(self, counter, iid, alpha, prior):
        """Compute conditional features. prior must be pre-computed by caller."""
        cnt = float(counter.get(iid, 0.0)) if counter else 0.0
        total = self._total_cache.get(id(counter))
        if total is None:
            total = float(sum(counter.values())) if counter else 0.0
            self._total_cache[id(counter)] = total
        prob = (cnt + alpha * prior) / (total + alpha) if total + alpha > 0 else prior
        lift = math.log((prob + 1e-12) / (prior + 1e-12))
        return math.log1p(cnt), math.log1p(total), prob, lift, cnt / (total + 1e-12)

    def pair_features(self, raw, dedup, cat_vals, iid, meta):
        """Compute features for a (user, candidate) pair.

        Args:
            raw: pre-split raw item list (from build_truncated_row or split_items)
            dedup: pre-split dedup item list
            cat_vals: pre-cleaned dict {col_name: scalar_value}
            iid: candidate item id
            meta: CandidateMeta
        """
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
            c, s, p, l, m = self._cond_feat(ct, iid, a, prior)
            f.update({f"{pr}_logcnt": c, f"{pr}_logsupport": s, f"{pr}_prob": p,
                       f"{pr}_lift": l, f"{pr}_mle": m})

        # Recent-item aggregate
        rps, rpm, rls, rlm, rcs, rhs = 0.0, 0.0, 0.0, -50.0, 0.0, 0.0
        for d, item in enumerate(reversed(dedup[-RECENT_ITEMS:])):
            ct = self.recent_item_to_target.get(item, {})
            c, _, p, l, _ = self._cond_feat(ct, iid, 12.0, prior)
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
        prof = tuple(cat_vals[c] for c in self.cat_cols)
        c, s, p, l, m = self._cond_feat(self.profile_to_target.get(prof, {}), iid, 10.0, prior)
        f.update({"prof_logcnt": c, "prof_logsupport": s, "prof_prob": p,
                   "prof_lift": l, "prof_mle": m})

        # NEW: coarse-profile features (V43 finding: better generalization)
        grp_feats = [cat_vals[c] for c in self.cat_cols[:4]]
        for name, grp_counter in [
            ("grp_fine", self.group_fine_to_target.get(tuple(grp_feats), {})),
            ("grp_mid", self.group_mid_to_target.get(tuple(grp_feats[:2]), {})),
            ("grp_coarse", self.group_coarse_to_target.get(grp_feats[0], {})),
        ]:
            c, s, p, l, m = self._cond_feat(grp_counter, iid, 8.0, prior)
            f.update({f"{name}_logcnt": c, f"{name}_logsupport": s, f"{name}_prob": p,
                       f"{name}_lift": l, f"{name}_mle": m})

        # Length bucket
        c, s, p, l, m = self._cond_feat(self.length_to_target.get(length_bucket(sl), {}), iid, 30.0, prior)
        f.update({"len_logcnt": c, "len_logsupport": s, "len_prob": p,
                   "len_lift": l, "len_mle": m})

        # Segment features
        sps, sls, scs = [], [], []
        for col in self.cat_cols:
            c, _, p, l, _ = self._cond_feat(
                self.segment_to_target[col].get(cat_vals[col], {}), iid, 25.0, prior)
            f[f"seg_{col}_logcnt"] = c; f[f"seg_{col}_prob"] = p; f[f"seg_{col}_lift"] = l
            sps.append(p); sls.append(l); scs.append(c)

        # Segment-pair features
        pps, pls, pcs = [], [], []
        for c1, c2 in self.pair_cols:
            key = (cat_vals[c1], cat_vals[c2])
            c, _, p, l, _ = self._cond_feat(
                self.segment_pair_to_target[(c1, c2)].get(key, {}), iid, 15.0, prior)
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

        # V25: I2I CF co-occurrence features
        i2i_sims = []
        if dedup:
            for item in dedup:
                sim = self._i2i_jaccard(item, iid)
                if sim > 0:
                    i2i_sims.append(sim)
        f["i2i_jaccard_max"] = max(i2i_sims) if i2i_sims else 0.0
        f["i2i_jaccard_sum"] = float(np.sum(i2i_sims)) if i2i_sims else 0.0
        f["i2i_jaccard_last1"] = self._i2i_jaccard(dedup[-1], iid) if dedup else 0.0

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

def prep_row(row, cat_cols):
    """Pre-compute per-row invariant parts to avoid repeated split_items/clean_scalar."""
    raw = split_items(row.get("item_seq_raw"))
    dedup = split_items(row.get("item_seq_dedup"))
    if not dedup: dedup = dedup_preserve_order(raw)
    cat_vals = {c: clean_scalar(row.get(c)) for c in cat_cols}
    return raw, dedup, cat_vals

def compress_oof_frame(frame):
    """Reduce OOF memory before parquet persistence and final training."""
    for col in frame.columns:
        if col in ("uid", "candidate_iid"):
            continue
        if col == "qid":
            frame[col] = frame[col].astype(np.int32)
        elif col in ("label", "positive_retrieved", "augmented"):
            frame[col] = frame[col].astype(np.int8)
        else:
            frame[col] = frame[col].astype(np.float32)
    return frame

def build_oof(train, cat_cols, oof_dir=OOF_WORK_DIR):
    """Build OOF, saving each fold to disk to keep RAM low."""
    fold_ids = make_group_folds(train["uid"].tolist(), N_FOLDS, RANDOM_STATE)
    recalls = []; qid = 0; source_rows = 0
    os.makedirs(oof_dir, exist_ok=True)
    for old_name in os.listdir(oof_dir):
        if old_name.endswith(".parquet"):
            os.remove(os.path.join(oof_dir, old_name))
    fold_nums = sorted(np.unique(fold_ids))
    chunk_paths = []

    for fold in fold_nums:
        fit_df = train.loc[fold_ids != fold].reset_index(drop=True)
        val_df = train.loc[fold_ids == fold].reset_index(drop=True)
        print(f"  Fold {fold+1}: fit={len(fit_df)}, valid={len(val_df)}")
        stats = RecallStats(fit_df, cat_cols)
        rows = []
        saved_rows = 0
        chunk_id = 0
        pos = 0

        def add_view_rows(view_row, tgt, view_qid, augmented):
            raw, dedup, cat_vals = prep_row(view_row, cat_cols)
            cands, meta = stats.generate_candidates(view_row)
            ret = int(tgt and tgt in cands)
            if tgt and tgt not in cands:
                cands.append(tgt); meta[tgt] = CandidateMeta()
            if len(cands) > MAX_CANDIDATES:
                if tgt in cands[MAX_CANDIDATES:]: cands = cands[:MAX_CANDIDATES-1] + [tgt]
                else: cands = cands[:MAX_CANDIDATES]
            for iid in cands:
                feats = stats.pair_features(raw, dedup, cat_vals, iid, meta.get(iid, CandidateMeta()))
                feats["qid"] = view_qid; feats["uid"] = view_row["uid"]
                feats["candidate_iid"] = iid
                feats["label"] = int(iid == tgt); feats["positive_retrieved"] = int(ret)
                feats["augmented"] = int(augmented)
                rows.append(feats)
            return ret

        def flush_rows():
            nonlocal rows, saved_rows, chunk_id
            if not rows:
                return
            df_chunk = compress_oof_frame(pd.DataFrame(rows))
            fpath = os.path.join(oof_dir, f"fold_{int(fold)}_chunk_{chunk_id:03d}.parquet")
            df_chunk.to_parquet(fpath, index=False)
            chunk_paths.append(fpath)
            saved_rows += len(df_chunk)
            print(f"    saved chunk {chunk_id}: {len(df_chunk):,} rows")
            chunk_id += 1
            rows = []
            del df_chunk; gc.collect()

        for idx, (_, row) in enumerate(val_df.iterrows()):
            tgt = clean_id(row["target_iid"])

            # ---- 原始完整序列视图 ----
            ret = add_view_rows(row, tgt, qid, augmented=0)
            qid += 1

            # ---- V19: 3种截断增强视图; V32: 每种视图独立 qid ----
            for aug_mode in AUGMENTATION_MODES:
                aug_seed = (
                    RANDOM_STATE
                    + AUGMENTATION_SEED_BASE_OFFSET
                    + idx * AUGMENTATION_ROW_SEED_STRIDE
                    + AUGMENTATION_SEED_OFFSETS[aug_mode]
                )
                aug_rng = np.random.default_rng(aug_seed)
                val_row = build_truncated_row(row, aug_rng, mode=aug_mode)
                add_view_rows(val_row, tgt, qid, augmented=1)
                qid += 1

            pos += ret
            source_rows += 1
            if (idx + 1) % OOF_CHUNK_USERS == 0:
                flush_rows()
        recall = pos / max(len(val_df), 1)
        print(f"    recall@{MAX_CANDIDATES}: {recall:.4f}")
        recalls.append(recall)
        flush_rows()
        print(f"    ranking groups: {len(val_df) * 4:,} (4 views per source row)")
        print(f"    saved fold rows: {saved_rows:,}")
        del stats, rows; gc.collect()

    # Reload all folds into memory (matches V19 flow exactly)
    dfs = [pd.read_parquet(path) for path in chunk_paths]
    oof = pd.concat(dfs, ignore_index=True, copy=False)
    del dfs; gc.collect()
    for path in chunk_paths:
        os.remove(path)
    os.rmdir(oof_dir)
    print(f"  Source rows: {source_rows:,}; ranking groups: {oof['qid'].nunique():,}")
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

def split_valid(frame, frac=META_VALID_FRAC):
    units = frame["uid"].drop_duplicates().to_numpy().copy()
    rng = np.random.default_rng(RANDOM_STATE + 19); rng.shuffle(units)
    nv = max(1, int(round(len(units) * frac)))
    vu = set(units[:nv])
    t = frame.loc[~frame["uid"].isin(vu)].sort_values("qid", kind="stable").reset_index(drop=True)
    v = frame.loc[frame["uid"].isin(vu)].sort_values("qid", kind="stable").reset_index(drop=True)
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
    """V21: LGB LambdaMART (matches V19 training flow exactly)."""
    cols = feature_cols(oof)
    mt, mv = split_valid(oof)
    mve = natural_recall_subset(mv)

    Xt = fm(mt, cols); yt = mt["label"].astype(np.int8).to_numpy()
    gtrain = group_sizes(mt)
    Xve = fm(mve, cols); yve = mve["label"].astype(np.int8).to_numpy()
    give = group_sizes(mve)

    print("\nTraining LightGBM LambdaMART...")
    lgbm = lgb.LGBMRanker(n_estimators=LGB_MAX_ESTIMATORS, **LGB_PARAMS)
    lgbm.fit(Xt, yt, group=gtrain,
             eval_set=[(Xve, yve)], eval_group=[give], eval_at=[10],
             callbacks=[
                 lgb.early_stopping(LGB_EARLY_STOPPING_ROUNDS, verbose=True),
                 lgb.log_evaluation(LGB_LOG_EVALUATION_PERIOD),
             ])
    lb = max(
        int(lgbm.best_iteration_ or FALLBACK_BEST_ITERATION),
        MIN_FINAL_ESTIMATORS,
    )

    # Meta-validation (LGB only)
    Xv = fm(mv, cols)
    ms = mv.copy()
    ms["lgb_score"] = lgbm.predict(Xv, num_iteration=lb).astype(np.float32)
    n, h = ndcg_hit_from_frame(ms, "lgb_score")
    print(f"\nMeta-validation LGB: Hit@10={h:.4f}  NDCG@10={n:.4f}  best_iter={lb}")
    ms_orig = ms.loc[ms["augmented"].eq(0)].reset_index(drop=True)
    if len(ms_orig):
        n0, h0 = ndcg_hit_from_frame(ms_orig, "lgb_score")
        print(f"Meta-validation original-view: Hit@10={h0:.4f}  NDCG@10={n0:.4f}")

    if SKIP_FINAL_REFIT:
        print("SKIP_FINAL_REFIT=1: use meta-trained LGB directly")
        return lgbm, cols, lb

    # Refit on all OOF (matches V19 exactly)
    del lgbm, Xt, Xv, Xve, yt, yve, mv; gc.collect()
    oof_s = oof.sort_values("qid", kind="stable").reset_index(drop=True)
    Xa = fm(oof_s, cols); ya = oof_s["label"].astype(np.int8).to_numpy()
    gsz = group_sizes(oof_s)

    print("\nRefitting LightGBM...")
    flgb = lgb.LGBMRanker(n_estimators=lb, **LGB_PARAMS)
    flgb.fit(Xa, ya, group=gsz)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    flgb.booster_.save_model(MODEL_PATH)
    with open(FEATURE_COLS_PATH, "w") as f:
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
    batch = PRED_BATCH_SIZE
    for start in range(0, len(test), batch):
        stop = min(start + batch, len(test))
        bt = test.iloc[start:stop]
        rows = []; qmap = {}
        for i, (_, row) in enumerate(bt.iterrows()):
            q = start + i; qmap[q] = row["uid"]
            raw, dedup, cat_vals = prep_row(row, cat_cols)
            cands, meta = stats.generate_candidates(row)
            for iid in cands:
                feats = stats.pair_features(raw, dedup, cat_vals, iid, meta.get(iid, CandidateMeta()))
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
    p.add_argument(
        "--print_config",
        action="store_true",
        help="打印集中配置后退出，不读取数据或训练",
    )
    args = p.parse_args()

    random.seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)
    config_snapshot = {
        "version": VERSION,
        "paths": {
            "data_root_env": DATA_ROOT_ENV_VAR,
            "data_candidates": list(DATA_DIR_CANDIDATES),
            "output_dir": OUTPUT_DIR,
            "oof_work_dir": OOF_WORK_DIR,
            "model_path": MODEL_PATH,
            "feature_cols_path": FEATURE_COLS_PATH,
            "submission_filename": SUBMISSION_FILENAME,
        },
        "runtime": {
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "n_jobs": N_JOBS,
            "n_folds": N_FOLDS,
            "random_state": RANDOM_STATE,
            "oof_chunk_users": OOF_CHUNK_USERS,
            "prediction_batch_size": PRED_BATCH_SIZE,
            "skip_final_refit": SKIP_FINAL_REFIT,
        },
        "candidate": {
            "max_candidates": MAX_CANDIDATES,
            "recent_items": RECENT_ITEMS,
            "recency_decay": RECENCY_DECAY,
            "max_pair_feature_cols": MAX_PAIR_FEATURE_COLS,
            "augmentation_modes": list(AUGMENTATION_MODES),
            "augmentation_seed_offsets": AUGMENTATION_SEED_OFFSETS,
        },
        "lightgbm": {
            **LGB_PARAMS,
            "max_estimators": LGB_MAX_ESTIMATORS,
            "fallback_best_iteration": FALLBACK_BEST_ITERATION,
            "min_final_estimators": MIN_FINAL_ESTIMATORS,
            "early_stopping_rounds": LGB_EARLY_STOPPING_ROUNDS,
            "log_evaluation_period": LGB_LOG_EVALUATION_PERIOD,
        },
    }
    if args.print_config:
        print(json.dumps(config_snapshot, ensure_ascii=False, indent=2))
        return

    print(f"=== {VERSION} ===")
    print(
        f"PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')}, "
        f"n_jobs={LGB_PARAMS['n_jobs']}, folds={N_FOLDS}, "
        f"candidates={MAX_CANDIDATES}\n"
    )

    print("Loading data...")
    train, test, user, sample = load_data()
    train, test, cat_cols = prepare_frames(train, test, user)
    print(f"  Train={len(train)}, Test={len(test)}, Cats={len(cat_cols)}")

    if args.infer_only:
        # 从已保存模型直接推理
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"模型文件未找到: {MODEL_PATH}\n请先运行训练: python train.py"
            )
        print(f"\n加载模型: {MODEL_PATH}")
        flgb = lgb.Booster(model_file=MODEL_PATH)
        with open(FEATURE_COLS_PATH) as f:
            cols = json.load(f)
        print(f"  特征数: {len(cols)}")
        preds, fstats = predict_test(flgb, cols, train, test, cat_cols, sample)
    else:
        print(f"\nBuilding OOF ranking data ({N_FOLDS}-fold)...")
        oof, recalls = build_oof(train, cat_cols)
        print(f"  OOF recall: {np.mean(recalls):.4f} +- {np.std(recalls):.4f}")
        print(f"  OOF rows: {len(oof):,}")

        flgb, cols, lb = train_rankers(oof)
        preds, fstats = predict_test(flgb, cols, train, test, cat_cols, sample)

    # Output: LGB single model only
    base_dir = args.save_to or SCRIPT_DIR
    os.makedirs(base_dir, exist_ok=True)
    out = sample.copy()
    out["uid"] = out["uid"].map(clean_id)
    out["prediction"] = out["uid"].map(
        lambda uid: ",".join(preds.get(uid, fstats.global_top[:10])[:10]))
    path = os.path.join(base_dir, SUBMISSION_FILENAME)
    out.to_csv(path, index=False)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
