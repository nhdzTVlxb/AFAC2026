
"""V46: 稳健共现特征 + 三层缩放残差网络 + NDCG 早停排序器。

相对基础 T2 版本的改动：
1. 在 Jaccard/PMI 之外加入 NPMI、共现支持度和近期加权聚合特征。
2. 使用三个带可学习缩放的瓶颈残差块，并保留 gated interaction/wide 分支。
3. 对 lift 和标准化输入做裁剪，按验证集 NDCG@10 调度学习率并早停。
4. 五折 OOF、qid 隔离、候选数和增强视图保持不变。
"""
from __future__ import annotations

import os
import subprocess
import sys
import math
import random
import argparse
import gc
import json
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Dict, List, Tuple, Optional

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# Config & Env
# ============================================================
REQUIRED_PYTHONHASHSEED = "0"
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
IN_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ

if IN_KAGGLE:
    os.environ["PYTHONHASHSEED"] = REQUIRED_PYTHONHASHSEED
else:
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

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

for thread_env_key in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"
):
    os.environ[thread_env_key] = "1"

os.environ.setdefault(
    "DATA_ROOT", os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data", "rec_data"))
)

# ============================================================
# Config: paths, data generation, recall and model parameters
# ============================================================
VERSION = "T2_NN_V46_DEEPER_COOCCURRENCE_NDCG_ES"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_ROOT_ENV_VAR = "DATA_ROOT"
DATA_DIR_CANDIDATES = (
    os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data", "rec_data")),
    os.path.join(SCRIPT_DIR, "data"),
    os.path.join(SCRIPT_DIR, "rec_data"),
)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", SCRIPT_DIR)
OOF_WORK_DIR = os.path.join(OUTPUT_DIR, "oof_folds")
MODEL_PATH = os.path.join(OUTPUT_DIR, "model_nn_t2_deeper.pt")
FEATURE_COLS_PATH = os.path.join(OUTPUT_DIR, "feature_cols.json")
OOF_CACHE_PATH = os.environ.get("OOF_CACHE_PATH", os.path.join(OUTPUT_DIR, "oof_cache.parquet"))
SUBMISSION_FILENAME = "A2_t2_deeper.csv"

# Reproducibility and runtime.
RANDOM_STATE = 2025
N_JOBS = 1
N_FOLDS = int(os.environ.get("N_FOLDS", "5"))
OOF_CHUNK_USERS = int(os.environ.get("OOF_CHUNK_USERS", "2000"))
PRED_BATCH_SIZE = int(os.environ.get("PRED_BATCH_SIZE", "300"))
SKIP_FINAL_REFIT = os.environ.get("SKIP_FINAL_REFIT", "0") == "1"
META_VALID_FRAC = 0.12
NN_EPOCHS = int(os.environ.get("NN_EPOCHS", "12"))
NN_BATCH_SIZE = int(os.environ.get("NN_BATCH_SIZE", "4096"))
NN_EARLY_STOPPING = int(os.environ.get("NN_EARLY_STOPPING", "3"))
NN_USE_AMP = os.environ.get("NN_USE_AMP", "0") == "1"
NN_LEARNING_RATE = float(os.environ.get("NN_LEARNING_RATE", "0.00015"))
REUSE_OOF_CACHE = os.environ.get("REUSE_OOF_CACHE", "1") == "1"

# Candidate and feature controls.
MAX_CANDIDATES = int(os.environ.get("MAX_CANDIDATES", "55"))
RECENT_ITEMS = 6
RECENCY_DECAY = 0.82
MAX_PAIR_FEATURE_COLS = 6
AUGMENTATION_MODES = ("testmix", "last3", "empty")
AUGMENTATION_SEED_BASE_OFFSET = 1900
AUGMENTATION_ROW_SEED_STRIDE = 313
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

# LightGBM params (fallback or for baseline)
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
# Utilities
# ============================================================
def find_data_path() -> str:
    env_path = os.environ.get(DATA_ROOT_ENV_VAR)
    if env_path and os.path.exists(os.path.join(env_path, "train.csv")):
        return env_path
    for p in DATA_DIR_CANDIDATES:
        if p and os.path.exists(os.path.join(p, "train.csv")):
            return p
    raise FileNotFoundError("未找到train.csv")

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
# Data Augmentation
# ============================================================
def testmix_truncate(seq_str: str, rng: np.random.Generator) -> str:
    return _truncate_seq(seq_str, rng, mode="testmix")

def last3_truncate(seq_str: str, rng: np.random.Generator) -> str:
    return _truncate_seq(seq_str, rng, mode="last3")

def empty_truncate(seq_str: str, rng: np.random.Generator) -> str:
    return ""

def _truncate_seq(seq_str: str, rng: np.random.Generator, mode: str) -> str:
    items = [x.strip() for x in str(seq_str).split(",") if x.strip()] if seq_str else []
    n = len(items)
    if n == 0: return ""
    if mode == "empty": return ""
    elif mode == "last3":
        k = min(int(rng.integers(1, 4)), n)
    elif mode == "testmix":
        r = rng.random()
        if r < 0.3515: k = 0
        elif r < 0.4518: k = 1
        elif r < 0.8992: k = min(int(rng.integers(2, 4)), n)
        elif r < 0.9006: k = min(int(rng.integers(4, 6)), n)
        elif r < 0.9053: k = min(int(rng.integers(6, 11)), n)
        elif r < 0.9202: k = min(int(rng.integers(11, 31)), n)
        else:
            if n <= 31: k = n
            else: k = int(rng.integers(31, min(n, 200) + 1))
    else:
        return seq_str
    items = items[-min(k, n):] if k > 0 else []
    return ",".join(items) if items else ""

def build_truncated_row(row, rng, mode="testmix"):
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
# Recall Statistics (V41: PMI Optimization)
# ============================================================
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
        self.group_fine_to_target = defaultdict(Counter)
        self.group_mid_to_target = defaultdict(Counter)
        self.group_coarse_to_target = defaultdict(Counter)
        self._total_cache = {}
        
        # I2I stats
        self.item_cooc = defaultdict(Counter)
        self.item_occur = Counter()
        
        self._build(df)
        self.n_targets = max(len(self.target_counts), 1)
        self.target_rank = {iid: r for r, (iid, _) in enumerate(self.target_counts.most_common(), start=1)}
        
        # Pre-compute Caches
        self._build_jaccard_cache()
        self._build_pmi_cache() # V41 NEW
        
        self.target_total = float(sum(self.target_counts.values())) or 1.0
        self.seq_item_total = float(sum(self.sequence_item_counts.values())) or 1.0
        self.global_top = [iid for iid, _ in self.target_counts.most_common(TOP_GLOBAL)]

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
            
            grp_feats = [clean_scalar(row.get(c)) for c in self.cat_cols[:4]]
            self.group_fine_to_target[tuple(grp_feats)][target] += 1.0
            self.group_mid_to_target[tuple(grp_feats[:2])][target] += 1.0
            self.group_coarse_to_target[grp_feats[0]][target] += 1.0
            
            # I2I
            if len(dedup) >= 2:
                self.item_occur.update(set(dedup))
                for a, b in combinations(dedup, 2):
                    if a < b: self.item_cooc[a][b] += 1.0
                    else: self.item_cooc[b][a] += 1.0
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

    def _build_jaccard_cache(self):
        cache = {}
        for a, coocs in self.item_cooc.items():
            occ_a = self.item_occur.get(a, 0.0)
            for b, cooc in coocs.items():
                occ_b = self.item_occur.get(b, 0.0)
                denom = occ_a + occ_b - cooc
                cache[(a, b)] = cooc / denom if denom > 0 else 0.0
        self._jaccard_cache = cache

    def _i2i_jaccard(self, item_a, item_b):
        if item_a == item_b: return 1.0
        a, b = (item_a, item_b) if item_a < item_b else (item_b, item_a)
        return self._jaccard_cache.get((a, b), 0.0)

    # PMI, normalized PMI and co-occurrence support share one sparse cache.
    def _build_pmi_cache(self):
        pmi_cache = {}
        npmi_cache = {}
        support_cache = {}
        n_rows = float(self.n_rows)
        for a, coocs in self.item_cooc.items():
            occ_a = self.item_occur.get(a, 1.0)
            for b, cooc in coocs.items():
                occ_b = self.item_occur.get(b, 1.0)
                if occ_a > 0 and occ_b > 0 and cooc > 0:
                    pmi = math.log((cooc * n_rows) / (occ_a * occ_b))
                    joint_surprisal = -math.log(cooc / n_rows)
                    npmi = pmi / joint_surprisal if joint_surprisal > 1e-12 else 0.0
                    key = (a, b)
                    pmi_cache[key] = max(-10.0, min(10.0, pmi))
                    npmi_cache[key] = max(-1.0, min(1.0, npmi))
                    support_cache[key] = math.log1p(cooc)
        self._pmi_cache = pmi_cache
        self._npmi_cache = npmi_cache
        self._cooc_support_cache = support_cache

    def _i2i_pmi(self, item_a, item_b):
        if item_a == item_b: return 0.0
        a, b = (item_a, item_b) if item_a < item_b else (item_b, item_a)
        return self._pmi_cache.get((a, b), -10.0)

    def _i2i_npmi(self, item_a, item_b):
        if item_a == item_b: return 0.0
        a, b = (item_a, item_b) if item_a < item_b else (item_b, item_a)
        return self._npmi_cache.get((a, b), -1.0)

    def _i2i_logsupport(self, item_a, item_b):
        if item_a == item_b: return 0.0
        a, b = (item_a, item_b) if item_a < item_b else (item_b, item_a)
        return self._cooc_support_cache.get((a, b), 0.0)

    def generate_candidates(self, row):
        raw = split_items(row.get("item_seq_raw"))
        dedup = split_items(row.get("item_seq_dedup"))
        if not dedup: dedup = dedup_preserve_order(raw)
        meta: Dict[str, CandidateMeta] = {}

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
        cnt = float(counter.get(iid, 0.0)) if counter else 0.0
        total = self._total_cache.get(id(counter))
        if total is None:
            total = float(sum(counter.values())) if counter else 0.0
            self._total_cache[id(counter)] = total
        prob = (cnt + alpha * prior) / (total + alpha) if total + alpha > 0 else prior
        
        # V41: Stability Enhancement
        # Clamp lift to avoid extreme values affecting NN training
        lift = math.log((prob + 1e-12) / (prior + 1e-12))
        lift = max(-5.0, min(5.0, lift))
        
        return math.log1p(cnt), math.log1p(total), prob, lift, cnt / (total + 1e-12)

    def pair_features(self, raw, dedup, cat_vals, iid, meta):
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

        tc = self.transition.get(dedup[-1], {}) if dedup else {}
        c, s, p, l, m = self._cond_feat(tc, iid, 15.0, prior=sprior)
        f.update({"trans_logcnt": c, "trans_logsupport": s, "trans_prob": p,
                  "trans_lift": l, "trans_mle": m})

        prof = tuple(cat_vals[c] for c in self.cat_cols)
        c, s, p, l, m = self._cond_feat(self.profile_to_target.get(prof, {}), iid, 10.0, prior)
        f.update({"prof_logcnt": c, "prof_logsupport": s, "prof_prob": p,
                  "prof_lift": l, "prof_mle": m})

        grp_feats = [cat_vals[c] for c in self.cat_cols[:4]]
        for name, grp_counter in [
            ("grp_fine", self.group_fine_to_target.get(tuple(grp_feats), {})),
            ("grp_mid", self.group_mid_to_target.get(tuple(grp_feats[:2]), {})),
            ("grp_coarse", self.group_coarse_to_target.get(grp_feats[0], {})),
        ]:
            c, s, p, l, m = self._cond_feat(grp_counter, iid, 8.0, prior)
            f.update({f"{name}_logcnt": c, f"{name}_logsupport": s, f"{name}_prob": p,
                      f"{name}_lift": l, f"{name}_mle": m})

        c, s, p, l, m = self._cond_feat(self.length_to_target.get(length_bucket(sl), {}), iid, 30.0, prior)
        f.update({"len_logcnt": c, "len_logsupport": s, "len_prob": p,
                  "len_lift": l, "len_mle": m})

        sps, sls, scs = [], [], []
        for col in self.cat_cols:
            c, _, p, l, _ = self._cond_feat(
                self.segment_to_target[col].get(cat_vals[col], {}), iid, 25.0, prior)
            f[f"seg_{col}_logcnt"] = c; f[f"seg_{col}_prob"] = p; f[f"seg_{col}_lift"] = l
            sps.append(p); sls.append(l); scs.append(c)

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

        f["repeat_x_recent"] = f["history_present"] * f["history_recency"]
        f["last1_x_history"] = f["last1_prob"] * (1.0 + f["history_logcount"])
        f["suffix_best_prob"] = max(f["last1_prob"], f["last2_prob"], f["last3_prob"])
        f["suffix_best_lift"] = max(f["last1_lift"], f["last2_lift"], f["last3_lift"])

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
            "src_supervised_hit_sum": float(f["last1_logcnt"] > 0) + float(f["last2_logcnt"] > 0) + float(f["last3_logcnt"] > 0) + float(f["recent_hit_sources"] > 0) + float(f["trans_logcnt"] > 0),
            "src_profile_pair_strength": f["prof_prob"] + f["pair_prob_max"],
            "src_suffix_recent_strength": f["suffix_best_prob"] + f["recent_prob_max"],
            "src_rrf_x_best_rank": f["rrf"] * f["best_rank_recip"],
        })

        # PMI is strong but biased toward rare pairs; NPMI and support expose
        # confidence so the ranker can distinguish rare accidents from signal.
        i2i_pmis, i2i_npmis, i2i_supports = [], [], []
        recent_pmi_weighted = 0.0
        recent_npmi_weighted = 0.0
        if dedup:
            for item in dedup:
                pmi = self._i2i_pmi(item, iid)
                support = self._i2i_logsupport(item, iid)
                if support > 0.0:
                    i2i_pmis.append(pmi)
                    i2i_npmis.append(self._i2i_npmi(item, iid))
                    i2i_supports.append(support)
            for distance, item in enumerate(reversed(dedup[-RECENT_ITEMS:])):
                weight = RECENCY_DECAY ** distance
                if self._i2i_logsupport(item, iid) > 0.0:
                    recent_pmi_weighted += weight * self._i2i_pmi(item, iid)
                    recent_npmi_weighted += weight * self._i2i_npmi(item, iid)

            f["i2i_pmi_max"] = max(i2i_pmis) if i2i_pmis else -10.0
            f["i2i_pmi_sum"] = float(np.sum(i2i_pmis)) if i2i_pmis else 0.0
            f["i2i_pmi_mean"] = float(np.mean(i2i_pmis)) if i2i_pmis else -10.0
            f["i2i_pmi_last1"] = self._i2i_pmi(dedup[-1], iid)
            f["i2i_npmi_max"] = max(i2i_npmis) if i2i_npmis else -1.0
            f["i2i_npmi_mean"] = float(np.mean(i2i_npmis)) if i2i_npmis else -1.0
            f["i2i_npmi_last1"] = self._i2i_npmi(dedup[-1], iid)
            f["i2i_logsupport_max"] = max(i2i_supports, default=0.0)
            f["i2i_logsupport_sum"] = float(np.sum(i2i_supports)) if i2i_supports else 0.0
            f["i2i_recent_pmi_weighted"] = recent_pmi_weighted
            f["i2i_recent_npmi_weighted"] = recent_npmi_weighted
            f["i2i_positive_pmi_ratio"] = (
                sum(value > 0.0 for value in i2i_pmis) / max(len(i2i_pmis), 1)
            )

            i2i_sims = []
            for item in dedup:
                sim = self._i2i_jaccard(item, iid)
                if sim > 0: i2i_sims.append(sim)
            f["i2i_jaccard_max"] = max(i2i_sims) if i2i_sims else 0.0
            f["i2i_jaccard_sum"] = float(np.sum(i2i_sims)) if i2i_sims else 0.0
            f["i2i_jaccard_last1"] = self._i2i_jaccard(dedup[-1], iid) if dedup else 0.0
        else:
            f.update({
                "i2i_pmi_max": -10.0,
                "i2i_pmi_sum": 0.0,
                "i2i_pmi_mean": -10.0,
                "i2i_pmi_last1": -10.0,
                "i2i_npmi_max": -1.0,
                "i2i_npmi_mean": -1.0,
                "i2i_npmi_last1": -1.0,
                "i2i_logsupport_max": 0.0,
                "i2i_logsupport_sum": 0.0,
                "i2i_recent_pmi_weighted": 0.0,
                "i2i_recent_npmi_weighted": 0.0,
                "i2i_positive_pmi_ratio": 0.0,
                "i2i_jaccard_max": 0.0,
                "i2i_jaccard_sum": 0.0,
                "i2i_jaccard_last1": 0.0,
            })

        f["i2i_pmi_x_suffix"] = f["i2i_pmi_max"] * f["suffix_best_prob"]
        f["i2i_npmi_x_recent"] = f["i2i_npmi_max"] * f["recent_prob_max"]
        f["profile_x_suffix"] = f["prof_prob"] * f["suffix_best_prob"]
        f["target_vs_sequence_logpop"] = (
            f["global_target_logcnt"] - f["seq_item_logcnt"]
        )
        f["rrf_x_source_count"] = f["rrf"] * math.log1p(f["source_count"])

        return f

# ============================================================
# OOF and Training Helpers
# ============================================================
def make_group_folds(groups, n_splits, seed):
    ug = np.array(pd.unique(pd.Series(groups)), dtype=object)
    rng = np.random.default_rng(seed); rng.shuffle(ug)
    n = min(n_splits, len(ug))
    g2f = {g: i % n for i, g in enumerate(ug)}
    return np.array([g2f[g] for g in groups], dtype=np.int16)

def prep_row(row, cat_cols):
    raw = split_items(row.get("item_seq_raw"))
    dedup = split_items(row.get("item_seq_dedup"))
    if not dedup: dedup = dedup_preserve_order(raw)
    cat_vals = {c: clean_scalar(row.get(c)) for c in cat_cols}
    return raw, dedup, cat_vals

def compress_oof_frame(frame):
    for col in frame.columns:
        if col in ("uid", "candidate_iid"): continue
        if col == "qid": frame[col] = frame[col].astype(np.int32)
        elif col in ("label", "positive_retrieved", "augmented"): frame[col] = frame[col].astype(np.int8)
        else: frame[col] = frame[col].astype(np.float32)
    return frame

def build_oof(train, cat_cols, oof_dir=OOF_WORK_DIR):
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
        print(f" Fold {fold+1}: fit={len(fit_df)}, valid={len(val_df)}")
        stats = RecallStats(fit_df, cat_cols)
        rows = []; saved_rows = 0; chunk_id = 0; pos = 0

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
            if not rows: return
            df_chunk = compress_oof_frame(pd.DataFrame(rows))
            fpath = os.path.join(oof_dir, f"fold_{int(fold)}_chunk_{chunk_id:03d}.parquet")
            df_chunk.to_parquet(fpath, index=False)
            chunk_paths.append(fpath)
            saved_rows += len(df_chunk)
            print(f" saved chunk {chunk_id}: {len(df_chunk):,} rows")
            chunk_id += 1
            rows = []
            del df_chunk; gc.collect()

        for idx, (_, row) in enumerate(val_df.iterrows()):
            tgt = clean_id(row["target_iid"])
            ret = add_view_rows(row, tgt, qid, augmented=0)
            qid += 1
            for aug_mode in AUGMENTATION_MODES:
                aug_seed = (RANDOM_STATE + AUGMENTATION_SEED_BASE_OFFSET + idx * AUGMENTATION_ROW_SEED_STRIDE + AUGMENTATION_SEED_OFFSETS[aug_mode])
                aug_rng = np.random.default_rng(aug_seed)
                val_row = build_truncated_row(row, aug_rng, mode=aug_mode)
                add_view_rows(val_row, tgt, qid, augmented=1)
                qid += 1
            pos += ret; source_rows += 1
            if (idx + 1) % OOF_CHUNK_USERS == 0: flush_rows()
        recall = pos / max(len(val_df), 1)
        print(f" recall@{MAX_CANDIDATES}: {recall:.4f}")
        recalls.append(recall)
        flush_rows()
        print(f" ranking groups: {len(val_df) * 4:,}")
        print(f" saved fold rows: {saved_rows:,}")
        del stats, rows; gc.collect()

    dfs = [pd.read_parquet(path) for path in chunk_paths]
    oof = pd.concat(dfs, ignore_index=True, copy=False)
    del dfs; gc.collect()
    for path in chunk_paths: os.remove(path)
    os.rmdir(oof_dir)
    print(f" Source rows: {source_rows:,}; ranking groups: {oof['qid'].nunique():,}")
    return oof, recalls

def feature_cols(frame):
    ignore = {"qid", "uid", "candidate_iid", "label", "positive_retrieved", "augmented"}
    cols = [c for c in frame.columns if c not in ignore]
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

def ndcg_hit_from_frame(frame, score_col):
    ndcgs, hits = [], []
    for _, g in frame.groupby("qid", sort=False):
        if int(g["positive_retrieved"].iloc[0]) == 0: ndcgs.append(0.0); hits.append(0.0); continue
        r = g.sort_values(score_col, ascending=False, kind="stable").head(10)
        hp = np.flatnonzero(r["label"].to_numpy() == 1)
        if len(hp):
            ndcgs.append(1.0 / math.log2(int(hp[0]) + 2.0)); hits.append(1.0)
        else: ndcgs.append(0.0); hits.append(0.0)
    return float(np.mean(ndcgs)), float(np.mean(hits))

# ============================================================
# V46: Three-block scaled residual ranker
# ============================================================
class BottleneckResidualBlock(nn.Module):
    def __init__(self, dim=320, bottleneck=160, dropout=0.10, residual_scale=0.25):
        super().__init__()
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, dim),
            nn.Dropout(dropout * 0.6),
        )

    def forward(self, x):
        return x + self.residual_scale * self.net(self.norm(x))


class ScaledResidualRanker(nn.Module):
    def __init__(self, in_dim, feature_mean=None, feature_std=None, width=448, interaction_dim=72):
        super().__init__()
        if feature_mean is None: feature_mean = np.zeros(in_dim, dtype=np.float32)
        if feature_std is None: feature_std = np.ones(in_dim, dtype=np.float32)
        self.register_buffer("feature_mean", torch.as_tensor(feature_mean, dtype=torch.float32))
        self.register_buffer("feature_std", torch.as_tensor(feature_std, dtype=torch.float32))
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Dropout(0.16),
        )
        self.residuals = nn.Sequential(
            BottleneckResidualBlock(width, bottleneck=224, dropout=0.14, residual_scale=0.35),
            BottleneckResidualBlock(width, bottleneck=160, dropout=0.12, residual_scale=0.25),
            BottleneckResidualBlock(width, bottleneck=112, dropout=0.10, residual_scale=0.15),
        )
        self.main_head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 224),
            nn.GELU(),
            nn.Dropout(0.14),
            nn.Linear(224, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        self.interaction_left = nn.Linear(in_dim, interaction_dim)
        self.interaction_gate = nn.Linear(in_dim, interaction_dim)
        self.interaction_head = nn.Sequential(
            nn.LayerNorm(interaction_dim),
            nn.Linear(interaction_dim, 36),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Linear(36, 1),
        )
        self.wide = nn.Linear(in_dim, 1)

    def forward(self, x):
        x = torch.clamp((x - self.feature_mean) / self.feature_std, -12.0, 12.0)
        main_score = self.main_head(self.residuals(self.input_proj(x))).squeeze(-1)
        interaction = torch.tanh(self.interaction_left(x)) * torch.sigmoid(self.interaction_gate(x))
        interaction_score = self.interaction_head(interaction).squeeze(-1)
        wide_score = self.wide(x).squeeze(-1)
        return main_score + 0.22 * interaction_score + 0.06 * wide_score

class BPRLoss(nn.Module):
    def forward(self, pos_score, neg_score):
        return F.softplus(-(pos_score - neg_score)).mean()

def build_bpr_pair_indices(y, qid, max_pairs_per_group=20):
    y = np.asarray(y)
    qid = np.asarray(qid)
    pos_indices = []; neg_indices = []
    rng = np.random.default_rng(42)
    
    order = pd.DataFrame({"idx": np.arange(len(y)), "qid": qid, "label": y})
    
    for _, g in order.groupby("qid"):
        pos = g.loc[g["label"] > 0, "idx"].values
        neg = g.loc[g["label"] <= 0, "idx"].values
        if len(pos) == 0 or len(neg) == 0: continue
        
        n_pairs = min(len(pos) * len(neg), max_pairs_per_group)
        
        p_idx = rng.choice(pos, n_pairs, replace=True)
        n_idx = rng.choice(neg, n_pairs, replace=True)
        
        pos_indices.append(p_idx)
        neg_indices.append(n_idx)
        
    if len(pos_indices) == 0: raise ValueError("No valid BPR pairs")
    return np.concatenate(pos_indices), np.concatenate(neg_indices)


def compute_feature_stats(X, chunk_rows=100_000):
    X = np.asarray(X, dtype=np.float32)
    sums = np.zeros(X.shape[1], dtype=np.float64)
    squared_sums = np.zeros(X.shape[1], dtype=np.float64)
    for start in range(0, len(X), chunk_rows):
        chunk = X[start:start + chunk_rows].astype(np.float64)
        sums += chunk.sum(axis=0)
        squared_sums += np.square(chunk).sum(axis=0)
    mean = sums / max(len(X), 1)
    variance = np.maximum(squared_sums / max(len(X), 1) - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std = np.where(np.isfinite(std) & (std > 1e-3), std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)

class NNRankerWrapper:
    def __init__(self):
        self.model = None
        self.in_dim = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(
        self,
        X,
        y,
        qid,
        X_val=None,
        y_val=None,
        qid_val=None,
        val_meta=None,
        epochs=NN_EPOCHS,
        batch_size=NN_BATCH_SIZE,
    ):
        X_array = np.asarray(X, dtype=np.float32)
        X_cpu = torch.from_numpy(X_array)
        pos_idx, neg_idx = build_bpr_pair_indices(y, qid)
        print(f"BPR training pairs: {len(pos_idx):,}")

        ds = TensorDataset(torch.from_numpy(pos_idx), torch.from_numpy(neg_idx))
        del pos_idx, neg_idx
        use_amp = self.device == "cuda" and NN_USE_AMP
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=use_amp,
            num_workers=0,
            generator=torch.Generator().manual_seed(RANDOM_STATE + 1),
        )

        val_dl = None
        X_val_cpu = None
        if X_val is not None:
            X_val_cpu = torch.from_numpy(np.asarray(X_val, dtype=np.float32))
            val_pos_idx, val_neg_idx = build_bpr_pair_indices(y_val, qid_val, max_pairs_per_group=4)
            print(f"BPR validation pairs: {len(val_pos_idx):,}")
            val_ds = TensorDataset(torch.from_numpy(val_pos_idx), torch.from_numpy(val_neg_idx))
            del val_pos_idx, val_neg_idx
            val_dl = DataLoader(
                val_ds,
                batch_size=batch_size * 2,
                shuffle=False,
                pin_memory=use_amp,
                num_workers=0,
            )

        self.in_dim = X.shape[1]
        feature_mean, feature_std = compute_feature_stats(X_array)
        self.model = ScaledResidualRanker(
            self.in_dim,
            feature_mean=feature_mean,
            feature_std=feature_std,
        ).to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=NN_LEARNING_RATE,
            weight_decay=8e-5,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=1,
            min_lr=2.5e-5,
        )
        criterion = BPRLoss()
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        best_state = None
        best_ndcg = -float("inf")
        best_hit = 0.0
        best_epoch = 0
        stale_epochs = 0

        def validation_loss():
            if val_dl is None:
                return None
            self.model.eval()
            losses = []
            with torch.no_grad():
                for pos_idx, neg_idx in val_dl:
                    pos_x = X_val_cpu[pos_idx].to(self.device, non_blocking=True)
                    neg_x = X_val_cpu[neg_idx].to(self.device, non_blocking=True)
                    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                        scores = self.model(torch.cat([pos_x, neg_x], dim=0))
                        pos_score, neg_score = scores.chunk(2)
                        losses.append(criterion(pos_score, neg_score).item())
            self.model.train()
            return float(np.mean(losses))

        for ep in range(epochs):
            self.model.train()
            total_loss = 0
            for pos_idx, neg_idx in dl:
                pos_x = X_cpu[pos_idx].to(self.device, non_blocking=True)
                neg_x = X_cpu[neg_idx].to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    scores = self.model(torch.cat([pos_x, neg_x], dim=0))
                    pos_score, neg_score = scores.chunk(2)
                    loss = criterion(pos_score, neg_score)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
            train_loss = total_loss / len(dl)
            val_loss = validation_loss()
            val_ndcg = None
            val_hit = None
            if X_val is not None and val_meta is not None:
                val_meta["nn_score"] = self.predict(X_val)
                val_ndcg, val_hit = ndcg_hit_from_frame(val_meta, "nn_score")
                self.model.train()
            scheduler_metric = val_ndcg if val_ndcg is not None else -(
                val_loss if val_loss is not None else train_loss
            )
            scheduler.step(scheduler_metric)
            lr = optimizer.param_groups[0]["lr"]
            if val_ndcg is None:
                print(f"BPR epoch {ep+1}/{epochs}, loss={train_loss:.5f}, lr={lr:.7f}")
                continue

            print(
                f"BPR epoch {ep+1}/{epochs}, loss={train_loss:.5f}, "
                f"val_loss={val_loss:.5f}, Hit@10={val_hit:.4f}, "
                f"NDCG@10={val_ndcg:.4f}, lr={lr:.7f}"
            )
            if val_ndcg > best_ndcg + 1e-5:
                best_ndcg = val_ndcg
                best_hit = val_hit
                best_epoch = ep + 1
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= NN_EARLY_STOPPING:
                    print(
                        f"Early stopping at epoch {ep+1}; best epoch={best_epoch}, "
                        f"Hit@10={best_hit:.4f}, NDCG@10={best_ndcg:.4f}"
                    )
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
        return self

    def predict(self, X):
        self.model.eval()
        result = []
        X_array = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(X_array), 8192):
                X_batch = torch.from_numpy(X_array[i:i+8192]).to(self.device)
                result.append(self.model(X_batch).cpu().numpy())
        return np.concatenate(result)

    def save(self, path):
        if self.model is None or self.in_dim is None:
            raise RuntimeError("Cannot save an untrained model")
        torch.save({"in_dim": self.in_dim, "state_dict": self.model.state_dict()}, path)

    @classmethod
    def load(cls, path):
        wrapper = cls()
        checkpoint = torch.load(path, map_location=wrapper.device)
        wrapper.in_dim = int(checkpoint["in_dim"])
        wrapper.model = ScaledResidualRanker(wrapper.in_dim).to(wrapper.device)
        wrapper.model.load_state_dict(checkpoint["state_dict"])
        wrapper.model.eval()
        return wrapper

def train_rankers(oof):
    cols = feature_cols(oof)
    units = oof["uid"].drop_duplicates().to_numpy().copy()
    rng = np.random.default_rng(RANDOM_STATE + 19)
    rng.shuffle(units)
    nv = max(1, int(round(len(units) * META_VALID_FRAC)))
    valid_users = set(units[:nv])
    valid_mask = oof["uid"].isin(valid_users).to_numpy()

    yt = oof.loc[~valid_mask, "label"].to_numpy(dtype=np.float32)
    qidt = oof.loc[~valid_mask, "qid"].to_numpy()
    yv = oof.loc[valid_mask, "label"].to_numpy(dtype=np.float32)
    qidv = oof.loc[valid_mask, "qid"].to_numpy()
    ms = oof.loc[valid_mask, ["qid", "label", "positive_retrieved"]].copy()
    Xt = oof.loc[~valid_mask, cols].to_numpy(dtype=np.float32, copy=True)
    Xv = oof.loc[valid_mask, cols].to_numpy(dtype=np.float32, copy=True)
    np.nan_to_num(Xt, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(Xv, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    del oof, valid_mask, units, valid_users
    gc.collect()

    print("\nTraining T2 deeper residual ranker with direct NDCG early stopping...")
    nn_ranker = NNRankerWrapper()
    nn_ranker.fit(
        Xt,
        yt,
        qidt,
        X_val=Xv,
        y_val=yv,
        qid_val=qidv,
        val_meta=ms,
    )

    ms["nn_score"] = nn_ranker.predict(Xv)
    n, h = ndcg_hit_from_frame(ms, "nn_score")
    print(f"\nMeta-validation T2 Deeper NN+BPR: Hit@10={h:.4f} NDCG@10={n:.4f}")
    return nn_ranker, cols, None

# ============================================================
# Prediction
# ============================================================
def predict_test(ranker, cols, train, test, cat_cols, sample):
    print("\nBuilding test statistics...")
    stats = RecallStats(train, cat_cols)
    print("Predicting test (batch)...")
    preds = {}
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
        cf["nn_score"] = ranker.predict(X).astype(np.float32)
        for q, g in cf.groupby("qid", sort=False):
            uid = qmap[int(q)]
            ranked = g.sort_values("nn_score", ascending=False, kind="stable")["candidate_iid"].drop_duplicates().tolist()
            top = ranked[:10]
            if len(top) < 10:
                for iid in stats.global_top:
                    if iid not in top: top.append(iid)
                    if len(top) >= 10: break
            preds[uid] = top
        print(f" {stop}/{len(test)}")
        del cf, X, rows; gc.collect()
    return preds, stats

# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save_to", type=str, default=None)
    p.add_argument("--infer_only", action="store_true", help="Skip training")
    p.add_argument("--print_config", action="store_true", help="Print config")
    args, unknown = p.parse_known_args()

    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    config_snapshot = {
        "version": VERSION,
        "runtime": {
            "random_state": RANDOM_STATE,
            "n_folds": N_FOLDS,
            "candidates": MAX_CANDIDATES,
            "epochs": NN_EPOCHS,
            "batch_size": NN_BATCH_SIZE,
            "early_stopping": NN_EARLY_STOPPING,
            "use_amp": NN_USE_AMP,
            "learning_rate": NN_LEARNING_RATE,
            "reuse_oof_cache": REUSE_OOF_CACHE,
            "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
        },
        "model": {
            "residual_blocks": 3,
            "width": 448,
            "gated_interaction_dim": 72,
            "features": "pmi+npmi+cooccurrence_support+recency_interactions",
        },
    }
    if args.print_config:
        print(json.dumps(config_snapshot, ensure_ascii=False, indent=2))
        return

    print(f"=== {VERSION} ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading data...")
    train, test, user, sample = load_data()
    train, test, cat_cols = prepare_frames(train, test, user)
    print(f" Train={len(train)}, Test={len(test)}, Cats={len(cat_cols)}")

    if args.infer_only:
        if not os.path.exists(MODEL_PATH): raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        print(f"\nLoading model: {MODEL_PATH}")
        ranker = NNRankerWrapper.load(MODEL_PATH)
        with open(FEATURE_COLS_PATH) as f: cols = json.load(f)
        preds, fstats = predict_test(ranker, cols, train, test, cat_cols, sample)
    else:
        if REUSE_OOF_CACHE and os.path.exists(OOF_CACHE_PATH):
            print(f"\nLoading OOF cache: {OOF_CACHE_PATH}")
            oof = pd.read_parquet(OOF_CACHE_PATH)
        else:
            print(f"\nBuilding OOF ranking data ({N_FOLDS}-fold)...")
            oof, recalls = build_oof(train, cat_cols)
            print(f" OOF recall: {np.mean(recalls):.4f} +- {np.std(recalls):.4f}")
            print(f" Saving OOF cache: {OOF_CACHE_PATH}")
            oof.to_parquet(OOF_CACHE_PATH, index=False)
        print(f" OOF rows: {len(oof):,}")

        oof_box = [oof]
        del oof
        ranker, cols, lb = train_rankers(oof_box.pop())
        ranker.save(MODEL_PATH)
        with open(FEATURE_COLS_PATH, "w") as f:
            json.dump(cols, f, ensure_ascii=False, indent=2)
        print(f" Saved model: {MODEL_PATH}")
        preds, fstats = predict_test(ranker, cols, train, test, cat_cols, sample)

    base_dir = args.save_to or SCRIPT_DIR
    os.makedirs(base_dir, exist_ok=True)
    out = sample.copy()
    out["uid"] = out["uid"].map(clean_id)
    out["prediction"] = out["uid"].map(lambda uid: ",".join(preds.get(uid, fstats.global_top[:10])[:10]))
    path = os.path.join(base_dir, SUBMISSION_FILENAME)
    out.to_csv(path, index=False)
    print(f"Saved: {path}")

if __name__ == "__main__":
    main()
