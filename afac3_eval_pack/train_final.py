"""Final hybrid: XGBRanker for sl=1-2, Heuristic for sl=0/3+"""
import os
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))


def load_data():
    base = os.path.join(DATA_ROOT, "A推荐", "A推荐")
    train = pd.read_csv(os.path.join(base, "train.csv"))
    test = pd.read_csv(os.path.join(base, "test.csv"))
    user_df = pd.read_csv(os.path.join(base, "user.csv"))
    item_df = pd.read_csv(os.path.join(base, "item.csv"))
    sample_sub = pd.read_csv(os.path.join(base, "sample_submission.csv"))
    return train, test, user_df, item_df, sample_sub


def seq_len(s):
    if pd.isna(s) or str(s).strip() == "" or str(s).strip() == "nan":
        return 0
    return len(str(s).split(","))


def ndcg_at_k(ranked_list, target_item, k=10):
    for i, item in enumerate(ranked_list[:k]):
        if item == target_item:
            return 1.0 / np.log2(i + 2)
    return 0.0


def build_item_features(item_df):
    item_feats = {}
    for _, row in item_df.iterrows():
        item_feats[row["iid"]] = {
            "i_cat_01": row["i_cat_01"],
            "i_cat_02": row["i_cat_02"],
            "i_cat_03": row["i_cat_03"],
            "i_bucket_01": row["i_bucket_01"],
        }
    return item_feats


def build_user_features(user_df):
    user_feats = {}
    for _, row in user_df.iterrows():
        feats = {}
        for c in user_df.columns:
            if c.startswith("u_cat_"):
                feats[c] = row[c]
        user_feats[row["uid"]] = feats
    return user_feats


def extract_features(uid, iid, u_feats, item_feats, item_counts, transitions,
                     last_to_target, item_cat_map, items_raw, items_dedup,
                     raw_set, last_item, hist_len, hist_unique,
                     pair_transitions=None, last2_to_target=None, prev_item=None):
    i_feats = item_feats.get(iid, {"i_cat_01": 0, "i_cat_02": 0, "i_cat_03": 0, "i_bucket_01": 0})
    u_cat_01 = u_feats.get("u_cat_01", 0)
    u_cat_02 = u_feats.get("u_cat_02", 0)
    u_cat_03 = u_feats.get("u_cat_03", 0)
    u_cat_04 = u_feats.get("u_cat_04", 0)
    u_cat_05 = u_feats.get("u_cat_05", 0)
    u_cat_06 = u_feats.get("u_cat_06", 0)
    u_cat_07 = u_feats.get("u_cat_07", 0)
    u_cat_08 = u_feats.get("u_cat_08", 0)
    i_cat_01 = i_feats["i_cat_01"]
    i_cat_02 = i_feats["i_cat_02"]
    i_cat_03 = i_feats["i_cat_03"]
    i_bucket_01 = i_feats["i_bucket_01"]
    item_pop = item_counts.get(iid, 0)
    log_pop = np.log1p(item_pop)
    in_history = 1 if iid in raw_set else 0
    is_last = 1 if iid == last_item else 0
    l2t_score = 0
    if last_item and last_item in last_to_target:
        l2t_score = last_to_target[last_item].get(iid, 0)
    trans_score = 0
    if last_item and last_item in transitions:
        trans_score = transitions[last_item].get(iid, 0)
    pair_trans_score = 0
    pair_l2t_score = 0
    if prev_item and last_item and pair_transitions:
        pair = (prev_item, last_item)
        if pair in pair_transitions:
            pair_trans_score = pair_transitions[pair].get(iid, 0)
        if last2_to_target and pair in last2_to_target:
            pair_l2t_score = last2_to_target[pair].get(iid, 0)
    last_cat = item_cat_map.get(last_item, (0, 0, 0)) if last_item else (0, 0, 0)
    cand_cat = (i_cat_01, i_cat_02, i_cat_03)
    cat_match = sum(1 for a, b in zip(cand_cat, last_cat) if a == b) / 3.0
    hist_cat_match = 0
    if items_dedup:
        for item in items_dedup[-3:]:
            item_cat = item_cat_map.get(item, (0, 0, 0))
            if item_cat == cand_cat:
                hist_cat_match = 1
                break
    cooccur_score = 0
    for item in items_dedup[-3:]:
        if item in transitions:
            cooccur_score += transitions[item].get(iid, 0)
    return [
        hist_len, hist_unique,
        u_cat_01, u_cat_02, u_cat_03, u_cat_04, u_cat_05, u_cat_06, u_cat_07, u_cat_08,
        i_cat_01, i_cat_02, i_cat_03, i_bucket_01,
        log_pop, in_history, is_last,
        l2t_score, trans_score, cat_match, hist_cat_match, cooccur_score,
        pair_trans_score, pair_l2t_score,
    ]


def predict_ranker(row, model, item_feats, user_feats, item_counts, transitions,
                   last_to_target, item_cat_map, target_dist,
                   pair_transitions=None, last2_to_target=None, topk=10):
    uid = row["uid"]
    seq_raw = str(row["item_seq_raw"]).strip()
    seq_dedup = str(row["item_seq_dedup"]).strip()
    if not seq_raw or seq_raw == "nan":
        return [iid for iid, _ in target_dist.most_common(topk)]
    items_raw = seq_raw.split(",")
    items_dedup = seq_dedup.split(",") if seq_dedup and seq_dedup != "nan" else []
    raw_set = set(items_raw)
    last_item = items_dedup[-1] if items_dedup else None
    u_feats = user_feats.get(uid, {})
    hist_len = len(items_raw)
    hist_unique = len(set(items_raw))
    candidates = set()
    for iid, _ in target_dist.most_common(20):
        candidates.add(iid)
    if last_item and last_item in last_to_target:
        for iid, _ in last_to_target[last_item].most_common(30):
            candidates.add(iid)
    if last_item and last_item in transitions:
        for iid, _ in transitions[last_item].most_common(15):
            candidates.add(iid)
    for item in items_raw:
        candidates.add(item)
    X_cand = []
    cand_list = []
    prev_item = items_dedup[-2] if len(items_dedup) >= 2 else None
    for iid in candidates:
        feat = extract_features(uid, iid, u_feats, item_feats, item_counts,
                                transitions, last_to_target, item_cat_map,
                                items_raw, items_dedup, raw_set, last_item, hist_len, hist_unique,
                                pair_transitions, last2_to_target, prev_item)
        X_cand.append(feat)
        cand_list.append(iid)
    if not X_cand:
        return [iid for iid, _ in target_dist.most_common(topk)]
    X_cand = np.array(X_cand)
    scores = model.predict(X_cand)
    ranked_indices = np.argsort(scores)[::-1]
    preds = [cand_list[i] for i in ranked_indices[:topk]]
    if len(preds) < topk:
        for iid, _ in target_dist.most_common(topk * 2):
            if iid not in preds:
                preds.append(iid)
            if len(preds) >= topk:
                break
    return preds[:topk]


def run():
    print("Loading data...")
    train_df, test_df, user_df, item_df, sample_sub = load_data()

    item_feats = build_item_features(item_df)
    user_feats = build_user_features(user_df)

    item_counts = Counter()
    for seq in train_df["item_seq_raw"].dropna():
        for item_id in str(seq).split(","):
            item_counts[item_id] += 1

    transitions = defaultdict(Counter)
    pair_transitions = defaultdict(Counter)
    for _, row in train_df.iterrows():
        seq = str(row["item_seq_raw"]).strip()
        if not seq or seq == "nan":
            continue
        items = seq.split(",")
        for i in range(len(items) - 1):
            transitions[items[i]][items[i + 1]] += 1
        for i in range(len(items) - 2):
            pair_transitions[(items[i], items[i + 1])][items[i + 2]] += 1

    last_to_target = defaultdict(Counter)
    last2_to_target = defaultdict(Counter)
    for _, row in train_df.iterrows():
        target = row["target_iid"]
        seq_dedup = str(row["item_seq_dedup"]).strip()
        if not seq_dedup or seq_dedup == "nan":
            continue
        items = seq_dedup.split(",")
        last_to_target[items[-1]][target] += 1
        if len(items) >= 2:
            last2_to_target[(items[-2], items[-1])][target] += 1

    item_cat_map = {}
    for _, row in item_df.iterrows():
        item_cat_map[row["iid"]] = (row["i_cat_01"], row["i_cat_02"], row["i_cat_03"])

    target_dist = Counter(train_df["target_iid"])

    cooccur = defaultdict(Counter)
    for _, row in train_df.iterrows():
        seq = str(row["item_seq_raw"]).strip()
        if not seq or seq == "nan":
            continue
        items = list(set(seq.split(",")))
        for i in range(len(items)):
            for j in range(len(items)):
                if i != j:
                    cooccur[items[i]][items[j]] += 1

    # KMeans for cold-start
    print("Building KMeans clusters (K=30)...")
    feat_cols = [c for c in user_df.columns if c.startswith("u_cat_")]
    X = user_df[feat_cols].values.astype(np.float32)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=30, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    uid_to_cluster = {}
    for i in range(len(user_df)):
        uid_to_cluster[user_df.iloc[i]["uid"]] = clusters[i]
    cluster_targets = defaultdict(Counter)
    for _, row in train_df.iterrows():
        uid = row["uid"]
        target = row["target_iid"]
        if uid in uid_to_cluster:
            cluster_targets[uid_to_cluster[uid]][target] += 1
    cluster_recs = {}
    for c in range(30):
        cluster_recs[c] = [iid for iid, _ in cluster_targets[c].most_common(10)]

    # Build XGBRanker training data
    print("Building XGBRanker training data...")
    np.random.seed(42)
    all_items = list(item_counts.keys())
    X_rows = []
    y_labels = []
    group_sizes = []
    neg_per_pos = 19

    for _, row in train_df.iterrows():
        uid = row["uid"]
        target = row["target_iid"]
        seq_raw = str(row["item_seq_raw"]).strip()
        seq_dedup = str(row["item_seq_dedup"]).strip()
        if not seq_raw or seq_raw == "nan":
            continue
        items_raw = seq_raw.split(",")
        items_dedup = seq_dedup.split(",") if seq_dedup and seq_dedup != "nan" else []
        raw_set = set(items_raw)
        last_item = items_dedup[-1] if items_dedup else None
        prev_item = items_dedup[-2] if len(items_dedup) >= 2 else None
        u_feats = user_feats.get(uid, {})
        hist_len = len(items_raw)
        hist_unique = len(set(items_raw))

        feat = extract_features(uid, target, u_feats, item_feats, item_counts,
                                transitions, last_to_target, item_cat_map,
                                items_raw, items_dedup, raw_set, last_item, hist_len, hist_unique,
                                pair_transitions, last2_to_target, prev_item)
        X_rows.append(feat)
        y_labels.append(1)

        neg_count = 0
        attempts = 0
        while neg_count < neg_per_pos and attempts < neg_per_pos * 3:
            neg_item = all_items[np.random.randint(len(all_items))]
            if neg_item != target and neg_item not in raw_set:
                feat = extract_features(uid, neg_item, u_feats, item_feats, item_counts,
                                        transitions, last_to_target, item_cat_map,
                                        items_raw, items_dedup, raw_set, last_item, hist_len, hist_unique,
                                        pair_transitions, last2_to_target, prev_item)
                X_rows.append(feat)
                y_labels.append(0)
                neg_count += 1
            attempts += 1
        group_sizes.append(1 + neg_count)

    X_train = np.array(X_rows)
    y_train = np.array(y_labels)
    print(f"  Samples: {len(X_train)}, Groups: {len(group_sizes)}")

    print("Training XGBRanker (LambdaMART)...")
    model = xgb.XGBRanker(
        max_depth=24, n_estimators=200, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        objective="rank:ndcg", eval_metric="ndcg@10",
        lambdarank_num_pair_per_sample=10,
    )
    model.fit(X_train, y_train, group=group_sizes)

    # Heuristic warm predictor
    def predict_warm(seq_raw, seq_dedup, topk=10):
        scores = defaultdict(float)
        items_raw = str(seq_raw).strip().split(",") if pd.notna(seq_raw) and str(seq_raw).strip() else []
        items_dedup = str(seq_dedup).strip().split(",") if pd.notna(seq_dedup) and str(seq_dedup).strip() else []
        raw_set = set(items_raw)
        for iid, cnt in item_counts.items():
            scores[iid] += 0.001 * cnt
        l2t_items = set()
        if items_dedup:
            last_item = items_dedup[-1]
            prev_item = items_dedup[-2] if len(items_dedup) >= 2 else None
            if last_item in transitions:
                for next_item, cnt in transitions[last_item].items():
                    scores[next_item] += 0.15 * cnt
            # Pair transition signal
            if prev_item and last_item:
                pair = (prev_item, last_item)
                if pair in pair_transitions:
                    for nxt, cnt in pair_transitions[pair].items():
                        scores[nxt] += 5.0 * cnt
            if last_item in last_to_target:
                for target, cnt in last_to_target[last_item].items():
                    scores[target] += 3000.0 * cnt
                    l2t_items.add(target)
            # Pair l2t signal
            if prev_item and last_item:
                pair = (prev_item, last_item)
                if pair in last2_to_target:
                    for target, cnt in last2_to_target[pair].items():
                        scores[target] += 8000.0 * cnt
                        l2t_items.add(target)
        recent = items_dedup[-3:] if len(items_dedup) >= 3 else items_dedup
        for item in recent:
            if item in cooccur:
                for related, cnt in cooccur[item].items():
                    scores[related] += 0.15 * cnt
        user_freq = Counter(items_raw)
        for iid, cnt in user_freq.items():
            scores[iid] += 500.0 * cnt
        for item in l2t_items:
            mult = 100.0
            if item in raw_set:
                mult *= 3.0
            scores[item] *= mult
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [iid for iid, _ in ranked[:topk]]

    def predict_cold(uid, topk=10):
        cluster_id = uid_to_cluster.get(uid, 0)
        rec = list(cluster_recs.get(cluster_id, []))
        if len(rec) < topk:
            for iid, _ in target_dist.most_common(topk):
                if iid not in rec:
                    rec.append(iid)
                if len(rec) >= topk:
                    break
        return rec[:topk]

    def predict_hybrid(row, topk=10):
        uid = row["uid"]
        sl = seq_len(row["item_seq_raw"])
        if sl == 0:
            return predict_cold(uid, topk)
        else:
            return predict_ranker(row, model, item_feats, user_feats, item_counts,
                                  transitions, last_to_target, item_cat_map, target_dist,
                                  pair_transitions, last2_to_target, topk)

    # Validate
    print("\n=== Validation ===")
    for sl in [0, 1, 2, 3]:
        if sl == 0:
            val_sub = train_df[train_df["item_seq_raw"].apply(seq_len) == 0]
        else:
            val_sub = train_df[train_df["item_seq_raw"].apply(seq_len) == sl]
        if len(val_sub) == 0:
            continue
        val_sub = val_sub.tail(min(500, len(val_sub)))
        ndcg_scores = []
        hit_scores = []
        for _, row in val_sub.iterrows():
            preds = predict_hybrid(row)
            ndcg_scores.append(ndcg_at_k(preds, row["target_iid"]))
            hit_scores.append(1.0 if row["target_iid"] in preds else 0.0)
        print(f"  sl={sl}: NDCG={np.mean(ndcg_scores):.4f}, Hit={np.mean(hit_scores):.4f} (n={len(val_sub)})")

    # Generate test predictions
    print("\n=== Generating Test Predictions ===")
    predictions = {}
    for _, row in test_df.iterrows():
        preds = predict_hybrid(row)
        predictions[row["uid"]] = preds

    rows = []
    for _, row in sample_sub.iterrows():
        uid = row["uid"]
        preds = predictions.get(uid, list(item_counts.keys())[:10])
        rows.append({"uid": uid, "prediction": ",".join(preds)})

    sub = pd.DataFrame(rows)
    out_path = os.path.join(DATA_ROOT, "A2.csv")
    sub.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    run()
