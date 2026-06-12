"""NDCG-optimized recommendation for test distribution (short seq + cold-start)"""

import os
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))


def load_data():
    base = os.path.join(DATA_ROOT, "A推荐", "A推荐")
    train = pd.read_csv(os.path.join(base, "train.csv"))
    test = pd.read_csv(os.path.join(base, "test.csv"))
    user_df = pd.read_csv(os.path.join(base, "user.csv"))
    sample_sub = pd.read_csv(os.path.join(base, "sample_submission.csv"))
    return train, test, user_df, sample_sub


def parse_seq_counts(s):
    if pd.isna(s) or str(s).strip() == "":
        return {}
    result = {}
    for part in str(s).split(","):
        parts = part.split(":")
        if len(parts) == 2:
            result[parts[0]] = int(parts[1])
    return result


def seq_len(s):
    if pd.isna(s) or str(s).strip() == "" or str(s).strip() == "nan":
        return 0
    return len(str(s).split(","))


def build_user_feature_matrix(user_df):
    feat_cols = [c for c in user_df.columns if c.startswith("u_cat_")]
    feat_matrix = pd.get_dummies(user_df[feat_cols], columns=feat_cols).values.astype(np.float32)
    return feat_matrix, user_df["uid"].values


def build_user_item_preferences(train_df):
    user_prefs = {}
    for _, row in train_df.iterrows():
        uid = row["uid"]
        target = row["target_iid"]
        user_prefs.setdefault(uid, []).append(target)
    return user_prefs


def predict_warm(seq_raw, seq_dedup, item_counts, transitions, cooccur,
                 last_to_target, topk=10):
    scores = defaultdict(float)
    items_raw = str(seq_raw).strip().split(",") if pd.notna(seq_raw) and str(seq_raw).strip() else []
    items_dedup = str(seq_dedup).strip().split(",") if pd.notna(seq_dedup) and str(seq_dedup).strip() else []
    raw_set = set(items_raw)

    for iid, cnt in item_counts.items():
        scores[iid] += 0.001 * cnt

    l2t_items = set()
    if items_dedup:
        last_item = items_dedup[-1]
        if last_item in transitions:
            for next_item, cnt in transitions[last_item].items():
                scores[next_item] += 0.18 * cnt
        if last_item in last_to_target:
            for target, cnt in last_to_target[last_item].items():
                scores[target] += 2000.0 * cnt
                l2t_items.add(target)

    recent = items_dedup[-3:] if len(items_dedup) >= 3 else items_dedup
    for item in recent:
        if item in cooccur:
            for related, cnt in cooccur[item].items():
                scores[related] += 0.18 * cnt

    user_freq = Counter(items_raw)
    for iid, cnt in user_freq.items():
        scores[iid] += 1000.0 * cnt

    for item in l2t_items:
        mult = 50.0
        if item in raw_set:
            mult *= 4.5
        scores[item] *= mult

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [iid for iid, _ in ranked[:topk]]


def predict_cold(uid, user_feat_matrix, user_uids, user_item_prefs, target_dist, topk=10):
    from sklearn.metrics.pairwise import cosine_similarity
    user_idx = np.where(user_uids == uid)[0]
    if len(user_idx) == 0:
        return [iid for iid, _ in target_dist.most_common(topk)]
    user_idx = user_idx[0]
    user_vec = user_feat_matrix[user_idx:user_idx+1]
    sims = cosine_similarity(user_vec, user_feat_matrix)[0]
    sim_indices = np.argsort(sims)[::-1]
    item_scores = defaultdict(float)
    count = 0
    for idx in sim_indices:
        sim_uid = user_uids[idx]
        if sim_uid != uid and sim_uid in user_item_prefs:
            for target in user_item_prefs[sim_uid]:
                item_scores[target] += sims[idx]
            count += 1
            if count >= 50:
                break
    for iid, cnt in target_dist.most_common(20):
        if iid not in item_scores:
            item_scores[iid] += 0.001 * cnt
    ranked = sorted(item_scores.items(), key=lambda x: -x[1])
    top_items = [iid for iid, _ in ranked[:topk]]
    if len(top_items) < topk:
        for iid, _ in target_dist.most_common(topk):
            if iid not in top_items:
                top_items.append(iid)
            if len(top_items) >= topk:
                break
    return top_items


def ndcg_at_k(ranked_list, target_item, k=10):
    for i, item in enumerate(ranked_list[:k]):
        if item == target_item:
            return 1.0 / np.log2(i + 2)
    return 0.0


def run():
    print("Loading data...")
    train_df, test_df, user_df, sample_sub = load_data()

    print("Building statistics...")
    item_counts = Counter()
    for seq in train_df["item_seq_raw"].dropna():
        for item_id in str(seq).split(","):
            item_counts[item_id] += 1

    target_dist = Counter(train_df["target_iid"])

    transitions = defaultdict(Counter)
    for _, row in train_df.iterrows():
        seq = str(row["item_seq_raw"]).strip()
        if not seq or seq == "nan":
            continue
        items = seq.split(",")
        for i in range(len(items) - 1):
            transitions[items[i]][items[i + 1]] += 1

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

    last_to_target = defaultdict(Counter)
    for _, row in train_df.iterrows():
        target = row["target_iid"]
        seq_dedup = str(row["item_seq_dedup"]).strip()
        if not seq_dedup or seq_dedup == "nan":
            continue
        items = seq_dedup.split(",")
        last_to_target[items[-1]][target] += 1

    user_feat_matrix, user_uids = build_user_feature_matrix(user_df)
    user_item_prefs = build_user_item_preferences(train_df)

    COLD_THRESHOLD = 1  # sl=0 uses cold, sl>=1 uses warm

    # Evaluate on train (long seq)
    print("\nEvaluating NDCG@10 on train (long seq)...")
    ndcg_scores = []
    hit_scores = []
    for _, row in train_df.head(2000).iterrows():
        target = row["target_iid"]
        sl = seq_len(row["item_seq_raw"])
        if sl >= COLD_THRESHOLD:
            preds = predict_warm(row["item_seq_raw"], row["item_seq_dedup"],
                                 item_counts, transitions, cooccur, last_to_target)
        else:
            preds = predict_cold(row["uid"], user_feat_matrix, user_uids,
                                 user_item_prefs, target_dist)
        ndcg_scores.append(ndcg_at_k(preds, target))
        hit_scores.append(1.0 if target in preds else 0.0)
    print(f"  Hit@10:  {np.mean(hit_scores):.4f}")
    print(f"  NDCG@10: {np.mean(ndcg_scores):.4f}")

    # Generate test predictions
    print("\nGenerating test predictions...")
    predictions = {}
    cold_count = 0
    warm_count = 0
    for _, row in test_df.iterrows():
        uid = row["uid"]
        sl = seq_len(row["item_seq_raw"])
        if sl >= COLD_THRESHOLD:
            preds = predict_warm(row["item_seq_raw"], row["item_seq_dedup"],
                                 item_counts, transitions, cooccur, last_to_target)
            warm_count += 1
        else:
            preds = predict_cold(uid, user_feat_matrix, user_uids,
                                 user_item_prefs, target_dist)
            cold_count += 1
        predictions[uid] = preds

    print(f"  Cold-start: {cold_count}, Warm: {warm_count}")

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
