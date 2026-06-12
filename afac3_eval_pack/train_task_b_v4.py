"""
Task B V4: Frequency-weighted recommendation + cold-start enhancement
Key improvements over V3:
  - Markov transitions weighted by count (not 0/1)
  - Co-occurrence weighted by frequency
  - User frequency signal strengthened
"""

import os
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))


def load_data():
    base = os.path.join(DATA_ROOT, "A推荐", "A推荐")
    train = pd.read_csv(os.path.join(base, "train.csv"))
    test = pd.read_csv(os.path.join(base, "test.csv"))
    user_df = pd.read_csv(os.path.join(base, "user.csv"))
    item_df = pd.read_csv(os.path.join(base, "item.csv"))
    sample_sub = pd.read_csv(os.path.join(base, "sample_submission.csv"))
    return train, test, user_df, item_df, sample_sub


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


# ─────────────────────────────────────────────
# Warm user recommendation (frequency-weighted)
# ─────────────────────────────────────────────

def predict_warm_user(seq_raw, seq_dedup, item_counts, transitions, cooccur,
                      last_to_target, freq_weight, topk=10):
    """For users with history: Markov transitions + co-occurrence + frequency, all weighted by count"""
    scores = defaultdict(float)

    items_raw = str(seq_raw).strip().split(",") if pd.notna(seq_raw) and str(seq_raw).strip() else []
    items_dedup = str(seq_dedup).strip().split(",") if pd.notna(seq_dedup) and str(seq_dedup).strip() else []

    # 1. Global popularity (low weight)
    for iid, cnt in item_counts.items():
        scores[iid] += 0.001 * cnt

    # 2. Markov transitions (weighted by user's item frequency)
    if items_dedup:
        last_item = items_dedup[-1]
        if last_item in transitions:
            for next_item, cnt in transitions[last_item].items():
                scores[next_item] += 0.5 * cnt
        # 3. Last item to target direct mapping
        if last_item in last_to_target:
            for target, cnt in last_to_target[last_item].items():
                scores[target] += 2.0 * cnt

    # 4. Co-occurrence (recent 3 unique items, weighted by frequency)
    recent = items_dedup[-3:] if len(items_dedup) >= 3 else items_dedup
    for item in recent:
        if item in cooccur:
            for related, cnt in cooccur[item].items():
                scores[related] += 0.3 * cnt

    # 5. User frequency (strengthened with frequency weight)
    user_freq = Counter(items_raw)
    for iid, cnt in user_freq.items():
        # Frequency-weighted: higher count = higher confidence
        scores[iid] += freq_weight * cnt

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [iid for iid, _ in ranked[:topk]]


# ─────────────────────────────────────────────
# Cold-start user recommendation (user feature similarity)
# ─────────────────────────────────────────────

def build_user_feature_matrix(user_df):
    feat_cols = [c for c in user_df.columns if c.startswith("u_cat_")]
    feat_matrix = pd.get_dummies(user_df[feat_cols], columns=feat_cols).values.astype(np.float32)
    return feat_matrix, user_df["uid"].values


def build_user_item_preferences(train_df):
    user_prefs = {}
    for _, row in train_df.iterrows():
        uid = row["uid"]
        target = row["target_iid"]
        items = [target]
        counts = parse_seq_counts(row["item_seq_counts"])
        if counts:
            top_items = sorted(counts.items(), key=lambda x: -x[1])[:5]
            items.extend([iid for iid, _ in top_items])
        user_prefs[uid] = items
    return user_prefs


def predict_cold_user(uid, user_feat_matrix, user_uids, user_item_prefs,
                      item_counts, topk=10):
    """For cold-start users: cosine similarity on user features"""
    from sklearn.metrics.pairwise import cosine_similarity

    user_idx = np.where(user_uids == uid)[0]
    if len(user_idx) == 0:
        return [iid for iid, _ in item_counts.most_common(topk)]
    user_idx = user_idx[0]

    user_vec = user_feat_matrix[user_idx:user_idx+1]
    sims = cosine_similarity(user_vec, user_feat_matrix)[0]

    sim_indices = np.argsort(sims)[::-1]
    top_sim_users = []
    for idx in sim_indices:
        sim_uid = user_uids[idx]
        if sim_uid != uid and sim_uid in user_item_prefs:
            top_sim_users.append((sim_uid, sims[idx]))
        if len(top_sim_users) >= 20:
            break

    item_scores = defaultdict(float)
    for sim_uid, sim_score in top_sim_users:
        for item in user_item_prefs[sim_uid]:
            item_scores[item] += sim_score

    for iid, cnt in item_counts.most_common(topk):
        if iid not in item_scores:
            item_scores[iid] += 0.001 * cnt

    ranked = sorted(item_scores.items(), key=lambda x: -x[1])
    top_items = [iid for iid, _ in ranked[:topk]]

    if len(top_items) < topk:
        for iid, _ in item_counts.most_common(topk):
            if iid not in top_items:
                top_items.append(iid)
            if len(top_items) >= topk:
                break

    return top_items


# ─────────────────────────────────────────────
# Build frequency-weighted statistics
# ─────────────────────────────────────────────

def build_stats(train_df):
    """Build Markov transitions and co-occurrence with frequency weighting"""
    # Global item counts
    item_counts = Counter()
    for seq in train_df["item_seq_raw"].dropna():
        for item_id in str(seq).split(","):
            item_counts[item_id] += 1

    # Markov transitions (count-weighted)
    transitions = defaultdict(Counter)
    for _, row in train_df.iterrows():
        seq = str(row["item_seq_raw"]).strip()
        if not seq or seq == "nan":
            continue
        items = seq.split(",")
        for i in range(len(items) - 1):
            transitions[items[i]][items[i + 1]] += 1

    # Co-occurrence (simple count-based)
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

    # Last item to target mapping
    last_to_target = defaultdict(Counter)
    for _, row in train_df.iterrows():
        target = row["target_iid"]
        seq_dedup = str(row["item_seq_dedup"]).strip()
        if not seq_dedup or seq_dedup == "nan":
            continue
        items = seq_dedup.split(",")
        last_to_target[items[-1]][target] += 1

    return item_counts, transitions, cooccur, last_to_target


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def run():
    print("Loading data...")
    train_df, test_df, user_df, item_df, sample_sub = load_data()

    print("Building statistics...")
    item_counts, transitions, cooccur, last_to_target = build_stats(train_df)

    print("Building user feature matrix...")
    user_feat_matrix, user_uids = build_user_feature_matrix(user_df)
    user_item_prefs = build_user_item_preferences(train_df)

    COLD_THRESHOLD = 3

    # Grid search for best freq_weight on validation set
    print("\nSearching best freq_weight...")
    best_score = 0
    best_weight = 0.2

    # Use first 2000 samples for validation
    val_df = train_df.head(2000)

    for fw in [1000.0]:
        hits = {1: 0, 5: 0, 10: 0}
        total = 0
        for _, row in val_df.iterrows():
            uid = row["uid"]
            target = row["target_iid"]
            sl = seq_len(row["item_seq_raw"])

            if sl >= COLD_THRESHOLD:
                preds = predict_warm_user(
                    row["item_seq_raw"], row["item_seq_dedup"],
                    item_counts, transitions, cooccur, last_to_target, fw, topk=10
                )
            else:
                preds = predict_cold_user(
                    uid, user_feat_matrix, user_uids, user_item_prefs,
                    item_counts, topk=10
                )

            for k in [1, 5, 10]:
                if target in preds[:k]:
                    hits[k] += 1
            total += 1

        h10 = hits[10] / total
        print(f"  fw={fw:.2f}: Hit@1={hits[1]/total:.4f} Hit@5={hits[5]/total:.4f} Hit@10={h10:.4f}")
        if h10 > best_score:
            best_score = h10
            best_weight = fw

    print(f"\nBest freq_weight: {best_weight} (Hit@10={best_score:.4f})")

    # Full evaluation with best weight
    print("\nFinal evaluation with best weight...")
    hits = {1: 0, 5: 0, 10: 0}
    total = 0
    for _, row in train_df.head(2000).iterrows():
        uid = row["uid"]
        target = row["target_iid"]
        sl = seq_len(row["item_seq_raw"])

        if sl >= COLD_THRESHOLD:
            preds = predict_warm_user(
                row["item_seq_raw"], row["item_seq_dedup"],
                item_counts, transitions, cooccur, last_to_target, best_weight, topk=10
            )
        else:
            preds = predict_cold_user(
                uid, user_feat_matrix, user_uids, user_item_prefs,
                item_counts, topk=10
            )

        for k in [1, 5, 10]:
            if target in preds[:k]:
                hits[k] += 1
        total += 1

    print(f"  Hit@1:  {hits[1]/total:.4f}")
    print(f"  Hit@5:  {hits[5]/total:.4f}")
    print(f"  Hit@10: {hits[10]/total:.4f}")

    # Generate test predictions
    print("\nGenerating test predictions...")
    predictions = {}
    cold_count = 0
    warm_count = 0
    for _, row in test_df.iterrows():
        uid = row["uid"]
        sl = seq_len(row["item_seq_raw"])

        if sl >= COLD_THRESHOLD:
            preds = predict_warm_user(
                row["item_seq_raw"], row["item_seq_dedup"],
                item_counts, transitions, cooccur, last_to_target, best_weight, topk=10
            )
            warm_count += 1
        else:
            preds = predict_cold_user(
                uid, user_feat_matrix, user_uids, user_item_prefs,
                item_counts, topk=10
            )
            cold_count += 1

        predictions[uid] = preds

    print(f"  Warm users: {warm_count}, Cold users: {cold_count}")

    # Save submission
    rows = []
    for _, row in sample_sub.iterrows():
        uid = row["uid"]
        preds = predictions.get(uid, list(item_counts.keys())[:10])
        rows.append({"uid": uid, "prediction": ",".join(preds)})

    sub = pd.DataFrame(rows)
    out_path = os.path.join(DATA_ROOT, "A2.csv")
    sub.to_csv(out_path, index=False)
    print(f"\nSubmission saved: {out_path}")
    print(f"Rows: {len(sub)}")
    print(f"Sample: {sub.iloc[0]['prediction'][:80]}...")


if __name__ == "__main__":
    run()
