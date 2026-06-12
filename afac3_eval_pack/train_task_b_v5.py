"""
Task B V5: Item feature fusion + diversity re-ranking
Improvements over V4:
  - Item category similarity for cold-start recall
  - Diversity penalty in Top10 (avoid same-category saturation)
"""

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


def build_item_cat_map(item_df):
    """Build item -> category mapping"""
    item_cat = {}
    for _, row in item_df.iterrows():
        item_cat[row["iid"]] = (row["i_cat_01"], row["i_cat_02"], row["i_cat_03"])
    return item_cat


def build_cat_to_items(item_df):
    """Build category -> item list mapping (using i_cat_01 as primary)"""
    cat_items = defaultdict(list)
    for _, row in item_df.iterrows():
        cat_items[row["i_cat_01"]].append(row["iid"])
    return cat_items


def item_similarity(cat1, cat2):
    """Simple category overlap similarity"""
    if cat1 is None or cat2 is None:
        return 0.0
    score = 0.0
    if cat1[0] == cat2[0] and cat1[0] != 0:
        score += 1.0
    if cat1[1] == cat2[1] and cat1[1] != 0:
        score += 0.5
    if cat1[2] == cat2[2] and cat1[2] != 0:
        score += 0.3
    return score


def diversity_rerank(candidates, item_cat, topk=10, diversity_weight=0.3):
    """Re-rank candidates with diversity penalty for same-category items"""
    if len(candidates) <= topk:
        return candidates

    selected = []
    selected_cats = Counter()
    remaining = list(candidates)

    for _ in range(topk):
        if not remaining:
            break
        best_idx = 0
        best_score = -float('inf')
        for i, item in enumerate(remaining):
            cat = item_cat.get(item, (0, 0, 0))
            # Diversity penalty: more items of same category -> lower score
            penalty = diversity_weight * selected_cats[cat[0]]
            score = len(candidates) - i - penalty  # position-based score
            if score > best_score:
                best_score = score
                best_idx = i
        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        cat = item_cat.get(chosen, (0, 0, 0))
        selected_cats[cat[0]] += 1

    return selected


# ─────────────────────────────────────────────
# Warm user prediction (frequency-weighted)
# ─────────────────────────────────────────────

def predict_warm_user(seq_raw, seq_dedup, item_counts, transitions, cooccur,
                      last_to_target, freq_weight, topk=10):
    scores = defaultdict(float)

    items_raw = str(seq_raw).strip().split(",") if pd.notna(seq_raw) and str(seq_raw).strip() else []
    items_dedup = str(seq_dedup).strip().split(",") if pd.notna(seq_dedup) and str(seq_dedup).strip() else []

    for iid, cnt in item_counts.items():
        scores[iid] += 0.001 * cnt

    if items_dedup:
        last_item = items_dedup[-1]
        if last_item in transitions:
            for next_item, cnt in transitions[last_item].items():
                scores[next_item] += 0.5 * cnt
        if last_item in last_to_target:
            for target, cnt in last_to_target[last_item].items():
                scores[target] += 2.0 * cnt

    recent = items_dedup[-3:] if len(items_dedup) >= 3 else items_dedup
    for item in recent:
        if item in cooccur:
            for related, cnt in cooccur[item].items():
                scores[related] += 0.3 * cnt

    user_freq = Counter(items_raw)
    for iid, cnt in user_freq.items():
        scores[iid] += freq_weight * cnt

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [iid for iid, _ in ranked[:topk]]


# ─────────────────────────────────────────────
# Cold-start with item features
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
                      item_counts, item_cat, cat_to_items, topk=10):
    """Cold-start: user similarity + item category recall"""
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

    # Item category recall: items in same category as similar users' items
    seen_cats = set()
    for sim_uid, _ in top_sim_users:
        for item in user_item_prefs[sim_uid]:
            cat = item_cat.get(item, (0, 0, 0))
            if cat[0] != 0:
                seen_cats.add(cat[0])

    for cat in seen_cats:
        for item in cat_to_items.get(cat, []):
            if item not in item_scores:
                item_scores[item] += 0.1  # small boost for category recall

    for iid, cnt in item_counts.most_common(topk):
        if iid not in item_scores:
            item_scores[iid] += 0.001 * cnt

    ranked = sorted(item_scores.items(), key=lambda x: -x[1])
    top_items = [iid for iid, _ in ranked[:topk * 2]]  # get more candidates

    # Diversity re-ranking
    top_items = diversity_rerank(top_items, item_cat, topk)

    if len(top_items) < topk:
        for iid, _ in item_counts.most_common(topk):
            if iid not in top_items:
                top_items.append(iid)
            if len(top_items) >= topk:
                break

    return top_items


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def run():
    print("Loading data...")
    train_df, test_df, user_df, item_df, sample_sub = load_data()

    print("Building item features...")
    item_cat = build_item_cat_map(item_df)
    cat_to_items = build_cat_to_items(item_df)

    print("Building statistics...")
    item_counts = Counter()
    for seq in train_df["item_seq_raw"].dropna():
        for item_id in str(seq).split(","):
            item_counts[item_id] += 1

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

    print("Building user feature matrix...")
    user_feat_matrix, user_uids = build_user_feature_matrix(user_df)
    user_item_prefs = build_user_item_preferences(train_df)

    COLD_THRESHOLD = 3
    FREQ_WEIGHT = 1000.0  # from V4 optimization

    # Search diversity_weight
    print("\nSearching diversity_weight...")
    best_score = 0
    best_div = 0.0

    val_df = train_df.head(2000)

    for dw in [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]:
        hits = {1: 0, 5: 0, 10: 0}
        total = 0
        for _, row in val_df.iterrows():
            uid = row["uid"]
            target = row["target_iid"]
            sl = seq_len(row["item_seq_raw"])

            if sl >= COLD_THRESHOLD:
                preds = predict_warm_user(
                    row["item_seq_raw"], row["item_seq_dedup"],
                    item_counts, transitions, cooccur, last_to_target, FREQ_WEIGHT, topk=10
                )
            else:
                preds = predict_cold_user(
                    uid, user_feat_matrix, user_uids, user_item_prefs,
                    item_counts, item_cat, cat_to_items, topk=10
                )

            for k in [1, 5, 10]:
                if target in preds[:k]:
                    hits[k] += 1
            total += 1

        h10 = hits[10] / total
        print(f"  dw={dw:.1f}: Hit@1={hits[1]/total:.4f} Hit@5={hits[5]/total:.4f} Hit@10={h10:.4f}")
        if h10 > best_score:
            best_score = h10
            best_div = dw

    print(f"\nBest diversity_weight: {best_div} (Hit@10={best_score:.4f})")

    # Final evaluation
    print("\nFinal evaluation...")
    hits = {1: 0, 5: 0, 10: 0}
    total = 0
    for _, row in train_df.head(2000).iterrows():
        uid = row["uid"]
        target = row["target_iid"]
        sl = seq_len(row["item_seq_raw"])

        if sl >= COLD_THRESHOLD:
            preds = predict_warm_user(
                row["item_seq_raw"], row["item_seq_dedup"],
                item_counts, transitions, cooccur, last_to_target, FREQ_WEIGHT, topk=10
            )
        else:
            preds = predict_cold_user(
                uid, user_feat_matrix, user_uids, user_item_prefs,
                item_counts, item_cat, cat_to_items, topk=10
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
                item_counts, transitions, cooccur, last_to_target, FREQ_WEIGHT, topk=10
            )
            warm_count += 1
        else:
            preds = predict_cold_user(
                uid, user_feat_matrix, user_uids, user_item_prefs,
                item_counts, item_cat, cat_to_items, topk=10
            )
            cold_count += 1

        predictions[uid] = preds

    print(f"  Warm: {warm_count}, Cold: {cold_count}")

    rows = []
    for _, row in sample_sub.iterrows():
        uid = row["uid"]
        preds = predictions.get(uid, list(item_counts.keys())[:10])
        rows.append({"uid": uid, "prediction": ",".join(preds)})

    sub = pd.DataFrame(rows)
    out_path = os.path.join(DATA_ROOT, "A2.csv")
    sub.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"Rows: {len(sub)}")


if __name__ == "__main__":
    run()
