"""
AFAC2026 赛题三 — 数据探索与质量报告
任务A: 图节点分类 (A分类/A1.npz)
任务B: 序列推荐 (A推荐/)
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from collections import Counter

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# 1. 图节点分类任务 (A分类)
# ─────────────────────────────────────────────

def explore_classification():
    print("=" * 70)
    print("  任务A: 图节点分类 (A分类/A1.npz)")
    print("=" * 70)

    npz_path = os.path.join(DATA_ROOT, "A分类", "A分类", "A1.npz")
    data = np.load(npz_path, allow_pickle=True)

    # 列出所有变量
    print(f"\n[npz 变量] {list(data.keys())}")

    # 还原邻接矩阵
    adj = csr_matrix(
        (data["adj_data"], data["adj_indices"], data["adj_indptr"]),
        shape=tuple(data["adj_shape"]),
    )
    # 还原特征矩阵
    features = csr_matrix(
        (data["attr_data"], data["attr_indices"], data["attr_indptr"]),
        shape=tuple(data["attr_shape"]),
    )
    labels = data["labels"]
    train_idx = data["train_idx"]
    test_idx = data["test_idx"]

    # 基本维度
    num_nodes, num_features = features.shape
    num_edges = adj.nnz
    num_classes = len(set(labels[labels >= 0]))
    num_train = len(train_idx)
    num_test = len(test_idx)

    print(f"\n── 基本信息 ──")
    print(f"  节点数 (N):        {num_nodes}")
    print(f"  特征维度 (F):      {num_features}")
    print(f"  边数 (nnz):        {num_edges}")
    print(f"  类别数:            {num_classes}")
    print(f"  训练节点数:        {num_train}")
    print(f"  测试节点数:        {num_test}")

    # 图统计
    density = num_edges / (num_nodes * num_nodes)
    avg_degree = num_edges / num_nodes
    # 计算每个节点的度
    indptr = data["adj_indptr"]
    degrees = np.diff(indptr)

    print(f"\n── 图结构统计 ──")
    print(f"  图密度:            {density:.6e}")
    print(f"  平均度:            {avg_degree:.2f}")
    print(f"  最大度:            {np.max(degrees)}")
    print(f"  最小度:            {np.min(degrees)}")
    print(f"  度标准差:          {np.std(degrees):.2f}")
    print(f"  中位数度:          {np.median(degrees):.0f}")

    # 检查是否对称 (无向)
    # 抽样检查: 取前100个节点
    is_symmetric = True
    check_n = min(200, num_nodes)
    adj_dense_sample = adj[:check_n, :check_n].toarray()
    if not np.allclose(adj_dense_sample, adj_dense_sample.T):
        is_symmetric = False
    # 更严格的全量检查 (使用转置)
    adj_T = adj.T
    diff = adj - adj_T
    if diff.nnz > 0:
        is_symmetric = False
    print(f"  是否对称 (无向):   {'是' if is_symmetric else '否'}")

    # 检查自环
    diag = adj.diagonal()
    num_self_loops = np.count_nonzero(diag)
    print(f"  自环数:            {num_self_loops}")

    # 边权统计
    edge_weights = data["adj_data"]
    print(f"  边权均值:          {np.mean(edge_weights):.4f}")
    print(f"  边权标准差:        {np.std(edge_weights):.4f}")
    print(f"  边权唯一值数:      {len(np.unique(edge_weights))}")

    # 特征矩阵统计
    print(f"\n── 特征矩阵统计 ──")
    attr_nnz = features.nnz
    attr_density = attr_nnz / (num_nodes * num_features)
    print(f"  非零元素数:        {attr_nnz}")
    print(f"  特征稀疏度:        {1 - attr_density:.4f} ({(1-attr_density)*100:.2f}% 为零)")
    print(f"  特征密度:          {attr_density:.6e}")
    print(f"  每节点平均非零特征: {attr_nnz / num_nodes:.2f}")

    # 特征值统计
    attr_vals = data["attr_data"]
    print(f"  特征值均值:        {np.mean(attr_vals):.6f}")
    print(f"  特征值标准差:      {np.std(attr_vals):.6f}")
    print(f"  特征值最小值:      {np.min(attr_vals):.6f}")
    print(f"  特征值最大值:      {np.max(attr_vals):.6f}")

    # 每节点特征数分布
    attr_indptr = data["attr_indptr"]
    nnz_per_node = np.diff(attr_indptr)
    print(f"  每节点特征数 - 均值: {np.mean(nnz_per_node):.2f}, "
          f"std: {np.std(nnz_per_node):.2f}, "
          f"min: {np.min(nnz_per_node)}, max: {np.max(nnz_per_node)}")

    # 标签分布
    print(f"\n── 标签分布 (训练集) ──")
    train_labels = labels[train_idx]
    label_counts = Counter(train_labels)
    for c in sorted(label_counts.keys()):
        cnt = label_counts[c]
        pct = cnt / num_train * 100
        print(f"  类别 {c}: {cnt:6d} ({pct:5.2f}%)")

    # 检查类别平衡
    max_count = max(label_counts.values())
    min_count = min(label_counts.values())
    imbalance_ratio = max_count / min_count
    print(f"  类别不平衡比:      {imbalance_ratio:.2f} (最大/最小)")

    # 测试标签检查
    test_labels = labels[test_idx]
    print(f"\n── 测试集标签 ──")
    print(f"  测试标签全部为-1:  {np.all(test_labels == -1)}")

    # 检查 train/test 节点是否有重叠
    overlap = set(train_idx) & set(test_idx)
    print(f"\n── 数据完整性 ──")
    print(f"  train/test 重叠:   {len(overlap)} 节点")
    print(f"  train ∪ test 覆盖: {num_train + num_test} / {num_nodes} 节点")
    print(f"  覆盖率:            {(num_train + num_test) / num_nodes * 100:.2f}%")

    # 检查孤立节点
    isolated = np.where(degrees == 0)[0]
    print(f"  孤立节点数:        {len(isolated)}")
    if len(isolated) > 0:
        isolated_in_train = sum(1 for n in isolated if n in set(train_idx))
        isolated_in_test = sum(1 for n in isolated if n in set(test_idx))
        print(f"    其中训练集:      {isolated_in_train}")
        print(f"    其中测试集:      {isolated_in_test}")

    # 数据类型检查
    print(f"\n── 数据类型 ──")
    print(f"  adj_data dtype:    {data['adj_data'].dtype}")
    print(f"  adj_indices dtype: {data['adj_indices'].dtype}")
    print(f"  attr_data dtype:   {data['attr_data'].dtype}")
    print(f"  labels dtype:      {labels.dtype}")
    print(f"  train_idx dtype:   {train_idx.dtype}")

    # NaN / Inf 检查
    has_nan_attr = np.any(np.isnan(attr_vals))
    has_inf_attr = np.any(np.isinf(attr_vals))
    print(f"\n── 数据质量 ──")
    print(f"  特征含 NaN:        {has_nan_attr}")
    print(f"  特征含 Inf:        {has_inf_attr}")

    # 提交模板检查
    sample_sub_path = os.path.join(DATA_ROOT, "A分类", "A分类", "sample_submission.csv")
    sample_sub = pd.read_csv(sample_sub_path)
    print(f"\n── 提交模板 ──")
    print(f"  行数:              {len(sample_sub)}")
    print(f"  列:                {list(sample_sub.columns)}")
    print(f"  test_idx 范围:     [{sample_sub['test_idx'].min()}, {sample_sub['test_idx'].max()}]")
    print(f"  test_idx 排序:     {'是' if sample_sub['test_idx'].is_monotonic_increasing else '否'}")
    print(f"  模板 test_idx == npz test_idx: {np.array_equal(sample_sub['test_idx'].values, test_idx)}")

    return {
        "num_nodes": num_nodes,
        "num_features": num_features,
        "num_edges": num_edges,
        "num_classes": num_classes,
        "num_train": num_train,
        "num_test": num_test,
        "avg_degree": avg_degree,
        "is_symmetric": is_symmetric,
        "self_loops": num_self_loops,
        "attr_density": attr_density,
        "imbalance_ratio": imbalance_ratio,
        "isolated_nodes": len(isolated),
    }


# ─────────────────────────────────────────────
# 2. 序列推荐任务 (A推荐)
# ─────────────────────────────────────────────

def explore_recommendation():
    print("\n" + "=" * 70)
    print("  任务B: 序列推荐 (A推荐)")
    print("=" * 70)

    base = os.path.join(DATA_ROOT, "A推荐", "A推荐")

    # 读取 metadata
    with open(os.path.join(base, "metadata.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    print(f"\n[metadata.json]")
    print(f"  数据集:      {meta['dataset_name']}")
    print(f"  任务类型:    {meta['task_family']}")
    print(f"  描述:        {meta['description']}")

    # 读取各文件
    train = pd.read_csv(os.path.join(base, "train.csv"))
    test = pd.read_csv(os.path.join(base, "test.csv"))
    user = pd.read_csv(os.path.join(base, "user.csv"))
    item = pd.read_csv(os.path.join(base, "item.csv"))
    sample_sub = pd.read_csv(os.path.join(base, "sample_submission.csv"))

    print(f"\n── 数据规模 ──")
    print(f"  train.csv:           {len(train)} 行")
    print(f"  test.csv:            {len(test)} 行")
    print(f"  user.csv:            {len(user)} 行")
    print(f"  item.csv:            {len(item)} 行")
    print(f"  sample_submission:   {len(sample_sub)} 行")

    # train.csv 详细信息
    print(f"\n── train.csv ──")
    print(f"  列: {list(train.columns)}")
    print(f"  uid 唯一数: {train['uid'].nunique()}")
    print(f"  target_iid 唯一数: {train['target_iid'].nunique()}")

    # 序列长度分析
    seq_lengths_raw = train["item_seq_raw"].apply(
        lambda x: len(str(x).split(",")) if pd.notna(x) and str(x).strip() else 0
    )
    seq_lengths_dedup = train["item_seq_dedup"].apply(
        lambda x: len(str(x).split(",")) if pd.notna(x) and str(x).strip() else 0
    )

    print(f"\n── 序列长度统计 (item_seq_raw) ──")
    print(f"  均值:    {seq_lengths_raw.mean():.2f}")
    print(f"  中位数:  {seq_lengths_raw.median():.0f}")
    print(f"  标准差:  {seq_lengths_raw.std():.2f}")
    print(f"  最小值:  {seq_lengths_raw.min()}")
    print(f"  最大值:  {seq_lengths_raw.max()}")
    print(f"  空序列数: {(seq_lengths_raw == 0).sum()}")

    print(f"\n── 序列长度统计 (item_seq_dedup) ──")
    print(f"  均值:    {seq_lengths_dedup.mean():.2f}")
    print(f"  中位数:  {seq_lengths_dedup.median():.0f}")
    print(f"  最小值:  {seq_lengths_dedup.min()}")
    print(f"  最大值:  {seq_lengths_dedup.max()}")

    # item 交互频次
    all_items = []
    for seq in train["item_seq_raw"].dropna():
        all_items.extend(str(seq).split(","))
    item_freq = Counter(all_items)
    print(f"\n── Item 交互频次 (训练集) ──")
    print(f"  总交互数:            {len(all_items)}")
    print(f"  唯一 item 数:        {len(item_freq)}")
    print(f"  最频繁 item:         {item_freq.most_common(1)[0]}")
    print(f"  最冷门 item 出现次数: {item_freq.most_common()[-1][1] if item_freq else 0}")

    # item.csv 中的 item 是否都在交互中出现
    item_ids_in_data = set(item["iid"])
    item_ids_in_train = set(all_items)
    items_no_interaction = item_ids_in_data - item_ids_in_train
    print(f"  item.csv 中无交互的 item: {len(items_no_interaction)}")

    # target_iid 在候选集中吗
    target_in_candidate = set(train["target_iid"]) - item_ids_in_data
    print(f"  target_iid 不在 item.csv 中: {len(target_in_candidate)}")

    # test.csv 详细信息
    print(f"\n── test.csv ──")
    print(f"  列: {list(test.columns)}")
    print(f"  uid 唯一数: {test['uid'].nunique()}")
    test_seq_lengths = test["item_seq_raw"].apply(
        lambda x: len(str(x).split(",")) if pd.notna(x) and str(x).strip() else 0
    )
    print(f"  序列长度均值: {test_seq_lengths.mean():.2f}")
    print(f"  空序列数: {(test_seq_lengths == 0).sum()}")

    # user.csv
    print(f"\n── user.csv ──")
    print(f"  列: {list(user.columns)}")
    print(f"  uid 唯一数: {user['uid'].nunique()}")
    for col in user.columns[1:]:
        nunique = user[col].nunique()
        print(f"  {col}: {nunique} 个唯一值, 范围 [{user[col].min()}, {user[col].max()}]")

    # user 特征缺失值
    user_missing = user.isnull().sum()
    if user_missing.any():
        print(f"  缺失值: {user_missing[user_missing > 0].to_dict()}")
    else:
        print(f"  无缺失值")

    # item.csv
    print(f"\n── item.csv ──")
    print(f"  列: {list(item.columns)}")
    print(f"  iid 唯一数: {item['iid'].nunique()}")
    for col in item.columns[1:]:
        nunique = item[col].nunique()
        print(f"  {col}: {nunique} 个唯一值, 范围 [{item[col].min()}, {item[col].max()}]")

    item_missing = item.isnull().sum()
    if item_missing.any():
        print(f"  缺失值: {item_missing[item_missing > 0].to_dict()}")
    else:
        print(f"  无缺失值")

    # train/test uid 重叠
    train_uids = set(train["uid"])
    test_uids = set(test["uid"])
    uid_overlap = train_uids & test_uids
    print(f"\n── 数据完整性 ──")
    print(f"  train uid 与 test uid 重叠: {len(uid_overlap)}")
    print(f"  train uid 在 user.csv 中: {len(train_uids - set(user['uid']))} 不在")
    print(f"  test uid 在 user.csv 中:  {len(test_uids - set(user['uid']))} 不在")

    # sample_submission 格式检查
    print(f"\n── 提交模板 ──")
    print(f"  列: {list(sample_sub.columns)}")
    print(f"  uid 唯一数: {sample_sub['uid'].nunique()}")
    print(f"  prediction 样例: {sample_sub['prediction'].iloc[0][:60]}...")
    print(f"  test uid == sample uid: {set(test['uid']) == set(sample_sub['uid'])}")

    # 长尾分布分析
    print(f"\n── Item 长尾分布 ──")
    freq_sorted = sorted(item_freq.values(), reverse=True)
    total_interactions = sum(freq_sorted)
    cumsum = 0
    for i, freq in enumerate(freq_sorted):
        cumsum += freq
        if cumsum >= total_interactions * 0.5:
            print(f"  Top {i+1} item 贡献了 50% 交互")
            break
    for i, freq in enumerate(freq_sorted):
        cumsum_actual = sum(freq_sorted[:i+1])
        if cumsum_actual >= total_interactions * 0.8:
            print(f"  Top {i+1} item 贡献了 80% 交互")
            break
    for i, freq in enumerate(freq_sorted):
        cumsum_actual = sum(freq_sorted[:i+1])
        if cumsum_actual >= total_interactions * 0.9:
            print(f"  Top {i+1} item 贡献了 90% 交互")
            break

    return {
        "num_train_users": len(train),
        "num_test_users": len(test),
        "num_items": len(item),
        "num_users": len(user),
        "avg_seq_len": seq_lengths_raw.mean(),
        "unique_items_in_train": len(item_freq),
        "items_no_interaction": len(items_no_interaction),
    }


# ─────────────────────────────────────────────
# 3. 汇总报告
# ─────────────────────────────────────────────

def print_summary(stats_cls, stats_rec):
    print("\n" + "=" * 70)
    print("  数据质量汇总报告")
    print("=" * 70)

    print(f"""
┌─────────────────────────────────────────────────────────────────┐
│  任务A: 图节点分类                                                │
│  ├── 节点: {stats_cls['num_nodes']:>8,}   特征维度: {stats_cls['num_features']:>5}   边数: {stats_cls['num_edges']:>10,}   │
│  ├── 类别: {stats_cls['num_classes']:>8}   训练: {stats_cls['num_train']:>8,}   测试: {stats_cls['num_test']:>8,}   │
│  ├── 平均度: {stats_cls['avg_degree']:>6.1f}   对称: {str(stats_cls['is_symmetric']):>5}   自环: {stats_cls['self_loops']:>6}   │
│  ├── 特征密度: {stats_cls['attr_density']:.4e}   类别不平衡: {stats_cls['imbalance_ratio']:.2f}         │
│  └── 孤立节点: {stats_cls['isolated_nodes']:>5}                                         │
├─────────────────────────────────────────────────────────────────┤
│  任务B: 序列推荐                                                  │
│  ├── 训练用户: {stats_rec['num_train_users']:>6,}   测试用户: {stats_rec['num_test_users']:>6,}           │
│  ├── 总用户: {stats_rec['num_users']:>8,}   总物品: {stats_rec['num_items']:>6,}              │
│  ├── 平均序列长: {stats_rec['avg_seq_len']:>6.1f}   训练唯一物品: {stats_rec['unique_items_in_train']:>5}  │
│  └── 无交互物品: {stats_rec['items_no_interaction']:>5}                                        │
└─────────────────────────────────────────────────────────────────┘
""")

    # 潜在问题
    issues = []
    if stats_cls["isolated_nodes"] > 0:
        issues.append(f"[A分类] {stats_cls['isolated_nodes']} 个孤立节点，GNN 无法聚合邻居信息")
    if stats_cls["imbalance_ratio"] > 3.0:
        issues.append(f"[A分类] 类别不平衡比 {stats_cls['imbalance_ratio']:.2f}，考虑使用类别权重或采样策略")
    if stats_cls["attr_density"] < 0.01:
        issues.append(f"[A分类] 特征非常稀疏 ({stats_cls['attr_density']:.4e})，考虑特征预处理")
    if stats_rec["items_no_interaction"] > 0:
        issues.append(f"[A推荐] {stats_rec['items_no_interaction']} 个物品无历史交互 (冷启动)")

    if issues:
        print("  [!] Potential issues:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  [OK] No obvious data quality issues found")


if __name__ == "__main__":
    stats_cls = explore_classification()
    stats_rec = explore_recommendation()
    print_summary(stats_cls, stats_rec)
