"""数据集定义 - 图数据和序列推荐数据

本模块包含:
1. GraphDataset: 加载.npz格式的图数据(CSR稀疏矩阵),用于Task 1
2. RecDataset: 推荐数据集(PyTorch Dataset),用于Task 2

Agent可以修改数据预处理逻辑、特征工程方式等。
"""
import os
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset


class GraphDataset:
    """图数据集 - 加载.npz格式的图数据

    数据格式说明:
    - .npz文件包含CSR格式的邻接矩阵和特征矩阵
    - labels: 节点标签(-1表示无标签)
    - train_idx/test_idx: 训练/测试节点索引
    """

    @staticmethod
    def load(data_path):
        """加载.npz文件

        Args:
            data_path: .npz文件路径

        Returns:
            字典,包含:
                - adj: scipy CSR稀疏邻接矩阵 (N, N)
                - features: scipy CSR稀疏特征矩阵 (N, F)
                - labels: 节点标签数组 (N,)
                - train_idx: 训练节点索引
                - test_idx: 测试节点索引
                - num_nodes: 节点数量
                - num_features: 特征维度
                - num_classes: 类别数量
        """
        data = np.load(data_path, allow_pickle=True)

        # 加载邻接矩阵(CSR格式)
        adj = sp.csr_matrix(
            (data['adj_data'], data['adj_indices'], data['adj_indptr']),
            shape=tuple(data['adj_shape'])
        )

        # 加载特征矩阵(CSR格式)
        features = sp.csr_matrix(
            (data['attr_data'], data['attr_indices'], data['attr_indptr']),
            shape=tuple(data['attr_shape'])
        )

        labels = data['labels']
        train_idx = data['train_idx']
        test_idx = data['test_idx']

        # 自动推断类别数
        num_classes = int(labels[labels >= 0].max()) + 1

        return {
            'adj': adj,
            'features': features,
            'labels': labels,
            'train_idx': train_idx,
            'test_idx': test_idx,
            'num_nodes': adj.shape[0],
            'num_features': features.shape[1],
            'num_classes': num_classes
        }

    @staticmethod
    def preprocess_features(features):
        """特征归一化 (行归一化)

        Agent可以替换为其他特征预处理方法:
        - StandardScaler (标准化)
        - PCA降维
        - 特征选择等
        """
        if sp.issparse(features):
            rowsum = np.array(features.sum(1)).flatten()
            r_inv = np.power(rowsum, -1)
            r_inv[np.isinf(r_inv)] = 0.
            r_mat_inv = sp.diags(r_inv)
            features = r_mat_inv.dot(features)
        return features


class RecDataset(Dataset):
    """推荐数据集 - 用于序列推荐

    数据格式:
    - item_seqs: 用户历史物品序列, shape (N, max_seq_len)
    - targets: 目标物品(下一个要预测的物品), shape (N,)
    - seq_lengths: 实际序列长度, shape (N,)
    """

    def __init__(self, item_seqs, targets, seq_lengths):
        """
        Args:
            item_seqs: 物品序列, list或numpy数组
            targets: 目标物品ID
            seq_lengths: 每个序列的实际长度
        """
        self.item_seqs = torch.LongTensor(item_seqs)
        self.targets = torch.LongTensor(targets)
        self.seq_lengths = torch.LongTensor(seq_lengths)

    def __len__(self):
        return len(self.item_seqs)

    def __getitem__(self, idx):
        return self.item_seqs[idx], self.targets[idx], self.seq_lengths[idx]


class NegativeSamplingDataset(Dataset):
    """带负采样的推荐数据集 - 用于BPR损失训练

    对每个正样本采样若干负样本。
    Agent可以修改负采样策略(随机/基于热度/基于模型等)。
    """

    def __init__(self, item_seqs, targets, seq_lengths, num_items,
                 neg_samples=1, neg_sampling_strategy='random'):
        """
        Args:
            item_seqs: 物品序列
            targets: 目标物品(正样本)
            seq_lengths: 序列长度
            num_items: 总物品数(用于负采样)
            neg_samples: 每个正样本采样的负样本数
            neg_sampling_strategy: 负采样策略
        """
        self.item_seqs = torch.LongTensor(item_seqs)
        self.targets = torch.LongTensor(targets)
        self.seq_lengths = torch.LongTensor(seq_lengths)
        self.num_items = num_items
        self.neg_samples = neg_samples
        self.strategy = neg_sampling_strategy

        # 预计算物品频率(用于 popularity-based 采样)
        targets_int = np.asarray(targets, dtype=np.int64)
        self.item_freq = np.bincount(targets_int, minlength=num_items + 1).astype(np.float32)

    def __len__(self):
        return len(self.item_seqs)

    def _sample_negative(self, target):
        """采样负样本"""
        if self.strategy == 'random':
            neg = np.random.randint(1, self.num_items + 1, size=self.neg_samples)
        elif self.strategy == 'popularity':
            probs = self.item_freq[1:]  # 跳过padding(0)
            probs = probs / probs.sum()
            neg = np.random.choice(self.num_items, size=self.neg_samples,
                                   p=probs, replace=True) + 1
        else:
            neg = np.random.randint(1, self.num_items + 1, size=self.neg_samples)

        # 确保负样本不等于正样本
        for i in range(len(neg)):
            while neg[i] == target:
                neg[i] = np.random.randint(1, self.num_items + 1)
        return neg

    def __getitem__(self, idx):
        pos_target = self.targets[idx].item()
        neg_targets = self._sample_negative(pos_target)
        return (self.item_seqs[idx],
                self.targets[idx],
                torch.LongTensor(neg_targets),
                self.seq_lengths[idx])


def load_rec_data(data_dir, max_seq_len=50):
    """加载推荐数据并构建序列

    Args:
        data_dir: 包含train.csv和test.csv的目录
        max_seq_len: 最大序列长度

    Returns:
        训练数据和测试数据的字典，包含:
            - train: (train_seqs, train_targets, train_lens)
            - test: (test_seqs, test_targets, test_lens)
            - num_users, num_items
            - iid2idx, idx2iid: 物品ID映射表
    """
    # 读取训练数据
    train_df = pd.read_csv(f'{data_dir}/train.csv')
    test_df = pd.read_csv(f'{data_dir}/test.csv')

    # 读取metadata获取物品总数（如果存在）
    metadata_path = f'{data_dir}/metadata.json'
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        num_users = metadata.get('num_users', None)
        num_items = metadata.get('num_items', None)
    else:
        num_users = None
        num_items = None

    # 兼容两种列名格式
    user_col = 'uid' if 'uid' in train_df.columns else 'user_id'
    target_col = 'target_iid' if 'target_iid' in train_df.columns else 'item_id'

    # 读取物品映射表（如果item.csv存在）
    item_csv_path = f'{data_dir}/item.csv'
    if os.path.exists(item_csv_path):
        item_df = pd.read_csv(item_csv_path)
        iid_col = 'iid' if 'iid' in item_df.columns else 'item_id'
        all_iids = item_df[iid_col].astype(str).unique().tolist()
    else:
        # 从训练数据收集所有物品ID（target_iid + 序列中的物品）
        all_iids = set(train_df[target_col].astype(str).unique())
        for col in ['item_seq_dedup', 'item_seq_raw']:
            if col in train_df.columns:
                for seq_str in train_df[col].dropna():
                    for item in str(seq_str).split(','):
                        item = item.strip()
                        if item:
                            all_iids.add(item)
        all_iids = list(all_iids)

    # 建立 iid -> idx 映射（1-based，0为padding）
    iid2idx = {iid: idx + 1 for idx, iid in enumerate(sorted(all_iids))}
    idx2iid = {idx + 1: iid for idx, iid in enumerate(sorted(all_iids))}

    num_items_mapped = len(iid2idx)

    # 获取用户数
    if num_users is None:
        all_uids = set(train_df[user_col].astype(str).unique())
        if user_col in test_df.columns:
            all_uids.update(test_df[user_col].astype(str).unique())
        num_users = len(all_uids)

    if num_items is None:
        num_items = num_items_mapped

    def parse_seq(seq_str, iid2idx):
        """解析逗号分隔的物品序列"""
        if pd.isna(seq_str):
            return []
        items = [x.strip() for x in str(seq_str).split(',') if x.strip()]
        return [iid2idx.get(item, 0) for item in items]

    # 构建训练序列
    train_seqs, train_targets, train_lens = [], [], []
    for _, row in train_df.iterrows():
        target_iid = str(row[target_col])
        target_idx = iid2idx.get(target_iid, 0)

        # 优先使用 item_seq_dedup，其次 item_seq_raw
        seq = []
        for col in ['item_seq_dedup', 'item_seq_raw']:
            if col in train_df.columns:
                seq = parse_seq(row[col], iid2idx)
                if len(seq) > 0:
                    break

        # 序列截断/填充
        if len(seq) > max_seq_len:
            seq = seq[-max_seq_len:]
        seq_len = len(seq)
        seq = [0] * (max_seq_len - len(seq)) + seq

        train_seqs.append(seq)
        train_targets.append(target_idx)
        train_lens.append(seq_len)

    # 构建测试序列
    test_seqs, test_targets, test_lens = [], [], []
    for _, row in test_df.iterrows():
        seq = []
        for col in ['item_seq_dedup', 'item_seq_raw']:
            if col in test_df.columns:
                seq = parse_seq(row[col], iid2idx)
                if len(seq) > 0:
                    break

        if len(seq) > max_seq_len:
            seq = seq[-max_seq_len:]
        seq_len = len(seq)
        seq = [0] * (max_seq_len - len(seq)) + seq

        test_seqs.append(seq)
        test_targets.append(0)  # 测试集无target
        test_lens.append(seq_len)

    return {
        'train': (np.array(train_seqs), np.array(train_targets), np.array(train_lens)),
        'test': (np.array(test_seqs), np.array(test_targets), np.array(test_lens)),
        'num_users': num_users,
        'num_items': num_items,
        'iid2idx': iid2idx,
        'idx2iid': idx2iid,
    }
