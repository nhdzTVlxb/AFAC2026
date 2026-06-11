"""推理入口 - 支持Task 1(GNN分类)和Task 2(序列推荐)

使用方法:
    # Task 1: GNN节点分类推理
    python infer.py --task task1 --data_path data/task1/graph.npz \
        --checkpoint output/task1/best_model.pt --output_path A1.csv

    # Task 2: 序列推荐推理
    python infer.py --task task2 --data_path data/task2/ \
        --checkpoint output/task2/best_model.pt --output_path A2.csv

输出格式:
    - A1.csv: node_id,predicted_category
    - A2.csv: user_id,top1,top2,...,top10

Agent可以修改推理逻辑,如集成学习、后处理等。
"""
import os
import sys
import argparse
import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import GNNClassifier, GRU4Rec, SASRec
from datasets import GraphDataset, RecDataset, load_rec_data
from utils import (
    normalize_adj, random_walk_normalize, sparse_to_torch,
    set_seed, get_device, setup_logger
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='自主科研Agent推理脚本')

    # 任务相关
    parser.add_argument('--task', type=str, required=True, choices=['task1', 'task2'],
                        help='任务类型: task1=图分类, task2=序列推荐')
    parser.add_argument('--data_path', type=str, required=True,
                        help='数据路径: task1为.npz文件, task2为数据目录')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--output_path', type=str, required=True,
                        help='输出CSV文件路径')

    # 推理相关
    parser.add_argument('--batch_size', type=int, default=256,
                        help='推理批次大小(Task 2)')
    parser.add_argument('--topk', type=int, default=10,
                        help='推荐Top-K数量(Task 2)')
    parser.add_argument('--device', type=str, default=None,
                        help='计算设备')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    # 可选: 集成多个模型
    parser.add_argument('--ensemble_checkpoints', type=str, nargs='+', default=None,
                        help='多个检查点路径,用于集成推理')

    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path, device):
    """从检查点加载模型

    Args:
        checkpoint_path: 检查点文件路径
        device: 目标设备

    Returns:
        加载好的模型和参数字典
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args_dict = checkpoint.get('args', {})

    # 根据任务类型和模型类型创建模型
    task = args_dict.get('task', 'task1')
    model_type = args_dict.get('model_type', 'sage')

    if task == 'task1' or model_type in ['gcn', 'sage', 'gat']:
        # GNN分类器
        num_features = args_dict.get('num_features', 767)
        num_classes = args_dict.get('num_classes', 10)
        hidden_dim = args_dict.get('hidden_dim', 128)
        num_layers = args_dict.get('num_layers', 2)
        dropout = args_dict.get('dropout', 0.5)

        # 如果args_dict中没有,尝试从数据推断
        if num_features == 767 and num_classes == 10:
            logging.info("使用默认参数: num_features=767, num_classes=10")

        model = GNNClassifier(
            in_dim=num_features,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=num_layers,
            dropout=dropout,
            model_type=model_type
        )
    else:
        # 序列推荐模型
        num_items = args_dict.get('num_items', 2156)
        embedding_dim = args_dict.get('embedding_dim', 64)
        hidden_dim = args_dict.get('hidden_dim', 128)
        num_layers = args_dict.get('num_layers', 1)
        dropout = args_dict.get('dropout', 0.2)
        max_len = args_dict.get('max_len', 50)
        num_heads = args_dict.get('num_heads', 2)

        if model_type == 'gru4rec':
            model = GRU4Rec(
                num_items=num_items,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                max_len=max_len
            )
        elif model_type == 'sasrec':
            model = SASRec(
                num_items=num_items,
                embedding_dim=embedding_dim,
                max_len=max_len,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    logging.info(f"模型加载成功: {model_type} (来自 {checkpoint_path})")
    logging.info(f"检查点信息: epoch={checkpoint.get('epoch', 'unknown')}")

    return model, args_dict


def infer_task1(args):
    """Task 1推理 - GNN节点分类

    加载图数据和训练好的GNN模型,对测试节点进行分类预测,
    输出结果到CSV文件。
    """
    logging.info("=" * 60)
    logging.info("开始 Task 1 推理 - GNN节点分类")
    logging.info("=" * 60)

    device = get_device(args.device)

    # 1. 加载数据
    logging.info(f"加载数据: {args.data_path}")
    data = GraphDataset.load(args.data_path)
    test_idx = data['test_idx']
    num_features = data['num_features']
    num_classes = data['num_classes']
    logging.info(f"测试节点数: {len(test_idx)}, 特征维度: {num_features}, 类别数: {num_classes}")

    # 2. 数据预处理
    if sp.issparse(data['features']):
        features = sparse_to_torch(data['features'], device=device)
    else:
        features = torch.FloatTensor(data['features']).to(device)

    # 邻接矩阵归一化 - 从检查点中读取归一化方式
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = checkpoint.get('args', {})
    normalize_type = saved_args.get('normalize', 'symmetric')

    if normalize_type == 'symmetric':
        adj = normalize_adj(data['adj']).to(device)
    elif normalize_type == 'random_walk':
        adj = random_walk_normalize(data['adj']).to(device)
    else:
        if sp.issparse(data['adj']):
            adj = sparse_to_torch(data['adj'], device=device)
        else:
            adj = torch.FloatTensor(data['adj']).to(device)

    test_idx_t = torch.LongTensor(test_idx).to(device)

    # 3. 加载模型
    model, model_args = load_model_from_checkpoint(args.checkpoint, device)

    # 4. 推理
    logging.info("开始推理...")
    model.eval()
    with torch.no_grad():
        logits = model(features, adj)
        test_logits = logits[test_idx_t]
        predictions = torch.argmax(test_logits, dim=1).cpu().numpy()

    logging.info(f"预测完成,预测类别分布:")
    unique, counts = np.unique(predictions, return_counts=True)
    for cls, cnt in zip(unique, counts):
        logging.info(f"  类别 {cls}: {cnt} 个")

    # 5. 保存结果
    # 输出格式: test_idx,label（与提交模板一致）
    result_df = pd.DataFrame({
        'test_idx': test_idx,
        'label': predictions
    })

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)) if os.path.dirname(args.output_path) else '.', exist_ok=True)
    result_df.to_csv(args.output_path, index=False)
    logging.info(f"结果已保存: {args.output_path} ({len(result_df)} 行)")

    return result_df


def infer_task2(args):
    """Task 2推理 - 序列推荐

    加载序列数据和训练好的推荐模型,为每个测试用户生成Top-10推荐,
    输出结果到CSV文件。
    """
    logging.info("=" * 60)
    logging.info("开始 Task 2 推理 - 序列推荐")
    logging.info("=" * 60)

    device = get_device(args.device)

    # 1. 加载数据
    logging.info(f"加载数据: {args.data_path}")
    rec_data = load_rec_data(args.data_path)

    # 读取测试数据 - 用于获取用户ID
    test_df = pd.read_csv(f'{args.data_path}/test.csv')
    test_seqs, test_targets, test_lens = rec_data['test']
    num_items = rec_data['num_items']
    logging.info(f"测试样本数: {len(test_seqs)}, 物品数: {num_items}")

    # 2. 加载模型
    model, model_args = load_model_from_checkpoint(args.checkpoint, device)
    model_type = model_args.get('model_type', 'gru4rec')

    # 3. 创建DataLoader
    test_dataset = RecDataset(test_seqs, test_targets, test_lens)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # 4. 推理
    logging.info(f"开始推理 (Top-{args.topk})...")
    all_predictions = []
    all_user_ids = []

    # 按用户分组的最后一个序列
    # 读取原始测试CSV来获取用户ID
    user_col = 'uid' if 'uid' in test_df.columns else 'user_id'
    user_ids = []
    seq_indices = []
    last_idx_per_user = {}

    for i, (uid, group) in enumerate(test_df.groupby(user_col)):
        items = group['item_id'].tolist()
        # 每个用户最后一个可预测的交互
        for j in range(1, len(items)):
            last_idx_per_user[uid] = len(user_ids)
            user_ids.append(uid)
            seq_indices.append(len(user_ids) - 1)

    # 如果测试数据是逐行格式,直接使用
    model.eval()
    with torch.no_grad():
        batch_idx = 0
        for batch in test_loader:
            item_seqs, targets, seq_lens = batch
            item_seqs = item_seqs.to(device)
            seq_lens = seq_lens.to(device)

            batch_size = item_seqs.size(0)

            # 获取序列表示
            if model_type == 'gru4rec':
                seq_repr = model(item_seqs, seq_lens)
            else:  # sasrec
                seq_repr = model(item_seqs)

            # 计算所有物品的分数
            all_item_emb = model.item_embedding.weight[1:]  # (num_items, embed_dim)
            scores = seq_repr @ all_item_emb.T  # (batch, num_items)

            # 排除历史交互过的物品
            for i in range(batch_size):
                hist_items = set(item_seqs[i].cpu().numpy())
                for item in hist_items:
                    if 1 <= item <= num_items:
                        scores[i, item - 1] = -1e10

            # 取Top-K
            top_scores, top_indices = torch.topk(scores, k=args.topk, dim=-1)
            top_indices = (top_indices + 1).cpu().numpy()  # 转回原始物品ID(1-based)

            all_predictions.extend(top_indices.tolist())

            batch_idx += 1
            if batch_idx % 100 == 0:
                logging.info(f"  已处理 {batch_idx * args.batch_size} 个样本...")

    # 5. 构建输出
    # 按用户聚合推荐结果
    # 每个用户取最后一次交互的推荐
    user_recommendations = {}

    # 重新按用户分组
    test_df = pd.read_csv(f'{args.data_path}/test.csv')
    user_col = 'uid' if 'uid' in test_df.columns else 'user_id'
    user_seq_idx = 0

    for user_id, group in test_df.groupby(user_col):
        items = group['item_id'].tolist()
        # 最后一个可预测位置
        n_seqs = len(items) - 1 if len(items) > 1 else 1
        # 取该用户最后一个序列的预测
        if user_seq_idx + n_seqs <= len(all_predictions):
            user_recommendations[user_id] = all_predictions[user_seq_idx + n_seqs - 1]
        else:
            # 回退:随机推荐
            user_recommendations[user_id] = list(range(1, args.topk + 1))
        user_seq_idx += n_seqs

    # 构建DataFrame
    result_rows = []
    for user_id in sorted(user_recommendations.keys()):
        recs = user_recommendations[user_id]
        row = {'user_id': user_id}
        for i in range(args.topk):
            row[f'top{i+1}'] = recs[i] if i < len(recs) else 0
        result_rows.append(row)

    result_df = pd.DataFrame(result_rows)

    # 确保列顺序: user_id, top1, top2, ..., top10
    col_order = ['user_id'] + [f'top{i}' for i in range(1, args.topk + 1)]
    result_df = result_df[col_order]

    # 6. 保存结果
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    result_df.to_csv(args.output_path, index=False)
    logging.info(f"结果已保存: {args.output_path} ({len(result_df)} 用户)")

    # 输出一些统计信息
    logging.info("推荐结果统计:")
    for col in [f'top{i}' for i in range(1, min(4, args.topk + 1))]:
        logging.info(f"  {col}: {result_df[col].nunique()} 个不同物品")

    return result_df


def infer_task2_v2(args):
    """Task 2推理 - 简化版(逐用户推理)

    这个版本直接在测试数据上为每个用户生成Top-10推荐,
    不需要逐序列预测,更适合竞赛提交格式。
    """
    logging.info("=" * 60)
    logging.info("开始 Task 2 推理 (v2) - 序列推荐")
    logging.info("=" * 60)

    device = get_device(args.device)

    # 1. 读取测试数据
    test_df = pd.read_csv(f'{args.data_path}/test.csv')

    user_col = 'uid' if 'uid' in test_df.columns else 'user_id'

    # 从checkpoint获取物品映射
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    idx2iid = checkpoint.get('idx2iid', {})
    iid2idx = checkpoint.get('iid2idx', {})
    num_items = checkpoint['args'].get('num_items', len(idx2iid))

    # 2. 加载模型
    model, model_args = load_model_from_checkpoint(args.checkpoint, device)
    model_type = model_args.get('model_type', 'gru4rec')
    max_len = model_args.get('max_len', 50)

    # 3. 确定序列列
    seq_col = None
    for col in ['item_seq_dedup', 'item_seq_raw', 'item_seq']:
        if col in test_df.columns:
            seq_col = col
            break

    # 4. 为每个用户构建序列并推理
    logging.info(f"为每个用户生成Top-{args.topk}推荐...")
    user_recommendations = {}

    model.eval()
    with torch.no_grad():
        for user_id, group in test_df.groupby(user_col):
            # 从序列列获取历史交互
            if seq_col and pd.notna(group[seq_col].iloc[0]):
                seq_str = str(group[seq_col].iloc[0])
                items = [x.strip() for x in seq_str.split(',') if x.strip()]
            else:
                items = []

            # 将字符串iid映射为整数索引
            item_indices = [iid2idx.get(item, 0) for item in items]

            # 构建序列(左填充)
            seq = item_indices[-max_len:]  # 取最近的max_len个
            seq_len = len(seq)
            seq = [0] * (max_len - seq_len) + seq

            item_seq = torch.LongTensor([seq]).to(device)
            seq_len_t = torch.LongTensor([seq_len]).to(device)

            # 获取序列表示
            if model_type == 'gru4rec':
                seq_repr = model(item_seq, seq_len_t)
            else:
                seq_repr = model(item_seq)

            # 计算所有物品分数
            all_item_emb = model.item_embedding.weight[1:]  # (num_items, embed_dim)
            scores = seq_repr @ all_item_emb.T  # (1, num_items)

            # 排除历史物品（使用整数索引）
            hist_indices = set(item_indices)
            for idx in hist_indices:
                if 1 <= idx <= num_items:
                    scores[0, idx - 1] = -1e10

            # 取Top-K
            _, top_indices = torch.topk(scores[0], k=args.topk)
            top_items = (top_indices + 1).cpu().numpy().tolist()

            user_recommendations[user_id] = top_items

    # 5. 构建输出DataFrame
    # 提交格式: uid,prediction（逗号分隔的item id列表）
    result_rows = []
    for user_id in sorted(user_recommendations.keys()):
        recs = user_recommendations[user_id]
        # 将整数索引映射回原始iid字符串
        rec_strs = []
        for r in recs:
            iid_str = idx2iid.get(r, str(r))
            rec_strs.append(iid_str)
        result_rows.append({
            'uid': user_id,
            'prediction': ','.join(rec_strs)
        })

    result_df = pd.DataFrame(result_rows)

    # 6. 保存
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)) if os.path.dirname(args.output_path) else '.', exist_ok=True)
    result_df.to_csv(args.output_path, index=False)
    logging.info(f"结果已保存: {args.output_path} ({len(result_df)} 用户)")

    return result_df


def main():
    """主入口函数"""
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 设置日志
    setup_logger()
    logging.info(f"任务: {args.task}, 数据: {args.data_path}, 检查点: {args.checkpoint}")

    # 设备信息
    device = get_device(args.device)
    logging.info(f"使用设备: {device}")

    # 根据任务类型执行推理
    if args.task == 'task1':
        result = infer_task1(args)
    elif args.task == 'task2':
        # 使用v2版本(更简洁的按用户推理)
        result = infer_task2_v2(args)
    else:
        raise ValueError(f"未知的任务类型: {args.task}")

    logging.info("推理脚本执行完毕!")
    return result


if __name__ == '__main__':
    main()
