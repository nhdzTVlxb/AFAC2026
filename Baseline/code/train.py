"""训练入口 - 支持Task 1(GNN分类)和Task 2(序列推荐)

使用方法:
    # Task 1: GNN节点分类
    python train.py --task task1 --data_path data/task1/graph.npz --model_type sage

    # Task 2: 序列推荐
    python train.py --task task2 --data_path data/task2/ --model_type gru4rec

Agent可以修改以下参数来优化模型:
    --model_type: 模型类型 (sage/gcn/gru4rec/sasrec)
    --hidden_dim: 隐藏层维度
    --num_layers: 层数
    --lr: 学习率
    --dropout: Dropout率
    --epochs: 训练轮数
    --weight_decay: 权重衰减
    --batch_size: 批次大小
"""
import os
import sys
import math
import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 导入自定义模块
from models import GNNClassifier, GRU4Rec, SASRec
from datasets import GraphDataset, RecDataset, NegativeSamplingDataset, load_rec_data
from utils import (
    normalize_adj, random_walk_normalize, sparse_to_torch,
    split_train_val, stratified_split, set_seed, get_device,
    compute_accuracy, compute_ndcg, compute_hit_rate, compute_mrr,
    save_checkpoint, setup_logger, log_config,
    EarlyStopping, MetricsTracker
)


def parse_args():
    """解析命令行参数 - Agent可以在这里添加新的超参数"""
    parser = argparse.ArgumentParser(description='自主科研Agent训练脚本')

    # 任务相关
    parser.add_argument('--task', type=str, required=True, choices=['task1', 'task2'],
                        help='任务类型: task1=图分类, task2=序列推荐')
    parser.add_argument('--data_path', type=str, required=True,
                        help='数据路径: task1为.npz文件, task2为数据目录')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='输出目录')

    # 模型相关
    parser.add_argument('--model_type', type=str, default='sage',
                        choices=['gcn', 'sage', 'gat', 'gru4rec', 'sasrec'],
                        help='模型类型')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='GNN层数或RNN/Transformer层数')
    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='嵌入维度(序列模型)')
    parser.add_argument('--num_heads', type=int, default=2,
                        help='注意力头数(SASRec)')
    parser.add_argument('--max_len', type=int, default=50,
                        help='最大序列长度')

    # 训练相关
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout率')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='权重衰减(L2正则化)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='批次大小')
    parser.add_argument('--patience', type=int, default=20,
                        help='早停容忍轮数')

    # 数据相关
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='验证集比例')
    parser.add_argument('--normalize', type=str, default='symmetric',
                        choices=['symmetric', 'random_walk', 'none'],
                        help='邻接矩阵归一化方式')
    parser.add_argument('--stratified_split', action='store_true',
                        help='使用分层划分')

    # 其他
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--device', type=str, default=None,
                        help='计算设备,如 cuda:0 或 cpu')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='日志打印间隔(轮数)')
    parser.add_argument('--save_best_only', action='store_true',
                        help='只保存最优模型')

    # 序列推荐特有
    parser.add_argument('--neg_samples', type=int, default=1,
                        help='负采样数量(推荐任务)')
    parser.add_argument('--loss_type', type=str, default='bpr',
                        choices=['bpr', 'ce'],
                        help='损失函数类型: bpr=BPR损失, ce=交叉熵损失')

    return parser.parse_args()


# ===== Task 1: GNN节点分类训练 =====

def train_task1(args):
    """Task 1训练流程 - GNN节点分类"""
    logging.info("=" * 60)
    logging.info("开始 Task 1 训练 - GNN节点分类")
    logging.info("=" * 60)

    # 1. 加载数据
    logging.info(f"加载数据: {args.data_path}")
    data = GraphDataset.load(args.data_path)
    num_nodes = data['num_nodes']
    num_features = data['num_features']
    num_classes = data['num_classes']
    logging.info(f"节点数: {num_nodes}, 特征维度: {num_features}, 类别数: {num_classes}")

    # 2. 划分训练/验证集
    if args.stratified_split:
        train_idx, val_idx = stratified_split(
            data['labels'], data['train_idx'], args.val_ratio, args.seed
        )
    else:
        train_idx, val_idx = split_train_val(
            data['train_idx'], args.val_ratio, args.seed
        )
    logging.info(f"训练集: {len(train_idx)}, 验证集: {len(val_idx)}, 测试集: {len(data['test_idx'])}")

    # 3. 数据预处理
    device = get_device(args.device)

    # 特征转tensor
    if sp.issparse(data['features']):
        features = sparse_to_torch(data['features'], device=device)
    else:
        features = torch.FloatTensor(data['features']).to(device)

    # 邻接矩阵归一化
    if args.normalize == 'symmetric':
        adj = normalize_adj(data['adj']).to(device)
    elif args.normalize == 'random_walk':
        adj = random_walk_normalize(data['adj']).to(device)
    else:
        if sp.issparse(data['adj']):
            adj = sparse_to_torch(data['adj'], device=device)
        else:
            adj = torch.FloatTensor(data['adj']).to(device)

    # 标签转tensor
    labels = torch.LongTensor(data['labels']).to(device)
    train_idx_t = torch.LongTensor(train_idx).to(device)
    val_idx_t = torch.LongTensor(val_idx).to(device)

    # 4. 创建模型
    model = GNNClassifier(
        in_dim=num_features,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
        model_type=args.model_type
    ).to(device)

    logging.info(f"模型: {args.model_type}, 隐藏维度: {args.hidden_dim}, 层数: {args.num_layers}")
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"总参数量: {total_params:,}")

    # 5. 优化器
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # 学习率调度器 - Agent可以修改调度策略
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=args.patience // 3
    )
    criterion = nn.CrossEntropyLoss()

    # 6. 早停和指标追踪
    early_stop = EarlyStopping(patience=args.patience, mode='max', verbose=True)
    tracker = MetricsTracker()

    # 7. 训练循环
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        # ---- 训练阶段 ----
        model.train()
        optimizer.zero_grad()

        logits = model(features, adj)
        loss = criterion(logits[train_idx_t], labels[train_idx_t])
        loss.backward()
        # 梯度裁剪 - 防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # 计算训练准确率
        with torch.no_grad():
            train_acc = compute_accuracy(logits[train_idx_t], labels[train_idx_t])

        # ---- 验证阶段 ----
        model.eval()
        with torch.no_grad():
            logits = model(features, adj)
            val_loss = criterion(logits[val_idx_t], labels[val_idx_t]).item()
            val_acc = compute_accuracy(logits[val_idx_t], labels[val_idx_t])

        # 更新学习率
        scheduler.step(val_acc)

        # 记录指标
        tracker.update(
            train_loss=loss.item(),
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            learning_rate=optimizer.param_groups[0]['lr']
        )

        # 打印日志
        if epoch % args.log_interval == 0 or epoch == 1:
            logging.info(
                f"Epoch [{epoch:3d}/{args.epochs}] "
                f"Train Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

        # 保存最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'args': vars(args)
            }
            best_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save(save_dict, best_path)

        # 早停检查
        early_stop(val_acc, epoch)
        if early_stop.early_stop:
            logging.info(f"早停触发! 最优验证准确率: {best_val_acc:.4f} (Epoch {best_epoch})")
            break

    # 8. 训练结束
    logging.info("=" * 60)
    logging.info(f"训练完成! 最优验证准确率: {best_val_acc:.4f} (Epoch {best_epoch})")
    logging.info("=" * 60)

    # 保存最终模型
    final_path = os.path.join(args.output_dir, 'final_model.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'args': vars(args)
    }, final_path)

    # 保存训练历史
    tracker.save(os.path.join(args.output_dir, 'metrics.json'))

    return best_val_acc


# ===== Task 2: 序列推荐训练 =====

def train_task2(args):
    """Task 2训练流程 - 序列推荐"""
    logging.info("=" * 60)
    logging.info("开始 Task 2 训练 - 序列推荐")
    logging.info("=" * 60)

    # 1. 加载数据
    logging.info(f"加载数据: {args.data_path}")
    rec_data = load_rec_data(args.data_path, max_seq_len=args.max_len)

    train_seqs, train_targets, train_lens = rec_data['train']
    test_seqs, test_targets, test_lens = rec_data['test']
    num_users = rec_data['num_users']
    num_items = rec_data['num_items']
    iid2idx = rec_data.get('iid2idx', {})
    idx2iid = rec_data.get('idx2iid', {})
    logging.info(f"用户数: {num_users}, 物品数: {num_items}")
    logging.info(f"训练样本: {len(train_seqs)}, 测试样本: {len(test_seqs)}")

    # 2. 划分训练/验证集
    n_train = len(train_seqs)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_train)
    val_size = int(n_train * args.val_ratio)

    val_seqs = train_seqs[perm[:val_size]]
    val_targets = train_targets[perm[:val_size]]
    val_lens = train_lens[perm[:val_size]]

    trn_seqs = train_seqs[perm[val_size:]]
    trn_targets = train_targets[perm[val_size:]]
    trn_lens = train_lens[perm[val_size:]]

    logging.info(f"训练集: {len(trn_seqs)}, 验证集: {len(val_seqs)}")

    # 3. 创建Dataset和DataLoader
    device = get_device(args.device)

    # 训练集使用负采样
    if args.loss_type == 'bpr':
        train_dataset = NegativeSamplingDataset(
            trn_seqs, trn_targets, trn_lens, num_items,
            neg_samples=args.neg_samples
        )
    else:
        train_dataset = RecDataset(trn_seqs, trn_targets, trn_lens)

    # 验证集不使用负采样
    val_dataset = RecDataset(val_seqs, val_targets, val_lens)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # 4. 创建模型
    if args.model_type == 'gru4rec':
        model = GRU4Rec(
            num_items=num_items,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            max_len=args.max_len
        )
    elif args.model_type == 'sasrec':
        model = SASRec(
            num_items=num_items,
            embedding_dim=args.embedding_dim,
            max_len=args.max_len,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    else:
        raise ValueError(f"Task 2不支持的模型类型: {args.model_type}")

    model = model.to(device)
    logging.info(f"模型: {args.model_type}")
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"总参数量: {total_params:,}")

    # 5. 优化器
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=args.patience // 3
    )

    # 6. 早停和指标追踪
    early_stop = EarlyStopping(patience=args.patience, mode='max', verbose=True)
    tracker = MetricsTracker()

    # 7. 训练循环
    best_val_ndcg = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        # ---- 训练阶段 ----
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            if args.loss_type == 'bpr':
                item_seqs, targets, neg_items, seq_lens = batch
                item_seqs = item_seqs.to(device)
                targets = targets.to(device)
                neg_items = neg_items.to(device)
                seq_lens = seq_lens.to(device)

                optimizer.zero_grad()

                # 获取序列表示
                seq_repr = model(item_seqs, seq_lens)  # (batch, embed_dim)

                # 正样本分数
                pos_emb = model.item_embedding(targets)  # (batch, embed_dim)
                pos_score = (seq_repr * pos_emb).sum(dim=-1)  # (batch,)

                # 负样本分数
                neg_emb = model.item_embedding(neg_items.squeeze(1))  # (batch, embed_dim)
                neg_score = (seq_repr * neg_emb).sum(dim=-1)  # (batch,)

                # BPR损失: -log(sigmoid(pos - neg))
                loss = -F.logsigmoid(pos_score - neg_score).mean()

            else:  # CE损失
                item_seqs, targets, seq_lens = batch
                item_seqs = item_seqs.to(device)
                targets = targets.to(device)
                seq_lens = seq_lens.to(device)

                optimizer.zero_grad()

                seq_repr = model(item_seqs, seq_lens)  # (batch, embed_dim)
                scores = seq_repr @ model.item_embedding.weight[1:].T  # (batch, num_items)
                # 过滤padding target（target=0表示padding）
                valid_mask = targets > 0
                if valid_mask.sum() == 0:
                    continue
                scores = scores[valid_mask]
                valid_targets = targets[valid_mask] - 1  # 转为0-based索引
                loss = F.cross_entropy(scores, valid_targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0

        # ---- 验证阶段 ----
        model.eval()
        val_predictions = []
        val_targets_list = []

        with torch.no_grad():
            for batch in val_loader:
                item_seqs, targets, seq_lens = batch
                item_seqs = item_seqs.to(device)
                seq_lens = seq_lens.to(device)

                # 获取序列表示
                if args.model_type == 'gru4rec':
                    seq_repr = model(item_seqs, seq_lens)
                else:
                    seq_repr = model(item_seqs)

                # 计算所有物品的分数
                all_item_emb = model.item_embedding.weight[1:]  # (num_items, embed_dim)
                scores = seq_repr @ all_item_emb.T  # (batch, num_items)

                # 排除历史交互过的物品
                for i in range(len(item_seqs)):
                    hist_items = set(item_seqs[i].cpu().numpy())
                    for item in hist_items:
                        if 1 <= item <= num_items:
                            scores[i, item - 1] = -1e10

                # 取Top-K
                _, top_indices = torch.topk(scores, k=10, dim=-1)
                top_indices = (top_indices + 1).cpu().numpy()  # 转回原始ID

                val_predictions.extend(top_indices.tolist())
                val_targets_list.extend(targets.numpy().tolist())

        # 计算验证指标
        val_ndcg = compute_ndcg(val_predictions, val_targets_list, k=10)
        val_hit = compute_hit_rate(val_predictions, val_targets_list, k=10)
        val_mrr = compute_mrr(val_predictions, val_targets_list)

        # 更新学习率
        scheduler.step(val_ndcg)

        # 记录指标
        tracker.update(
            train_loss=avg_train_loss,
            val_ndcg=val_ndcg,
            val_hit=val_hit,
            val_mrr=val_mrr,
            learning_rate=optimizer.param_groups[0]['lr']
        )

        # 打印日志
        if epoch % args.log_interval == 0 or epoch == 1:
            logging.info(
                f"Epoch [{epoch:3d}/{args.epochs}] "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val NDCG@10: {val_ndcg:.4f}, Hit@10: {val_hit:.4f}, MRR: {val_mrr:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

        # 保存最优模型
        if val_ndcg > best_val_ndcg:
            best_val_ndcg = val_ndcg
            best_epoch = epoch
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_ndcg': val_ndcg,
                'args': vars(args),
                'iid2idx': iid2idx,
                'idx2iid': idx2iid,
            }
            best_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save(save_dict, best_path)

        # 早停检查
        early_stop(val_ndcg, epoch)
        if early_stop.early_stop:
            logging.info(f"早停触发! 最优验证NDCG@10: {best_val_ndcg:.4f} (Epoch {best_epoch})")
            break

    # 8. 训练结束
    logging.info("=" * 60)
    logging.info(f"训练完成! 最优验证NDCG@10: {best_val_ndcg:.4f} (Epoch {best_epoch})")
    logging.info("=" * 60)

    # 保存最终模型
    final_path = os.path.join(args.output_dir, 'final_model.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'args': vars(args)
    }, final_path)

    # 保存训练历史
    tracker.save(os.path.join(args.output_dir, 'metrics.json'))

    return best_val_ndcg


# ===== 主函数 =====

def main():
    """主入口函数"""
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置日志
    log_dir = os.path.join(args.output_dir, 'logs')
    setup_logger(log_dir=log_dir)

    # 记录配置
    log_config(args, log_dir)

    # 设备信息
    device = get_device(args.device)
    logging.info(f"使用设备: {device}")

    # 根据任务类型执行训练
    if args.task == 'task1':
        best_metric = train_task1(args)
    elif args.task == 'task2':
        best_metric = train_task2(args)
    else:
        raise ValueError(f"未知的任务类型: {args.task}")

    logging.info("训练脚本执行完毕!")
    return best_metric


if __name__ == '__main__':
    main()
