"""工具函数

本模块包含各类工具函数:
1. 图相关: 邻接矩阵归一化、稀疏矩阵转换
2. 数据相关: 训练/验证集划分
3. 评估指标: NDCG@K, Accuracy, F1
4. 通用: 随机种子设置、日志记录等

Agent可以自由添加新的工具函数。
"""
import os
import json
import pickle
import random
import math
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F


# ===== 图相关工具 =====

def normalize_adj(adj):
    """对称归一化邻接矩阵: D^(-1/2) * (A + I) * D^(-1/2)

    这是GCN论文中提出的归一化方式。
    Agent可以尝试其他归一化方式如:
    - 随机游走归一化: D^(-1) * (A + I)
    - 不进行归一化

    Args:
        adj: 邻接矩阵, scipy稀疏矩阵或numpy数组或torch.Tensor

    Returns:
        归一化后的邻接矩阵, torch.Tensor (N, N)
    """
    # 转换为torch.Tensor
    if sp.issparse(adj):
        adj = torch.FloatTensor(adj.toarray())
    elif not isinstance(adj, torch.Tensor):
        adj = torch.FloatTensor(adj)

    # 添加自环
    I = torch.eye(adj.size(0), device=adj.device)
    adj_hat = adj + I

    # 计算度矩阵
    D = adj_hat.sum(dim=1)  # (N,)
    D_inv_sqrt = torch.pow(D, -0.5)
    D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0
    D_inv_sqrt = torch.diag(D_inv_sqrt)  # (N, N)

    return D_inv_sqrt @ adj_hat @ D_inv_sqrt


def random_walk_normalize(adj):
    """随机游走归一化: D^(-1) * (A + I)

    这种方式更适合GraphSAGE等模型。
    """
    if sp.issparse(adj):
        adj = torch.FloatTensor(adj.toarray())
    elif not isinstance(adj, torch.Tensor):
        adj = torch.FloatTensor(adj)

    I = torch.eye(adj.size(0), device=adj.device)
    adj_hat = adj + I
    D = adj_hat.sum(dim=1)
    D_inv = torch.pow(D, -1)
    D_inv[torch.isinf(D_inv)] = 0
    D_inv = torch.diag(D_inv)
    return D_inv @ adj_hat


def sparse_to_torch(sparse_mx, device='cpu', return_sparse=False):
    """scipy稀疏矩阵转torch Tensor

    Args:
        sparse_mx: scipy稀疏矩阵
        device: 目标设备
        return_sparse: 是否返回稀疏tensor

    Returns:
        torch.Tensor或torch.sparse_coo_tensor
    """
    if sp.issparse(sparse_mx):
        if return_sparse:
            coo = sparse_mx.tocoo().astype(np.float32)
            indices = torch.LongTensor(np.vstack((coo.row, coo.col)))
            values = torch.FloatTensor(coo.data)
            shape = torch.Size(coo.shape)
            return torch.sparse_coo_tensor(indices, values, shape).to(device)
        else:
            return torch.FloatTensor(sparse_mx.toarray()).to(device)
    elif isinstance(sparse_mx, np.ndarray):
        return torch.FloatTensor(sparse_mx).to(device)
    elif isinstance(sparse_mx, torch.Tensor):
        return sparse_mx.to(device)
    else:
        raise TypeError(f"不支持的类型: {type(sparse_mx)}")


# ===== 数据划分工具 =====

def split_train_val(train_idx, val_ratio=0.2, seed=42):
    """划分训练/验证集

    Args:
        train_idx: 训练索引数组
        val_ratio: 验证集比例
        seed: 随机种子

    Returns:
        train_idx_new, val_idx: 新的训练索引和验证索引
    """
    rng = np.random.default_rng(seed)
    n = len(train_idx)
    perm = rng.permutation(n)
    val_size = int(n * val_ratio)
    return train_idx[perm[val_size:]], train_idx[perm[:val_size]]


def stratified_split(labels, train_idx, val_ratio=0.2, seed=42):
    """分层划分 - 保证每个类别在验证集中的比例一致

    对于类别不平衡的数据集,这种方式更合理。
    """
    rng = np.random.default_rng(seed)
    val_idx_list = []
    train_idx_list = []

    for cls in np.unique(labels[train_idx]):
        cls_idx = train_idx[labels[train_idx] == cls]
        n = len(cls_idx)
        perm = rng.permutation(n)
        val_size = max(int(n * val_ratio), 1)
        val_idx_list.append(cls_idx[perm[:val_size]])
        train_idx_list.append(cls_idx[perm[val_size:]])

    return np.concatenate(train_idx_list), np.concatenate(val_idx_list)


# ===== 评估指标 =====

def compute_accuracy(logits, labels):
    """计算分类准确率

    Args:
        logits: 模型输出的logits, shape (N, C)
        labels: 真实标签, shape (N,)

    Returns:
        准确率 (0-1)
    """
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    return correct / len(labels)


def compute_f1(logits, labels, average='macro'):
    """计算F1分数

    Args:
        logits: 模型输出logits
        labels: 真实标签
        average: 'macro', 'micro', 'weighted'

    Returns:
        F1分数
    """
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()

    from sklearn.metrics import f1_score
    return f1_score(labels, preds, average=average)


def compute_ndcg(predictions, targets, k=10):
    """计算NDCG@K

    Args:
        predictions: 每个用户的推荐列表列表, list of list
        targets: 每个用户的真实目标物品, list
        k: 截断位置

    Returns:
        NDCG@K均值
    """
    ndcg_scores = []
    for pred_list, target in zip(predictions, targets):
        dcg = 0.0
        for i, item in enumerate(pred_list[:k]):
            if item == target:
                dcg = 1.0 / math.log2(i + 2)
                break
        ndcg_scores.append(dcg)
    return np.mean(ndcg_scores) if ndcg_scores else 0.0


def compute_hit_rate(predictions, targets, k=10):
    """计算HitRate@K

    Args:
        predictions: 推荐列表列表
        targets: 目标物品列表
        k: 截断位置

    Returns:
        HitRate@K
    """
    hits = 0
    for pred_list, target in zip(predictions, targets):
        if target in pred_list[:k]:
            hits += 1
    return hits / len(targets) if targets else 0.0


def compute_mrr(predictions, targets):
    """计算MRR (Mean Reciprocal Rank)

    Args:
        predictions: 推荐列表列表
        targets: 目标物品列表

    Returns:
        MRR均值
    """
    rr_scores = []
    for pred_list, target in zip(predictions, targets):
        try:
            rank = pred_list.index(target) + 1
            rr_scores.append(1.0 / rank)
        except ValueError:
            rr_scores.append(0.0)
    return np.mean(rr_scores) if rr_scores else 0.0


# ===== 通用工具 =====

def set_seed(seed=42):
    """设置全局随机种子,保证实验可复现

    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(gpu_id=None):
    """获取计算设备

    Args:
        gpu_id: GPU编号, None表示自动选择；也可以传入 'cpu' 或 'cuda'

    Returns:
        torch.device
    """
    if gpu_id is None:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(gpu_id, str):
        if gpu_id == 'cpu':
            return torch.device('cpu')
        if gpu_id.startswith('cuda'):
            return torch.device(gpu_id)
        # 其他字符串，尝试作为cuda编号
        return torch.device(f'cuda:{gpu_id}')
    # 数字编号
    return torch.device(f'cuda:{gpu_id}')


def save_checkpoint(model, optimizer, epoch, metrics, path):
    """保存模型检查点

    Args:
        model: PyTorch模型
        optimizer: 优化器
        epoch: 当前轮数
        metrics: 评估指标字典
        path: 保存路径
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, path)
    logging.info(f"检查点已保存: {path}")


def load_checkpoint(model, path, optimizer=None, device='cpu'):
    """加载模型检查点

    Args:
        model: PyTorch模型
        path: 检查点路径
        optimizer: 优化器(可选)
        device: 目标设备

    Returns:
        包含epoch和metrics的字典
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    logging.info(f"检查点已加载: {path}, epoch={checkpoint.get('epoch', 'unknown')}")
    return checkpoint


def setup_logger(log_dir=None, log_file=None, level=logging.INFO):
    """设置日志记录器

    Args:
        log_dir: 日志目录
        log_file: 日志文件名
        level: 日志级别

    Returns:
        logging.Logger
    """
    logger = logging.getLogger()
    logger.setLevel(level)

    # 清除已有handler
    logger.handlers.clear()

    # 格式化
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件输出
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        if log_file is None:
            log_file = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(os.path.join(log_dir, log_file))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_config(args, log_dir=None):
    """记录配置参数

    Args:
        args: argparse.Namespace或字典
        log_dir: 日志目录
    """
    if isinstance(args, dict):
        config = args
    else:
        config = vars(args)

    logging.info("=" * 50)
    logging.info("训练配置:")
    for key, value in sorted(config.items()):
        logging.info(f"  {key}: {value}")
    logging.info("=" * 50)

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2, default=str)


class EarlyStopping:
    """早停机制

    当验证指标不再改善时提前停止训练,防止过拟合。
    Agent可以修改patience和monitor指标。
    """

    def __init__(self, patience=10, mode='max', delta=0.0, verbose=True):
        """
        Args:
            patience: 容忍轮数
            mode: 'max'表示指标越高越好(如Accuracy), 'min'表示越低越好(如Loss)
            delta: 改善阈值
            verbose: 是否打印日志
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

        if mode == 'max':
            self.is_better = lambda score, best: score > best + delta
        else:
            self.is_better = lambda score, best: score < best - delta

    def __call__(self, score, epoch):
        """检查是否应该早停

        Args:
            score: 当前验证指标值
            epoch: 当前轮数

        Returns:
            如果指标有改善返回True,否则返回False
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return True

        if self.is_better(score, self.best_score):
            if self.verbose:
                logging.info(f"验证指标改善: {self.best_score:.6f} -> {score:.6f}")
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.verbose:
                logging.info(f"早停计数: {self.counter}/{self.patience} (最优={self.best_score:.6f}, 当前={score:.6f})")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logging.info(f"早停触发! 最优轮数: {self.best_epoch}, 最优指标: {self.best_score:.6f}")
            return False


class MetricsTracker:
    """指标追踪器 - 记录训练和验证指标的历史

    Agent可以使用此类来分析训练趋势和过拟合情况。
    """

    def __init__(self):
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_ndcg': [],
            'learning_rate': []
        }

    def update(self, **kwargs):
        """更新指标

        Args:
            **kwargs: 键值对,如 train_loss=0.5, val_acc=0.9
        """
        for key, value in kwargs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

    def get_best(self, metric='val_acc', mode='max'):
        """获取最优指标值及对应轮数

        Args:
            metric: 指标名称
            mode: 'max'或'min'

        Returns:
            (最优值, 轮数)
        """
        if metric not in self.history or len(self.history[metric]) == 0:
            return None, -1
        values = self.history[metric]
        if mode == 'max':
            best_val = max(values)
        else:
            best_val = min(values)
        best_epoch = values.index(best_val)
        return best_val, best_epoch

    def save(self, path):
        """保存指标历史到文件"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def plot(self, save_path=None):
        """绘制训练曲线"""
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # 损失曲线
            if 'train_loss' in self.history and 'val_loss' in self.history:
                axes[0].plot(self.history['train_loss'], label='Train Loss')
                axes[0].plot(self.history['val_loss'], label='Val Loss')
                axes[0].set_xlabel('Epoch')
                axes[0].set_ylabel('Loss')
                axes[0].set_title('Loss Curve')
                axes[0].legend()
                axes[0].grid(True)

            # 准确率曲线
            if 'val_acc' in self.history:
                axes[1].plot(self.history['val_acc'], label='Val Acc')
                if 'train_acc' in self.history:
                    axes[1].plot(self.history['train_acc'], label='Train Acc')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('Accuracy')
                axes[1].set_title('Accuracy Curve')
                axes[1].legend()
                axes[1].grid(True)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150)
                logging.info(f"训练曲线已保存: {save_path}")
            else:
                plt.show()
            plt.close()
        except ImportError:
            logging.warning("matplotlib未安装,跳过绘图")
