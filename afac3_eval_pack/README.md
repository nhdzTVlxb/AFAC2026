# AFAC2026 赛题三 - 稀疏反馈下的自动化实验挑战

## 项目简介

本项目是 AFAC2026 赛题三 "稀疏反馈下的自动化实验挑战" 的参赛方案。核心目标是构建一个自动化实验 Agent 系统，在有限预算内自动搜索最优模型配置，完成图节点分类和序列推荐两个任务。

## 任务概述

### 任务A: 产品分类（图节点分类）
- **数据**: 13,752 节点，767 维稀疏特征，10 类，CSR 格式邻接矩阵
- **挑战**: 图非常稀疏（平均度 2.04），4,259 个孤立节点，类别严重不平衡（17.71:1）
- **方案**: GCN / GAT / MLP 基线，支持特征归一化、图对称化、类别权重

### 任务B: 产品推荐（序列推荐）
- **数据**: 50,000 用户，2,156 物品，平均序列长度 103.65
- **挑战**: 冷启动问题（548 物品无交互），长尾分布（Top 21 物品贡献 50% 交互）
- **方案**: SASRec 自注意力序列模型 + 启发式基线（马尔可夫转移 + 协同过滤）

## 项目结构

```
├── data_explore.py          # 数据探索与质量报告
├── train_cls.py             # 任务A GCN训练脚本
├── predict_cls.py           # 任务A GCN+LP集成预测
├── predict_cls_deg.py       # 任务A 度加权集成+阈值优化
├── gat_model.py             # 任务A 稀疏GAT模型
├── appnp_model.py           # 任务A APPNP模型
├── train_cls_feat.py        # 任务A GCN+图结构特征
├── train_task_b_v4.py       # 任务B 频率加权推荐
├── train_task_b_v6.py       # 任务B l2t权重优化
├── train_rec.py             # 任务B SASRec训练
├── predict_rec.py           # 任务B SASRec预测
├── agent_framework.py       # 自动化实验 Agent 框架
├── experiment_runner.py     # 实验执行器
├── trajectory_logger.py     # 轨迹日志记录器
├── trajectory_B1.json       # 任务A 实验轨迹
├── trajectory_B2.json       # 任务B 实验轨迹
├── A1.csv                   # 任务A 提交文件 (73.92%)
├── A2.csv                   # 任务B 提交文件 (81.3%)
├── prediction.zip           # 最终提交包
└── README.md                # 项目说明
```

## 核心组件

### 1. Agent 框架 (agent_framework.py)

自动化实验系统的核心，包含：

- **配置空间**: 定义模型类型、隐藏维度、学习率等可搜索参数
- **策略模块**: 支持随机搜索 → 贪心改进的两阶段策略
- **预算管理**: 控制实验次数和运行时间
- **停止决策**: 连续 N 轮无提升则停止

```bash
# 运行 Agent 自动搜索
python agent_framework.py --task cls --max_experiments 20 --max_time_hours 2
python agent_framework.py --task rec --max_experiments 20 --max_time_hours 2
```

### 2. 实验执行器 (experiment_runner.py)

封装了完整的训练/评估流程：

- GCN/GAT/MLP 三种图神经网络模型
- SASRec 自注意力序列推荐模型
- 自动处理数据预处理、训练、验证、指标计算

### 3. 轨迹日志 (trajectory_logger.py)

记录每轮实验的完整信息：

- 配置参数、训练指标、反馈信息
- 输出 trajectory_B1.json / trajectory_B2.json

## 快速开始

### 环境要求
- Python 3.10+
- PyTorch 2.0+
- NumPy, Pandas, SciPy, scikit-learn

### 训练与预测

```bash
# 任务A: 图节点分类
python train_cls.py        # 训练 GCN 模型
python predict_cls.py      # 生成 A1.csv

# 任务B: 序列推荐
python train_rec.py        # 训练 SASRec 模型
python predict_rec.py      # 生成 A2.csv

# 数据探索
python data_explore.py     # 输出数据质量报告
```

## 实验结果

| 任务 | 模型 | 验证指标 | 备注 |
|------|------|----------|------|
| A 分类 | GCN-3layer | 70.83% Accuracy | hidden=256, dropout=0.5 |
| A 分类 | GCN+LP 平均集成 | 73.15% Accuracy | GCN + Label Propagation |
| A 分类 | GCN+LP 度加权集成 | 73.65% Accuracy | 度感知LP-heavy策略 |
| A 分类 | GCN+LP+阈值优化 | **73.92% Accuracy** | +per-class阈值boost |
| A 分类 | GAT | 70.88% Accuracy | 稀疏GAT，不如GCN |
| A 分类 | APPNP | 72.65% Accuracy | K=5, alpha=0.1 |
| B 推荐 | 启发式基线 | Hit@10=64.7% | 马尔可夫 + 协同过滤 |
| B 推荐 | 频率加权 | Hit@10=75.9% | freq_weight=1000 |
| B 推荐 | 频率+l2t加权 | Hit@10=81.3% | l2t_weight=200 |
| B 推荐 | NDCG优化+交叉增强 | **Hit@10=85.4%, NDCG@10=0.606** | l2t_boost=50, ib=4.5 |
| B 推荐 | SASRec | Hit@10=38.85% | CPU 训练，欠拟合 |

## Agent 搜索实验轨迹

Agent 框架在分类任务上运行了 5 轮实验：

- 最优配置: GAT, hidden=256, 3层, dropout=0.5, lr=0.01
- 指标范围: 0.35 ~ 0.68
- 发现: 图对称化和类别权重对性能有显著影响

## 技术亮点

1. **自动化实验**: Agent 框架实现了配置搜索、反馈分析、策略调整的闭环
2. **稀疏图处理**: 针对高度稀疏的图数据，实现了稀疏 GAT 避免 OOM
3. **类别不平衡**: 使用加权交叉熵损失处理 17.71:1 的类别比
4. **冷启动**: 启发式基线通过马尔可夫转移和协同过滤缓解冷启动问题

## 后续改进方向

- [ ] 在 GPU 上充分训练 SASRec 模型
- [ ] 实现 GAT + GCN 集成
- [ ] 引入对比学习增强序列表示
- [ ] 使用贝叶斯优化替代随机搜索
- [ ] 实现跨任务知识迁移

## 许可证

本项目仅用于 AFAC2026 比赛参赛，数据版权归比赛主办方所有。
