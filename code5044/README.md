# AFAC2026 Task2 0.5044 方案：1211 SSL freeze4

这个目录是线上 `1211 = 0.5044` 的独立复现实验包。

## 方法

纯神经 SASRec 路线，不使用 LGB，不做规则重排。

核心流程：

1. 用 `train + test` 的历史序列做 self-supervised next-item 预训练。
2. 将预训练得到的 `item_emb / pos_emb / SASRec encoder` 迁移到 Task2 235 类目标分类模型。
3. fine-tune 时前 4 个 epoch 冻结 SSL backbone，只训练 Task2 头部和用户融合层。
4. 第 5 个 epoch 解冻全模型继续训练。

输出文件：

```text
output_1211_freeze4/output1211_ssl_freeze4/A2.csv
```

## 文件说明

```text
train_1211_freeze4.py              # 单独复现 1211 的入口
train_neural_rec_1100_1105.py      # 基础 SASRec / 数据集 / 评估 / 保存提交
train_ssl_seq_pretrain_1200_1204.py# SSL next-item 预训练函数
train_ssl_transfer_1210_1214.py    # SSL 迁移 fine-tune 函数
run_1211_freeze4.sh                # 从零跑：SSL 预训练 + 1211 fine-tune
run_1211_reuse_ssl.sh              # 复用已存在 SSL checkpoint 后重跑 fine-tune
```

## 运行

默认数据目录：

```text
/home/cyp/speedsci/AFAC2026/v1/framework/data/rec_data
```

从零训练：

```bash
cd /home/cyp/speedsci/AFAC2026/code504
bash run_1211_freeze4.sh
```

指定数据目录和输出目录：

```bash
bash run_1211_freeze4.sh \
  /home/cyp/speedsci/AFAC2026/v1/framework/data/rec_data \
  /home/cyp/speedsci/AFAC2026/code504/output_1211_freeze4
```

如果 `output_1211_freeze4/ssl_nextitem_sasrec_l3.pt` 已存在，只重跑 1211 fine-tune：

```bash
bash run_1211_reuse_ssl.sh
```

## 预期日志关键点

SSL 预训练 loss 会大致从 `3.x` 降到 `1.7x`。

1211 fine-tune 关键日志：

```text
output1211_ssl_freeze4 unfroze ssl backbone
...
output1211_ssl_freeze4 ep=20 ... val_test_ndcg≈0.49696
```

最终提交文件：

```text
/home/cyp/speedsci/AFAC2026/code504/output_1211_freeze4/output1211_ssl_freeze4/A2.csv
```

线上已测分数：

```text
0.5044
```
