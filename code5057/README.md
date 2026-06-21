# AFAC2026 Task2 0.5057 方案：1214 SSL transfer fusion

这个目录是线上 `1214 = 0.5057` 的独立打包版。

## 方法

纯神经融合，不使用 LGB。

1214 是五个已训练神经提交的 rank fusion：

| 分量 | 方法 | 权重 |
|---|---|---:|
| 1120 | BPR/item2vec 预训练 + SASRec testmix | 1.10 |
| 1113 | 早期 BPR/item2vec 预训练深 SASRec | 0.95 |
| 1200 | SSL next-item pretrain + 正常 fine-tune | 1.00 |
| 1210 | SSL checkpoint + backbone 低学习率迁移 | 0.85 |
| 1211 | SSL checkpoint + freeze4 再解冻 | 0.75 |

融合公式是按每个分量 top10 的 rank 加权：

```text
score(item) += weight * (10 - rank) / 10
```

然后每个用户取 score 最高的 10 个 item。

## 文件说明

```text
make_1214_fusion.py                  # 用 components 里的五个 A2 重新生成 1214
run_make_1214.sh                     # 一键重建 1214 A2
components/*.csv                     # 五个分量提交文件
output1214_ssl_transfer_x_stable/A2.csv # 已测 0.5057 的最终提交

train_neural_rec_1100_1105.py
train_neural_pretrain_1110_1116.py
train_neural_pretrain_1120_1126.py
train_ssl_seq_pretrain_1200_1204.py
train_ssl_transfer_1210_1214.py
```

训练脚本也放在目录里，便于追溯各分量来源；但如果只是复现最终 1214，不需要重训，直接运行融合脚本即可。

## 直接重建 1214

```bash
cd /home/cyp/speedsci/AFAC2026/code5057
bash run_make_1214.sh
```

输出：

```text
/home/cyp/speedsci/AFAC2026/code5057/output1214_ssl_transfer_x_stable/A2.csv
```

## 手动指定 sample/output

```bash
bash run_make_1214.sh \
  /home/cyp/speedsci/AFAC2026/v1/framework/data/rec_data/sample_submission.csv \
  /home/cyp/speedsci/AFAC2026/code5057/output1214_ssl_transfer_x_stable/A2.csv
```

线上已测分数：

```text
0.5057
```

