# T1 图纠错模型交接说明

## 版本定位

这是一个“传统 13 模基线 + 图模型小范围纠错”的完整交接版本。

主流程不是只迭代图模型：`baseline_train.py` 会完整训练传统基线，并导出
纠错器训练所需的基线 OOF 概率。后续队友修改传统基线代码后，纠错器会自动
使用本次新生成的基线 OOF 重新训练，不会被冻结 npz 卡住。

图模型当前作为纠错专家：只在“基线预测与图模型预测不同，且该节点有训练邻居
覆盖”的少量样本上考虑改写。

## 一键运行

解压后在本目录执行：

```powershell
python train.py --output-dir output/run1
```

默认完整流程：

1. 运行 `baseline_train.py`，训练传统 13 模基线；
2. `baseline_train.py` 写出 `output/run1/baseline/A1.csv` 和 `base_oof_probs.npz`；
3. `train.py` 使用本次新生成的 `base_oof_probs.npz` 重建纠错器训练输入；
4. 读取 `deps/graph_correction_artifacts.npz` 中的图模型概率；
5. 在 OOF 分歧样本上五折训练 `ExtraTrees` 纠错器；
6. 使用阈值 `0.65` 做保守纠错；
7. 写出最终 `output/run1/A1.csv`。

快速等价检查：

```powershell
python train.py --output-dir output/run1 --reuse-baseline
```

该模式会复用 `artifacts/reference_baseline_A1.csv` 和
`artifacts/reference_base_oof_probs.npz`，跳过基线重训。它只用于快速检查，
不是默认迭代路径。

## 包内文件

- `train.py`：唯一入口，负责基线训练、OOF 读取、图纠错和最终提交。
- `baseline_train.py`：传统 13 模基线训练代码，会导出基线提交和基线 OOF。
- `base_v7.py`：图相关工具函数。
- `data/`：任务数据与提交模板。
- `deps/`：图纠错和固定验证所需的冻结产物。
- `artifacts/`：已验证参考产物，仅用于快速复检和对照。
- `provenance/`：冻结图产物和验证辅助产物的生成代码快照。
- `output/run1/A1.csv`：已验证的最终提交文件示例。

## 冻结产物来源

`deps/` 中仍有图纠错相关冻结产物，因为当前交接版本的图模型作为外部纠错专家
使用。它们的生成代码快照已经放在 `provenance/` 下，方便队友后续重建或替换。

传统 13 模基线不再依赖冻结 OOF：默认完整运行会由 `baseline_train.py`
重新生成基线 OOF。

## 已验证结果

- 基线提交 SHA256：
  `7CE5239BEAF969FCB6FF72B8CC9644D7009A809B868A91F2875BFE10E54D2160`
- 最终提交 SHA256：
  `F06E40E2A47F9046B557B84A9B7EE11B19EAF7262111EDACBDA8CF3A2E45B4B1`
- 最终提交相对基线只改写 `11` 行。

## 后续迭代建议

后续应复制整个目录到新目录后再修改。若改传统基线模型，直接改
`baseline_train.py`；完整运行后会自动用新的基线 OOF 训练纠错器。若改图纠错
策略，则优先看 `train.py` 开头的 `CONFIG`：纠错器类型、阈值、门槛规则。
