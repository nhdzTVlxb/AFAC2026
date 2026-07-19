# 冻结产物来源说明

默认完整流程会重新训练传统 13 模基线，并由 `baseline_train.py`
生成新的 `output/run1/baseline/base_oof_probs.npz`。因此，后续修改
`baseline_train.py` 后，纠错器会自动使用新的基线 OOF。

`deps/` 中的冻结产物主要来自图模型与固定验证辅助流程：

- `graph_correction_artifacts.npz`：图模型在训练集 OOF 和测试集上的概率。
- `graph_testlike_probabilities.npz`：图模型在固定 test-like 验证集上的概率。
- `baseline_testlike_artifacts.npz`：基线模型在固定 test-like 验证集上的概率。
- `domain_propensity.csv`：节点域倾向特征，供纠错器使用。

`artifacts/reference_base_oof_probs.npz` 和
`artifacts/reference_baseline_A1.csv` 只用于 `--reuse-baseline` 快速复检，
不是默认完整训练路径。

本目录下各子目录保存的是这些产物的生成代码快照，供追溯或后续重建使用。
