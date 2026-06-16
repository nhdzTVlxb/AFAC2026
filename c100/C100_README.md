# c100 Task2 Package

This folder contains the c100 experiments for Task2:

- `train_lgb_xgb.py`: original 0.4941 code.
- `make_4941_source657.py`: generator that adds output657 source-rank features, GPU knobs, candidate knobs, and LGB-only memory-saving switches.
- `run_4941_source657.sh`: base runner used by the c100 scripts.
- `blend_4941_657.py`: result-level rank-percentile fusion between 0.4941 A2 and output657 A2.

## Recommended Run

Run c100 LGB-only first. It is the best match to current online evidence because `output657/A2_lgbm` scored 0.4945 and XGB fusion often hurt.

```bash
cd /home/cyp/speedsci/AFAC2026/afac3_eval_pack/任务二4941版本代码/c100
chmod +x *.sh

DATA_ROOT=/home/cyp/speedsci/AFAC2026/afac3_eval_pack/A推荐/A推荐 \
OUT_DIR=/home/cyp/speedsci/AFAC2026/afac3_eval_pack/任务二4941版本代码/c100/output_c100_lgb_only \
bash run_c100_lgb_only.sh
```

Submit first:

```text
output_c100_lgb_only/A2_lgb.csv
```

## If c100 OOMs

If it is killed after meta-validation, use the skip-final probe:

```bash
DATA_ROOT=/home/cyp/speedsci/AFAC2026/afac3_eval_pack/A推荐/A推荐 \
OUT_DIR=/home/cyp/speedsci/AFAC2026/afac3_eval_pack/任务二4941版本代码/c100/output_c100_lgb_only_skip_final \
bash run_c100_lgb_only_skip_final.sh
```

This uses the meta-trained LGB directly for test prediction. It is a probe, not the cleanest final-training setup.

## Full LGB/XGB Version

This is high-risk for memory and not the current preferred submission, but it is included for completeness:

```bash
DATA_ROOT=/home/cyp/speedsci/AFAC2026/afac3_eval_pack/A推荐/A推荐 \
OUT_DIR=/home/cyp/speedsci/AFAC2026/afac3_eval_pack/任务二4941版本代码/c100/output_c100_lgb_xgb_full \
bash run_c100_lgb_xgb_full.sh
```

Outputs:

- `A2_lgb.csv`: LGB single model.
- `A2_xgb.csv`: XGB single model.
- `A2_ensemble.csv`: LGB/XGB rank fusion.
- `A2.csv`: default ensemble.

## Result-Level Blend

Use this if you already have a 0.4941 A2 file and want to fuse it with output657:

```bash
A2_4941=/path/to/0.4941/A2.csv \
A2_657=/home/cyp/speedsci/AFAC2026/afac3_eval_pack/task2_feature_model_submit/output657_lgb_source_rank_features/A2.csv \
OUT_DIR=/home/cyp/speedsci/AFAC2026/afac3_eval_pack/任务二4941版本代码/c100/blend_4941_657_outputs \
bash run_blend_4941_657.sh
```

The script writes several weighted blends and a `summary.csv`.
