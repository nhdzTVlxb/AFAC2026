# Task 2 final code package

This directory is a code snapshot for the three Task 2 models and their
fusion pipelines. Existing trained models and submissions remain in their
original directories.

## Layout

- `models/lgb/train_lgb.py`: reproduced LightGBM ranker.
- `models/nn1_deeper/train_nn1_deeper.py`: deeper residual neural ranker.
- `models/nn2_medium/train_nn2_medium.py`: medium residual neural ranker.
- `fusion/LGB_NN_Ensemble_train.py`: shared-OOF LGB + NN training/fusion base.
- `fusion/run_LGB_NN_deeper_blend.py`: LGB + NN1 training launcher.
- `fusion/make_blend_from_scores.py`: configurable two-model post-processing.
- `fusion/make_LGB_NN1_NN2_blend.py`: final LGB:NN1:NN2 = 4:3:3 fusion.

## Environment and data

```bash
conda activate speedsci
export DATA_ROOT=/home/cyp/speedsci/AFAC2026/data/rec_data
```

The original working artifacts used by the final fusion are under:

- `/home/cyp/speedsci/AFAC2026/task2_lgb_reproduce`
- `/home/cyp/speedsci/AFAC2026/task2_t2_deeper`
- `/home/cyp/speedsci/AFAC2026/task2_nn_medium/nn2_fivefold`
- `/home/cyp/speedsci/AFAC2026/task2_blend/output`

## Final fusion

The verified three-model fusion command is:

```bash
cd /home/cyp/speedsci/AFAC2026/final-task2/fusion
PYTHONHASHSEED=0 python -u make_LGB_NN1_NN2_blend.py
```

It writes both rank and z-score variants. The primary submission uses z-score
normalization and weights LGB 0.4, NN1 0.3, and NN2 0.3.
