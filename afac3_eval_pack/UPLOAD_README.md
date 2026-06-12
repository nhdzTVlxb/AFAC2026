# AFAC-3 Eval Pack

This folder contains the runnable subset of `luozhonglzw/AFAC-3`.

## Server Usage

Upload this folder to:

```bash
/home/cyp/speedsci/AFAC2026/afac3_eval_pack
```

Run:

```bash
cd /home/cyp/speedsci/AFAC2026/afac3_eval_pack
chmod +x run_afac3_best.sh
bash run_afac3_best.sh
```

The script expects the original v1 data here:

```bash
/home/cyp/speedsci/AFAC2026/v1/framework/data/cls_data
/home/cyp/speedsci/AFAC2026/v1/framework/data/rec_data
```

Outputs:

```bash
/home/cyp/speedsci/AFAC2026/afac3_eval_pack/prediction_afac3.zip
/home/cyp/speedsci/AFAC2026/v1/framework/output_afac3/prediction.zip
```

Optional XGBRanker version:

```bash
cd /home/cyp/speedsci/AFAC2026/afac3_eval_pack
chmod +x run_afac3_xgb_optional.sh
bash run_afac3_xgb_optional.sh
```
