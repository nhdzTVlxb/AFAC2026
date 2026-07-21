"""Launch the remote T2 LGB + deeper-NN 50/50 fusion experiment."""
from __future__ import annotations

import importlib.util
import os
import random
from pathlib import Path


ROOT = Path("/home/cyp/speedsci/AFAC2026")
BLEND_ROOT = ROOT / "task2_blend"


def load_base():
    path = BLEND_ROOT / "LGB_NN_Ensemble_train.py"
    spec = importlib.util.spec_from_file_location("lgb_nn_ensemble_base", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load fusion base: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def train_deeper_nn(nnmod, lgbmod, oof, cols):
    import numpy as np
    import torch

    random.seed(2025)
    np.random.seed(2025)
    torch.manual_seed(2025)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2025)

    mt, mv = lgbmod.split_valid(oof, frac=0.12)
    model = nnmod.NNRankerWrapper()
    model.fit(
        lgbmod.fm(mt, cols),
        mt["label"].astype(np.float32).to_numpy(),
        mt["qid"].to_numpy(),
        X_val=lgbmod.fm(mv, cols),
        y_val=mv["label"].astype(np.float32).to_numpy(),
        qid_val=mv["qid"].to_numpy(),
        val_meta=mv.copy(),
        epochs=nnmod.NN_EPOCHS,
        batch_size=nnmod.NN_BATCH_SIZE,
    )
    mv = mv.copy()
    mv["nn_score"] = model.predict(lgbmod.fm(mv, cols)).astype("float32")
    return model, mv


def main():
    base = load_base()
    base.VERSION = "T2_LGB_REPRO_NN_V46_DEEPER_FUSION_50_50"
    base.VERSION_ROOT = BLEND_ROOT
    base.DATA_ROOT = ROOT / "data" / "rec_data"
    base.LGB_SOURCE = ROOT / "task2_lgb_reproduce" / "T2_LGB_train.py"
    base.NN_SOURCE = ROOT / "task2_t2_deeper" / "T2_NN_train.py"
    base.OUTPUT_DIR = BLEND_ROOT / "output"
    base.OOF_WORK_DIR = BLEND_ROOT / "work" / "oof_folds"
    base.SUBMISSION_NAME = "A2_blend_50_50.csv"
    base.BLEND_RATIOS = ((0.5, 0.5),)
    base.train_nn = train_deeper_nn

    original_load_module = base.load_module

    def load_compatible_module(name, path):
        module = original_load_module(name, path)
        if name == "lgb_v37_source":
            original_build_oof = module.build_oof

            def build_oof(train, cat_cols, base_dir):
                return original_build_oof(
                    train,
                    cat_cols,
                    oof_dir=str(Path(base_dir) / "oof_folds"),
                )

            module.build_oof = build_oof
        return module

    base.load_module = load_compatible_module
    os.environ["DATA_ROOT"] = str(base.DATA_ROOT)
    base.main()


if __name__ == "__main__":
    main()
