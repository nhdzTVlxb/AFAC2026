#!/usr/bin/env python3

from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

try:
    import timm
except ImportError as exc:
    raise ImportError(
        "当前集成包含 ConvNeXt-Small，提交环境必须安装 timm。"
    ) from exc


# =========================================================
# 文件路径
# =========================================================
CLASSES_PATH = Path("/home/jovyan/work/classes.txt")

ENET_MODEL_PATH = Path(
    "/home/jovyan/work/results/best_acc_enet.pth"
)
TINY_MODEL_PATH = Path(
    "/home/jovyan/work/results/best_acc_tiny.pth"
)
SMALL_MODEL_PATH = Path(
    "/home/jovyan/work/results/best_acc_small.pth"
)

NUM_CLASSES = 25


# =========================================================
# 三模型融合比例
# =========================================================
# 默认：
# EfficientNetV2-S : ConvNeXt-Tiny : ConvNeXt-Small = 3 : 3 : 4
#
# 不要求总和等于 1，程序会自动归一化。
# 将某个权重设为 0，可以临时关闭对应模型。
ENSEMBLE_WEIGHTS = {
    "enet": 3.0,
    "tiny": 3.0,
    "small": 4.0,
}


# =========================================================
# 多尺寸 TTA
# =========================================================
# True：每个模型分别在全部 TTA_SIZES 上预测，再平均 logits。
# False：每个模型使用其 checkpoint 中保存的训练 image_size。
ENABLE_MULTI_SCALE_TTA = True

# 可自行修改，例如：
# TTA_SIZES = (320, 352, 384)
# TTA_SIZES = (352, 384, 416)
# TTA_SIZES = (384,)
#
# 交通标志包含左转/右转等镜像异类，不使用水平翻转 TTA。
TTA_SIZES = (352, 384, 416)

# CUDA 可用时启用 AMP 半精度推理。
ENABLE_AMP = True

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)
_LOAD_LOCK = Lock()

_MODELS = None
_CLASSES = None
_INPUT_SIZES = None
_NORMALIZED_WEIGHTS = None


# =========================================================
# 类别读取
# =========================================================
def normalize_label(label):
    return " ".join(str(label).strip().split())


def load_classes(path=CLASSES_PATH):
    if not path.is_file():
        raise FileNotFoundError(
            "classes.txt 不存在：{}".format(path)
        )

    indexed = {}

    for line_number, raw_line in enumerate(
        path.read_text(
            encoding="utf-8-sig"
        ).splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(
                "classes.txt 第 {} 行格式错误，"
                "应为 '<类别ID> <类别名称>'".format(
                    line_number
                )
            )

        try:
            class_id = int(parts[0])
        except ValueError as exc:
            raise ValueError(
                "classes.txt 第 {} 行类别 ID 不是整数".format(
                    line_number
                )
            ) from exc

        class_name = normalize_label(parts[1])

        if class_id in indexed:
            raise ValueError(
                "classes.txt 第 {} 行类别 ID 重复：{}".format(
                    line_number,
                    class_id,
                )
            )

        if not class_name:
            raise ValueError(
                "classes.txt 第 {} 行类别名称为空".format(
                    line_number
                )
            )

        indexed[class_id] = class_name

    if sorted(indexed) != list(range(NUM_CLASSES)):
        raise ValueError(
            "classes.txt 必须且只能包含连续编号 0～24"
        )

    classes = [
        indexed[index]
        for index in range(NUM_CLASSES)
    ]

    if len(set(classes)) != NUM_CLASSES:
        raise ValueError(
            "classes.txt 规范化后存在重复类别"
        )

    return classes


# =========================================================
# checkpoint 工具
# =========================================================
def torch_load_checkpoint(path):
    if not path.is_file():
        raise FileNotFoundError(
            "模型权重不存在：{}".format(path)
        )

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(
            path,
            map_location="cpu",
        )

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "checkpoint 必须是字典格式：{}".format(path)
        )

    if "model_state_dict" not in checkpoint:
        checkpoint = {
            "model_state_dict": checkpoint,
            "args": {},
        }

    return checkpoint


def clean_state_dict(state_dict):
    if state_dict and all(
        key.startswith("module.")
        for key in state_dict
    ):
        return {
            key[7:]: value
            for key, value in state_dict.items()
        }

    return state_dict


def checkpoint_classes(
    checkpoint,
    fallback,
    model_name,
):
    mapping = checkpoint.get("id_to_class")

    if mapping is not None:
        if isinstance(mapping, list):
            result = [
                normalize_label(name)
                for name in mapping
            ]
        else:
            result = [
                normalize_label(
                    mapping.get(
                        index,
                        mapping.get(str(index)),
                    )
                )
                for index in range(NUM_CLASSES)
            ]

    else:
        class_to_idx = checkpoint.get(
            "class_to_idx"
        )

        if class_to_idx is None:
            return fallback

        result_temp = [None] * NUM_CLASSES

        for class_name, class_id in class_to_idx.items():
            class_id = int(class_id)

            if 0 <= class_id < NUM_CLASSES:
                result_temp[class_id] = normalize_label(
                    class_name
                )

        if any(
            name is None
            for name in result_temp
        ):
            raise ValueError(
                "{} checkpoint 的 class_to_idx 不完整".format(
                    model_name
                )
            )

        result = [
            str(name)
            for name in result_temp
        ]

    if (
        len(result) != NUM_CLASSES
        or any(not name for name in result)
    ):
        raise ValueError(
            "{} checkpoint 的类别映射不完整".format(
                model_name
            )
        )

    if result != fallback:
        raise ValueError(
            "{} checkpoint 类别顺序与 classes.txt 不一致".format(
                model_name
            )
        )

    return result


def checkpoint_input_size(
    checkpoint,
    default=384,
):
    args = checkpoint.get("args") or {}

    image_size = args.get(
        "image_size",
        checkpoint.get(
            "input_size",
            default,
        ),
    )

    if isinstance(
        image_size,
        (list, tuple),
    ):
        image_size = image_size[0]

    image_size = int(image_size)

    if image_size <= 0:
        raise ValueError(
            "checkpoint 中的 image_size 非法：{}".format(
                image_size
            )
        )

    return image_size


# =========================================================
# 模型定义
# =========================================================
def build_enet(checkpoint):
    backbone = str(
        checkpoint.get(
            "backbone",
            "efficientnet_v2_s",
        )
    )

    if backbone != "efficientnet_v2_s":
        raise ValueError(
            "best_acc_enet.pth 不是 EfficientNetV2-S 权重："
            "backbone={!r}".format(backbone)
        )

    args = checkpoint.get("args") or {}
    dropout = float(
        args.get("dropout", 0.35)
    )

    # 推理阶段不联网下载，完整参数随后由 checkpoint 覆盖。
    try:
        model = models.efficientnet_v2_s(
            weights=None
        )
    except TypeError:
        model = models.efficientnet_v2_s(
            pretrained=False
        )

    in_features = model.classifier[-1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(
            p=dropout,
            inplace=True,
        ),
        nn.Linear(
            in_features,
            NUM_CLASSES,
        ),
    )

    return model


def build_tiny(checkpoint):
    backbone = str(
        checkpoint.get(
            "backbone",
            "convnext_tiny",
        )
    )

    if backbone != "convnext_tiny":
        raise ValueError(
            "best_acc_tiny.pth 不是 ConvNeXt-Tiny 权重："
            "backbone={!r}".format(backbone)
        )

    args = checkpoint.get("args") or {}
    dropout = float(
        args.get("dropout", 0.30)
    )

    try:
        model = models.convnext_tiny(
            weights=None
        )
    except TypeError:
        model = models.convnext_tiny(
            pretrained=False
        )

    in_features = model.classifier[-1].in_features

    model.classifier = nn.Sequential(
        model.classifier[0],
        model.classifier[1],
        nn.Dropout(p=dropout),
        nn.Linear(
            in_features,
            NUM_CLASSES,
        ),
    )

    return model


class ConvNeXtSmallTrafficSign(nn.Module):
    def __init__(
        self,
        model_name,
        num_classes,
        dropout,
        drop_path_rate,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            global_pool="avg",
            drop_path_rate=drop_path_rate,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(
                self.backbone.num_features
            ),
            nn.Dropout(dropout),
            nn.Linear(
                self.backbone.num_features,
                num_classes,
            ),
        )

    def forward(self, images):
        features = self.backbone(images)
        return self.classifier(features)


def build_small(checkpoint):
    args = checkpoint.get("args") or {}

    model_name = str(
        args.get(
            "model_name",
            checkpoint.get(
                "backbone",
                "convnext_small.fb_in22k_ft_in1k",
            ),
        )
    )

    dropout = float(
        args.get("dropout", 0.35)
    )

    drop_path = float(
        args.get("drop_path", 0.10)
    )

    return ConvNeXtSmallTrafficSign(
        model_name=model_name,
        num_classes=NUM_CLASSES,
        dropout=dropout,
        drop_path_rate=drop_path,
    )


# =========================================================
# 集成权重和模型加载
# =========================================================
def normalized_ensemble_weights():
    expected = {
        "enet",
        "tiny",
        "small",
    }

    if set(ENSEMBLE_WEIGHTS) != expected:
        raise ValueError(
            "ENSEMBLE_WEIGHTS 必须包含且只包含："
            "'enet'、'tiny'、'small'"
        )

    weights = {}

    for name, raw_weight in ENSEMBLE_WEIGHTS.items():
        weight = float(raw_weight)

        if not np.isfinite(weight):
            raise ValueError(
                "{} 的集成权重不是有限数值：{}".format(
                    name,
                    weight,
                )
            )

        if weight < 0:
            raise ValueError(
                "{} 的集成权重不能为负数：{}".format(
                    name,
                    weight,
                )
            )

        weights[name] = weight

    total = sum(weights.values())

    if total <= 0:
        raise ValueError(
            "三个集成权重不能全部为 0"
        )

    return {
        name: weight / total
        for name, weight in weights.items()
    }


def load_one_model(
    name,
    path,
    build_function,
    classes,
):
    checkpoint = torch_load_checkpoint(path)

    checkpoint_classes(
        checkpoint=checkpoint,
        fallback=classes,
        model_name=name,
    )

    model = build_function(checkpoint)

    state_dict = clean_state_dict(
        checkpoint["model_state_dict"]
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.to(_DEVICE)
    model.to(
        memory_format=torch.channels_last
    )
    model.eval()

    input_size = checkpoint_input_size(
        checkpoint
    )

    return model, input_size


def load_models_once():
    global _MODELS
    global _CLASSES
    global _INPUT_SIZES
    global _NORMALIZED_WEIGHTS

    if (
        _MODELS is not None
        and _CLASSES is not None
        and _INPUT_SIZES is not None
        and _NORMALIZED_WEIGHTS is not None
    ):
        return (
            _MODELS,
            _CLASSES,
            _INPUT_SIZES,
            _NORMALIZED_WEIGHTS,
        )

    with _LOAD_LOCK:
        if (
            _MODELS is not None
            and _CLASSES is not None
            and _INPUT_SIZES is not None
            and _NORMALIZED_WEIGHTS is not None
        ):
            return (
                _MODELS,
                _CLASSES,
                _INPUT_SIZES,
                _NORMALIZED_WEIGHTS,
            )

        classes = load_classes()
        weights = normalized_ensemble_weights()

        model_specs = {
            "enet": (
                ENET_MODEL_PATH,
                build_enet,
            ),
            "tiny": (
                TINY_MODEL_PATH,
                build_tiny,
            ),
            "small": (
                SMALL_MODEL_PATH,
                build_small,
            ),
        }

        loaded_models = {}
        input_sizes = {}

        for name, spec in model_specs.items():
            path, build_function = spec

            if weights[name] <= 0:
                continue

            model, input_size = load_one_model(
                name=name,
                path=path,
                build_function=build_function,
                classes=classes,
            )

            loaded_models[name] = model
            input_sizes[name] = input_size

        if not loaded_models:
            raise RuntimeError(
                "没有启用任何集成模型"
            )

        _MODELS = loaded_models
        _CLASSES = classes
        _INPUT_SIZES = input_sizes
        _NORMALIZED_WEIGHTS = weights

        active_weights = {
            name: round(
                weights[name],
                6,
            )
            for name in loaded_models
        }

        print(
            "Loaded ensemble | "
            "device={} | "
            "amp={} | "
            "weights={} | "
            "input_sizes={} | "
            "multi_scale_tta={} | "
            "tta_sizes={}".format(
                _DEVICE,
                ENABLE_AMP
                and _DEVICE.type == "cuda",
                active_weights,
                input_sizes,
                ENABLE_MULTI_SCALE_TTA,
                TTA_SIZES
                if ENABLE_MULTI_SCALE_TTA
                else "checkpoint",
            )
        )

        return (
            _MODELS,
            _CLASSES,
            _INPUT_SIZES,
            _NORMALIZED_WEIGHTS,
        )


# =========================================================
# 图片预处理
# =========================================================
def validate_tta_sizes(values):
    sizes = []

    for value in values:
        size = int(value)

        if size <= 0:
            raise ValueError(
                "TTA 尺寸必须为正整数，实际为：{}".format(
                    size
                )
            )

        if size not in sizes:
            sizes.append(size)

    if not sizes:
        raise ValueError(
            "TTA_SIZES 不能为空"
        )

    return tuple(sizes)


def numpy_to_rgb_tensor(image):
    """
    平台通常传入 cv2 读取的 BGR uint8 图像。

    返回：
        1×3×H×W、RGB、float32、值域 [0,1] 的 tensor。
    """
    if (
        not isinstance(image, np.ndarray)
        or image.size == 0
    ):
        raise ValueError(
            "X 必须是非空 np.ndarray"
        )

    if not np.issubdtype(
        image.dtype,
        np.number,
    ):
        raise TypeError(
            "不支持的图像 dtype：{}".format(
                image.dtype
            )
        )

    if not np.isfinite(image).all():
        raise ValueError(
            "图像包含 NaN 或无穷值"
        )

    if image.ndim == 2:
        rgb = np.repeat(
            image[..., None],
            3,
            axis=2,
        )

    elif (
        image.ndim == 3
        and image.shape[2] == 1
    ):
        rgb = np.repeat(
            image,
            3,
            axis=2,
        )

    elif (
        image.ndim == 3
        and image.shape[2] == 3
    ):
        if np.issubdtype(
            image.dtype,
            np.integer,
        ):
            rgb = image[..., ::-1]
        else:
            rgb = image

    elif (
        image.ndim == 3
        and image.shape[2] == 4
    ):
        if np.issubdtype(
            image.dtype,
            np.integer,
        ):
            rgb = image[..., [2, 1, 0]]
        else:
            rgb = image[..., :3]

    else:
        raise ValueError(
            "不支持的图像形状：{}".format(
                image.shape
            )
        )

    rgb = np.ascontiguousarray(
        rgb
    ).astype(np.float32)

    if np.issubdtype(
        image.dtype,
        np.integer,
    ):
        rgb /= float(
            np.iinfo(image.dtype).max
        )
    else:
        minimum = float(rgb.min())
        maximum = float(rgb.max())

        if (
            minimum < 0.0
            or maximum > 255.0
        ):
            raise ValueError(
                "浮点图像值域必须在 [0,1] 或 [0,255]"
            )

        if maximum > 1.0:
            rgb /= 255.0

    tensor = torch.from_numpy(
        rgb.transpose(2, 0, 1)
    ).unsqueeze(0)

    return tensor.to(
        _DEVICE,
        non_blocking=True,
    )


def resize_bilinear(
    tensor,
    size,
):
    try:
        return F.interpolate(
            tensor,
            size=(size, size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    except TypeError:
        return F.interpolate(
            tensor,
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        )


def normalize_batch(batch):
    mean = batch.new_tensor(
        IMAGENET_MEAN
    ).view(1, 3, 1, 1)

    std = batch.new_tensor(
        IMAGENET_STD
    ).view(1, 3, 1, 1)

    return (batch - mean) / std


def amp_autocast():
    enabled = (
        ENABLE_AMP
        and _DEVICE.type == "cuda"
    )

    return torch.cuda.amp.autocast(
        enabled=enabled
    )


# =========================================================
# 单模型 TTA 和最终预测
# =========================================================
@torch.no_grad()
def model_tta_logits(
    model,
    base_image,
    sizes,
):
    logits_sum = None

    for size in sizes:
        batch = resize_bilinear(
            base_image,
            size=size,
        )

        batch = normalize_batch(batch)

        batch = batch.contiguous(
            memory_format=torch.channels_last
        )

        with amp_autocast():
            logits = model(batch)

        logits = logits.float()

        if logits_sum is None:
            logits_sum = logits
        else:
            logits_sum = (
                logits_sum + logits
            )

    if logits_sum is None:
        raise RuntimeError(
            "没有产生模型预测 logits"
        )

    return logits_sum / float(len(sizes))


@torch.no_grad()
def predict(X):
    """
    参数：
        X: np.ndarray，通常由 cv2.imread 读取，BGR 格式。

    返回：
        str，classes.txt 中 25 个正式类别名称之一。
    """
    (
        models_dict,
        classes,
        input_sizes,
        weights,
    ) = load_models_once()

    base_image = numpy_to_rgb_tensor(X)

    if ENABLE_MULTI_SCALE_TTA:
        shared_tta_sizes = validate_tta_sizes(
            TTA_SIZES
        )
    else:
        shared_tta_sizes = None

    ensemble_logits = None
    used_weight = 0.0

    for name, model in models_dict.items():
        if ENABLE_MULTI_SCALE_TTA:
            model_sizes = shared_tta_sizes
        else:
            model_sizes = (
                input_sizes[name],
            )

        logits = model_tta_logits(
            model=model,
            base_image=base_image,
            sizes=model_sizes,
        )

        weight = float(weights[name])

        if ensemble_logits is None:
            ensemble_logits = (
                logits * weight
            )
        else:
            ensemble_logits = (
                ensemble_logits
                + logits * weight
            )

        used_weight += weight

    if (
        ensemble_logits is None
        or used_weight <= 0
    ):
        raise RuntimeError(
            "集成预测失败：没有有效模型"
        )

    ensemble_logits = (
        ensemble_logits / used_weight
    )

    class_index = int(
        ensemble_logits.argmax(
            dim=1
        ).item()
    )

    if not (
        0 <= class_index < NUM_CLASSES
    ):
        raise RuntimeError(
            "模型输出类别索引越界：{}".format(
                class_index
            )
        )

    return normalize_label(
        classes[class_index]
    )


if __name__ == "__main__":
    print(
        "Ensemble submission module is ready.\n"
        "Device: {}\n"
        "Weights: {}\n"
        "Multi-scale TTA: {}\n"
        "TTA sizes: {}".format(
            _DEVICE,
            ENSEMBLE_WEIGHTS,
            ENABLE_MULTI_SCALE_TTA,
            TTA_SIZES,
        )
    )
