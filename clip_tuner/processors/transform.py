import logging
from typing import Dict, Any, Callable

from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    ColorJitter,
    GaussianBlur,
    RandomGrayscale,
    ToTensor,
    Resize,
)

from torchvision.transforms.functional import InterpolationMode

_TRANSFORM_REGISTRY = {
    "Resize": Resize,
    "RandomResizedCrop": RandomResizedCrop,
    "CenterCrop": CenterCrop,
    "RandomHorizontalFlip": RandomHorizontalFlip,
    "ColorJitter": ColorJitter,
    "GaussianBlur": GaussianBlur,
    "RandomGrayscale": RandomGrayscale,
}


def build_transform_pipeline(
    image_size,
    image_mean,
    image_std,
    transform_config: Dict[str, Dict[str, Any]] = None,
    logger=None,
) -> Callable:
    """从 config 构建 Compose 流水线

    Args:
        transform_config (Dict[str, Dict[str, Any]]): 见 clip_tuner/configs/transform/*.yaml 中的 YAML 配置

    Returns:
        Callable: Compose 流水线
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    transforms = []
    if "RandomResizedCrop" in transform_config:
        params = dict(transform_config["RandomResizedCrop"])
        params.setdefault("size", tuple(image_size))
        if "interpolation" in params:
            i = params["interpolation"]
            params["interpolation"] = InterpolationMode[i.upper()]
        transforms.append(RandomResizedCrop(**params))
    else:
        if "Resize" in transform_config:
            # Resize
            resize_params = dict(transform_config["Resize"])
            resize_params.setdefault("size", tuple(image_size))
            if "interpolation" in resize_params:
                i = resize_params["interpolation"]
                resize_params["interpolation"] = InterpolationMode[i.upper()]
        else:
            resize_params = {
                "size": tuple(image_size),
                "interpolation": InterpolationMode.BICUBIC,
            }
        transforms.append(Resize(**resize_params))

        if "CenterCrop" in transform_config:
            cparams = dict(transform_config["CenterCrop"])
            params.setdefault("size", tuple(image_size))
        else:
            cparams = {"size": tuple(image_size)}
        transforms.append(CenterCrop(**cparams))

    for name, params in transform_config.items():
        if name in ("RandomResizedCrop", "Resize", "CenterCrop"):
            continue

        cls = _TRANSFORM_REGISTRY.get(name)
        if cls is None:
            logger.warning(f"Ignoring unknown transform '{name}'")
            continue

        transforms.append(_TRANSFORM_REGISTRY[name](**params))
    transforms.append(ToTensor())
    transforms.append(Normalize(tuple(image_mean), tuple(image_std)))
    return Compose(transforms)


def get_train_transforms(
    image_processor=None,
    image_size=(224, 224),
    image_mean=[0.485, 0.456, 0.406],
    image_std=[0.229, 0.224, 0.225],
    transform_config: Dict[str, Dict[str, Any]] = None,
):
    if image_processor is not None:
        pipline = build_transform_pipeline(
            image_size=image_processor.image_size,
            image_mean=image_processor.image_mean,
            image_std=image_processor.image_std,
            transform_config=transform_config,
        )
    else:
        pipline = build_transform_pipeline(
            image_size=image_size,
            image_mean=image_mean,
            image_std=image_std,
            transform_config=transform_config,
        )

    def train_transforms(example_batch):
        """Apply _train_transforms across a batch."""
        # 对一个批次的数据应用_train_transforms
        example_batch["pixel_values"] = [
            pipline(pil_img.convert("RGB")) for pil_img in example_batch["image"]
        ]

        del example_batch["image"]
        return example_batch

    return train_transforms


def get_val_transforms(
    image_size=224,
    image_mean=[0.485, 0.456, 0.406],
    image_std=[0.229, 0.224, 0.225],
):
    pipline = Compose(
        [
            Resize(image_size),
            CenterCrop(image_size),
            ToTensor(),
            Normalize(image_mean, image_std),
        ]
    )

    def val_transforms(example_batch):
        """Apply _val_transforms across a batch."""
        example_batch["pixel_values"] = [
            pipline(pil_img.convert("RGB")) for pil_img in example_batch["image"]
        ]
        del example_batch["image"]
        return example_batch

    return val_transforms
