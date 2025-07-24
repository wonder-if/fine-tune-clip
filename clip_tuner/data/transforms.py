import logging
from typing import Callable
from hydra.utils import instantiate

from torchvision.transforms.functional import InterpolationMode


def build_image_transforms(transform_cfg, image_processor=None) -> Callable:

    def fix_interp(cfg):
        if hasattr(cfg, "interpolation") and isinstance(cfg.interpolation, str):
            cfg.interpolation = InterpolationMode[cfg.interpolation]

    logger = logging.getLogger(__name__)

    if image_processor is not None:
        transform_cfg.common.image_size = [
            image_processor.crop_size["height"],
            image_processor.crop_size["width"],
        ]
        transform_cfg.common.image_mean = image_processor.image_mean
        transform_cfg.common.image_std = image_processor.image_std

    train_transforms = None
    eval_transforms = None

    if transform_cfg.train:
        # 处理 interpolation 字段
        for t in transform_cfg.train.transforms:
            fix_interp(t)

        pipline = instantiate(transform_cfg.train)

        logger.info(f"Train transforms: {pipline}")

        def train_transforms(example_batch):
            """Apply train_transforms across a batch."""
            # 对一个批次的数据应用 train_transforms
            example_batch["pixel_values"] = [
                pipline(pil_img.convert("RGB")) for pil_img in example_batch["image"]
            ]

            del example_batch["image"]
            return example_batch

    if transform_cfg.eval:
        # 处理 interpolation 字段
        for t in transform_cfg.eval.transforms:
            fix_interp(t)

        pipline = instantiate(transform_cfg.eval)
        logger.info(f"Eval transforms: {pipline}")

        def val_transforms(example_batch):
            """Apply val_transforms across a batch."""
            example_batch["pixel_values"] = [
                pipline(pil_img.convert("RGB")) for pil_img in example_batch["image"]
            ]
            del example_batch["image"]
            return example_batch

    return train_transforms, eval_transforms
