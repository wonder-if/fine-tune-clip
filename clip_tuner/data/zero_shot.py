import os
from typing import Callable
import logging
from hydra.utils import instantiate
import torch
from torchvision.transforms.functional import InterpolationMode
from .loading import load_dataset

logger = logging.getLogger(__name__)


def build_pixel_mapping(image_processor):

    def _image2pixel(example_batch):
        # image processing
        example_batch["pixel_values"] = image_processor(
            example_batch["image"], return_tensors="pt"
        )["pixel_values"]

        del example_batch["image"]
        return example_batch

    return _image2pixel


def zero_shot_collator(examples):
    pixel_values = torch.stack(
        [torch.tensor(example["pixel_values"]) for example in examples]
    )
    labels = torch.tensor([example["label"] for example in examples], dtype=torch.long)

    return {
        "pixel_values": pixel_values,
        "labels": labels,
    }


def build_zero_shot_dataset(
    dataset_info_cfg,
    dataset_cfg,
    processor,
    cache_path="./.cache",
):
    cache_file_name = os.path.join(
        cache_path,
        dataset_cfg.zero_shot.dataset_name,
        dataset_cfg.zero_shot.domain_name + ".arrow",
    )

    logger.info(f"cache file name: {cache_file_name}")

    # load dataset
    data_manager = instantiate(dataset_info_cfg)
    dataset = instantiate(
        dataset_cfg.zero_shot,
        data_manager=data_manager,
        _target_=load_dataset,
    )

    zero_shot_mapping = build_pixel_mapping(
        image_processor=processor.image_processor,
    )

    dataset = dataset.map(
        function=zero_shot_mapping,
        batched=True,
        desc="Running image processing on zero-shot dataset",
        num_proc=4,
        cache_file_name=cache_file_name,
    )

    logger.info("zero-shot dataset")
    logger.info(dataset)

    return dataset, zero_shot_collator
