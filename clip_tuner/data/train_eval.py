from typing import Callable
import logging
from hydra.utils import instantiate
import torch
from torchvision.transforms.functional import InterpolationMode
from .loading import load_dataset

logger = logging.getLogger(__name__)


def build_image_transforms(transform_cfg, image_processor=None) -> Callable:

    def fix_interp(cfg):
        if hasattr(cfg, "interpolation") and isinstance(cfg.interpolation, str):
            cfg.interpolation = InterpolationMode[cfg.interpolation]

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

        def eval_transforms(example_batch):
            """Apply val_transforms across a batch."""
            example_batch["pixel_values"] = [
                pipline(pil_img.convert("RGB")) for pil_img in example_batch["image"]
            ]
            del example_batch["image"]
            return example_batch

    return train_transforms, eval_transforms


def build_pixel_prompt_mapping(
    prompt_cfg,
    clip_tokenizer,
    dataset,
):
    labels = dataset.features["label"]
    prompt_template = prompt_cfg.prompt_template

    def _tokenize_labels(example_batch):

        prompt_with_label = [
            prompt_template.format(labels.int2str(label))
            for label in example_batch["label"]
        ]
        text_inputs = clip_tokenizer(
            prompt_with_label,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )
        example_batch["input_ids"] = text_inputs["input_ids"]
        example_batch["attention_mask"] = text_inputs["attention_mask"]
        return example_batch

    return _tokenize_labels


def train_collator(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    input_ids = torch.tensor(
        [example["input_ids"] for example in examples], dtype=torch.long
    )
    attention_mask = torch.tensor(
        [example["attention_mask"] for example in examples], dtype=torch.long
    )
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "return_loss": True,
    }


def build_train_eval_dataset(
    dataset_info_cfg,
    dataset_cfg,
    transforms_cfg,
    prompts_cfg,
    processor,
    tokenizer,
):
    # load dataset
    data_manager = instantiate(dataset_info_cfg)

    train_dataset = instantiate(
        dataset_cfg.train,
        data_manager=data_manager,
        _target_=load_dataset,
    )

    eval_dataset = instantiate(
        dataset_cfg.eval,
        data_manager=data_manager,
        _target_=load_dataset,
    )

    train_prompt_encoder = build_pixel_prompt_mapping(
        prompts_cfg, clip_tokenizer=tokenizer, dataset=train_dataset
    )
    eval_prompt_encoder = build_pixel_prompt_mapping(
        prompts_cfg, clip_tokenizer=tokenizer, dataset=eval_dataset
    )

    # visualize_dataset_samples(train_dataset)

    train_dataset = train_dataset.map(
        function=train_prompt_encoder,
        batched=True,
        desc="Running sample prompt and tokenizer on train dataset",
    )

    eval_dataset = eval_dataset.map(
        function=eval_prompt_encoder,
        batched=True,
        desc="Running sample prompt and tokenizer on eval dataset",
    )

    # load transforms and prompt encoders to dataset
    train_transforms, eval_transforms = build_image_transforms(
        transforms_cfg, image_processor=processor.image_processor
    )

    train_dataset.set_transform(train_transforms)
    eval_dataset.set_transform(eval_transforms)

    logger.info(f"train_dataset{train_dataset}")
    logger.info(f"eval_dataset{eval_dataset}")

    return train_dataset, eval_dataset, train_collator
