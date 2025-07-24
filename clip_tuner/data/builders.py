import logging
from hydra.utils import instantiate

from .loading import load_dataset
from .transforms import build_image_transforms
from .prompts import build_sample_prompt_encoder, build_class_prompt_encoder
from .collators import train_collator, zero_shot_collator
from ..utils import visualize_dataset_samples

from PIL import Image

logger = logging.getLogger(__name__)


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

    train_prompt_encoder = build_sample_prompt_encoder(
        prompts_cfg, clip_tokenizer=tokenizer, dataset=train_dataset
    )
    eval_prompt_encoder = build_sample_prompt_encoder(
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
    logger.info("train_dataset")
    logger.info(train_dataset)
    logger.info("eval_dataset")
    logger.info(eval_dataset)

    # load transforms and prompt encoders to dataset
    train_transforms, eval_transforms = build_image_transforms(
        transforms_cfg, image_processor=processor.image_processor
    )

    train_dataset.set_transform(train_transforms)
    eval_dataset.set_transform(eval_transforms)

    return train_dataset, eval_dataset, train_collator


def build_zero_shot_dataset(
    dataset_info_cfg,
    dataset_cfg,
):
    # load dataset
    data_manager = instantiate(dataset_info_cfg)
    dataset = instantiate(
        dataset_cfg.zero_shot,
        data_manager=data_manager,
        _target_=load_dataset,
    )

    logger.info("zero-shot dataset")
    logger.info(dataset)

    return dataset, zero_shot_collator
