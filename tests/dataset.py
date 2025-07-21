import os
import logging
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from pprint import pprint

from clip_tuner import (
    load_dataset,
    load_model,
    build_image_transforms,
    build_sample_prompt_encoder,
    build_class_prompt_encoder,
    Base,
)


@hydra.main(
    version_base=None, config_path="pkg://clip_tuner/configs", config_name="config"
)
def main(cfg: DictConfig):
    print("Hydra run dir:", os.getcwd())
    logger = logging.getLogger(__name__)

    # load dataset
    dataset = instantiate(
        cfg.datasets_info,
        **cfg.dataset,
        _target_=load_dataset,
    )
    logger.info(dataset)

    # load model
    clip_model, clip_tokenizer, clip_processor = instantiate(
        cfg.models_info, **cfg.model, _target_=load_model
    )
    logger.info(f"clip_model: {clip_model}")
    logger.info(f"clip_tokenizer: {clip_tokenizer}")
    logger.info(f"clip_processor: {clip_processor}")

    # load transforms and prompt encoders to dataset
    train_transforms, val_transforms = build_image_transforms(
        cfg.image_transforms, image_processor=clip_processor.image_processor
    )
    sample_prompt_encoder = build_sample_prompt_encoder(
        cfg.sample_prompt_encoder, clip_tokenizer=clip_tokenizer, dataset=dataset
    )

    dataset.set_transforms(train_transforms)
    dataset.map(
        function=sample_prompt_encoder,
        batched=True,
        desc="Running sample prompt and tokenizer on dataset",
    )

    #

    #


if __name__ == "__main__":
    main()
