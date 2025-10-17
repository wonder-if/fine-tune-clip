import logging

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from torch.utils.data import DataLoader

from clip_tuner import (
    build_cached_feature_dataset,
    build_zero_shot_dataset,
    load_model,
)


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="config",
)
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    model, tokenizer, processor = load_model(
        **cfg.models_info,
        **cfg.pretrained_model,
    )

    for param in model.parameters():
        param.requires_grad_(False)

    zero_shot_dataset, zero_shot_collator = build_zero_shot_dataset(
        cfg.datasets_info,
        cfg.dataset,
        processor,
    )

    trainer = instantiate(
        cfg.trainer,
        model=model,
        zero_shot_dataset=zero_shot_dataset,
        zero_shot_collator=zero_shot_collator,
        clip_tokenizer=tokenizer,
        train_dataset=None,
        eval_dataset=None,
        data_collator=None,
    )

    model = model.to(trainer.args.device)

    feature_dataset, feature_collator = build_cached_feature_dataset(
        model=model,
        tokenizer=tokenizer,
        datasets_info_cfg=cfg.datasets_info,
        dataset_cfg=cfg.dataset,
        processor=processor,
        cache_root=cfg.get("feature_cache_root", "./.cache/features"),
        model_name=cfg.pretrained_model.model_name,
        device=model.device,
        normalize=True,
        max_samples=None,
        show_progress=True,
        include_text=True,
        prompt_template=cfg.trainer.prompt_template,
        force_refresh=cfg.get("force_cache_refresh", False),
    )

    logger.info(
        "Feature dataset ready with %d samples.",
        len(feature_dataset),
    )

    metrics = trainer.zero_shot_evaluate(
        feature_dataset,
        zero_shot_collator=feature_collator,
    )
    logger.info("Zero-shot metrics (from cache): %s", metrics)

    # downstream example: iterate cached feature batches
    feature_loader = DataLoader(
        feature_dataset,
        batch_size=cfg.trainer.args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=feature_collator,
    )
    batch = next(iter(feature_loader))
    logger.info(
        "Feature batch shapes - features: %s, labels: %s",
        tuple(batch["features"].shape),
        None if batch["labels"] is None else tuple(batch["labels"].shape),
    )


if __name__ == "__main__":
    main()
