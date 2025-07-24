import os
import logging
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from pprint import pprint

from clip_tuner import (
    build_train_eval_dataset,
    build_zero_shot_dataset,
    load_model,
)


@hydra.main(
    version_base=None, config_path="pkg://clip_tuner/configs", config_name="config"
)
def main(cfg: DictConfig):
    print("Hydra run dir:", os.getcwd())
    logger = logging.getLogger(__name__)

    # load model
    model, tokenizer, processor = load_model(
        **cfg.models_info,
        **cfg.pretrained_model,
    )

    # build dataset and data collect_fn
    train_dataset, eval_dataset, collator = build_train_eval_dataset(
        cfg.datasets_info,  # 已管理的所有数据集的信息
        cfg.dataset,  # 选择的数据集
        cfg.transforms,  # 数据增强
        cfg.prompts,  # 提示词
        processor,  # 匹配模型的数据预处理
        tokenizer,  # tokenize
    )

    zero_shot_dataset, zero_shot_collator = build_zero_shot_dataset(
        cfg.datasets_info,
        cfg.dataset,
    )

    trainer = instantiate(
        cfg.trainer,
        model=model,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        zero_shot_dataset=zero_shot_dataset,
        zero_shot_collator=zero_shot_collator,
        processor=processor,
    )

    print("Trainer:", trainer)


if __name__ == "__main__":
    main()
